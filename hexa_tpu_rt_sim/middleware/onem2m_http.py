"""
middleware/onem2m_http.py
===========================
Real HTTP binding for the Standardization & Middleware Layer, speaking
the actual oneM2M `mca`/`mcc` HTTP protocol binding (TS-0004 clause
8.2) to a live, standard-compliant CSE -- ACME, tinyIoT, or Mobius (the
three implementations the ESTIMED hackathon names explicitly), or any
other conformant CSE. `middleware/onem2m.py`'s in-memory `MN_CSE` is a
same-process simulation of the resource tree; this module replaces
that transport with real HTTP requests against a CSE process actually
running on the network, while keeping the exact same
`handle_primitive(RequestPrimitive) -> ResponsePrimitive` interface --
so `ADN_AE` (middleware/onem2m.py) works unchanged against either one.
Swap the transport `ADN_AE` is constructed with; nothing else in the
integration changes.

Verified against a locally-run ACME CSE (`pip install acmecse`,
`python -m acmecse --headless ...`) during development: AE
registration, container creation, and contentInstance push/retrieve
all round-trip correctly over real HTTP against that implementation.
tinyIoT and Mobius implement the same TS-0004 HTTP binding and should
work identically -- only `base_url`/`cse_id` need to change.

Design notes / deliberate simplifications:

  - **Addressing**: oneM2M supports both structured (name-path) and
    unstructured (resourceID-path) addressing. Structured addressing
    against a self-assigned (unregistered-originator) AE registration
    did not resolve reliably against the ACME instance tested here, so
    this client tracks each resource's assigned `ri` from its own
    CREATE responses and always addresses children by `ri` afterward
    -- robust across CSE implementations without depending on a
    particular structured-addressing quirk.
  - **Originator handling**: AE registration uses the oneM2M
    unregistered-originator convention (`fr="S"`, TS-0004 clause
    6.2.2) so the CSE assigns the AE-ID; every subsequent request
    under that AE's subtree then presents the assigned AE-ID as
    originator, per the standard's registration flow.
  - **`latest_content_instances`**: rather than depend on a specific
    CSE's discovery/child-resource-reference query-parameter dialect
    (which vendors implement slightly differently), this client keeps
    a local append-only mirror of what it has itself pushed, updated
    on every successful CREATE. The CREATE calls themselves are real
    network round-trips to the CSE (that's the interoperability
    surface this layer exists to exercise); this convenience accessor
    just avoids a second round-trip to re-fetch data this process
    already has in hand.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import requests

from middleware.onem2m import (
    Operation, ResourceType, ResponseStatusCode,
    RequestPrimitive, ResponsePrimitive, OneM2MError,
)


class HttpCSEClient:
    """Talks real oneM2M HTTP (TS-0004 clause 8.2) to a live CSE.
    Drop-in replacement for `middleware.onem2m.MN_CSE` behind
    `ADN_AE`: same `handle_primitive` signature and semantics."""

    RELEASE_VERSION = "4"        # X-M2M-RVI: oneM2M Release 4

    def __init__(self, base_url: str = "http://127.0.0.1:8080",
                 cse_id: str = "id-in", admin_originator: str = "CAdmin",
                 timeout_s: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.cse_id = cse_id
        self.admin_originator = admin_originator
        self.timeout_s = timeout_s

        self._ri_by_path: Dict[str, str] = {}          # "ae/container" -> ri
        self._originator_by_root: Dict[str, str] = {}  # ae rn -> assigned AE-ID
        self._content_log: Dict[str, List[Dict[str, Any]]] = {}  # path -> pushed contentInstances

        self.request_log: List[Dict[str, Any]] = []    # standards-compliant audit trail

    # -- HTTP plumbing ---------------------------------------------------
    def _headers(self, originator: str, request_id: str,
                 resource_type: Optional[ResourceType] = None, accept_json: bool = True
                 ) -> Dict[str, str]:
        h = {
            "X-M2M-Origin": originator,
            "X-M2M-RI": request_id,
            "X-M2M-RVI": self.RELEASE_VERSION,
        }
        if resource_type is not None:
            h["Content-Type"] = f"application/json;ty={int(resource_type)}"
        if accept_json:
            h["Accept"] = "application/json"
        return h

    def _cse_root_url(self) -> str:
        return f"{self.base_url}/{self.cse_id}"

    def _url_for_ri(self, ri: str) -> str:
        return f"{self.base_url}/{ri}"

    def _originator_for_path(self, path: str) -> str:
        root = path.split("/")[0] if path else ""
        return self._originator_by_root.get(root, self.admin_originator)

    def _resolve_ri(self, path: str) -> str:
        ri = self._ri_by_path.get(path)
        if ri is None:
            raise OneM2MError(ResponseStatusCode.NOT_FOUND,
                               f"'{path}' has no known resourceID -- was it created "
                               f"through this client?")
        return ri

    def _do_request(self, method: str, url: str, headers: Dict[str, str],
                     json_body: Optional[dict] = None) -> requests.Response:
        try:
            return requests.request(method, url, headers=headers, json=json_body,
                                     timeout=self.timeout_s)
        except requests.exceptions.ConnectionError as e:
            raise OneM2MError(
                ResponseStatusCode.INTERNAL_SERVER_ERROR,
                f"could not reach CSE at {self.base_url} -- is it running? "
                f"(pip install acmecse; python -m acmecse --headless "
                f"--no-coap --no-mqtt --no-ws --no-remote-cse --http-port 8080) ({e})",
            )
        except requests.exceptions.Timeout as e:
            raise OneM2MError(ResponseStatusCode.INTERNAL_SERVER_ERROR,
                               f"request to CSE at {url} timed out after {self.timeout_s}s ({e})")

    # -- the single standard entry point (matches MN_CSE) -----------------
    def handle_primitive(self, req: RequestPrimitive) -> ResponsePrimitive:
        self.request_log.append({"op": req.operation.name, "to": req.to, "fr": req.fr,
                                  "rqi": req.request_id})
        try:
            if req.operation == Operation.CREATE:
                return self._create(req)
            if req.operation == Operation.RETRIEVE:
                return self._retrieve(req)
            if req.operation == Operation.NOTIFY:
                return ResponsePrimitive(req.request_id, ResponseStatusCode.OK, {"notified": True})
            raise OneM2MError(ResponseStatusCode.BAD_REQUEST,
                               f"unsupported operation {req.operation}")
        except OneM2MError as e:
            return ResponsePrimitive(req.request_id, e.rsc, content={"error": str(e)})

    # -- CREATE ------------------------------------------------------------
    def _create(self, req: RequestPrimitive) -> ResponsePrimitive:
        if req.resource_type == ResourceType.AE:
            return self._create_ae(req)
        if req.resource_type == ResourceType.CONTAINER:
            return self._create_container(req)
        if req.resource_type == ResourceType.CONTENT_INSTANCE:
            return self._create_content_instance(req)
        raise OneM2MError(ResponseStatusCode.BAD_REQUEST,
                           f"unsupported resourceType {req.resource_type}")

    def _create_ae(self, req: RequestPrimitive) -> ResponsePrimitive:
        rn = req.content.get("rn") if isinstance(req.content, dict) else None
        if not rn:
            raise OneM2MError(ResponseStatusCode.BAD_REQUEST, "AE registration missing 'rn'")

        body = {"m2m:ae": {
            "rn": rn,
            "api": req.content.get("api", "Napp"),
            "rr": req.content.get("rr", True),
            "srv": [self.RELEASE_VERSION],
        }}
        # Unregistered-originator convention (TS-0004 6.2.2): "S" tells
        # the CSE to assign a fresh AE-ID for this registration.
        headers = self._headers("S", req.request_id, resource_type=ResourceType.AE)
        resp = self._do_request("POST", self._cse_root_url(), headers, body)

        if resp.status_code == 409 or resp.status_code == 400:
            # ACME reports "already exists" as a generic bad-request/dbg
            # message rather than a distinct HTTP status in some
            # versions; surface it as ALREADY_EXISTS either way since
            # that's the actionable, standard-shaped signal callers expect.
            raise OneM2MError(ResponseStatusCode.ALREADY_EXISTS,
                               f"AE '{rn}' registration rejected by CSE: {resp.text}")
        if resp.status_code >= 300:
            raise OneM2MError(ResponseStatusCode.BAD_REQUEST,
                               f"AE registration failed ({resp.status_code}): {resp.text}")

        ae = resp.json()["m2m:ae"]
        ri, aei = ae["ri"], ae["aei"]
        self._ri_by_path[rn] = ri
        self._originator_by_root[rn] = aei

        content = {"ty": int(ResourceType.AE), "ri": ri, "rn": ae["rn"], "pi": ae.get("pi"),
                   "ct": ae.get("ct"), "lt": ae.get("lt"), "aei": aei}
        return ResponsePrimitive(req.request_id, ResponseStatusCode.CREATED, content)

    def _create_container(self, req: RequestPrimitive) -> ResponsePrimitive:
        rn = req.content.get("rn") if isinstance(req.content, dict) else None
        if not rn:
            raise OneM2MError(ResponseStatusCode.BAD_REQUEST, "container missing 'rn'")

        full_path = f"{req.to}/{rn}"
        if full_path in self._ri_by_path:
            # Idempotent, like the in-memory MN_CSE: creating the same
            # container twice returns OK rather than an error.
            return ResponsePrimitive(req.request_id, ResponseStatusCode.OK,
                                      {"ri": self._ri_by_path[full_path], "rn": rn})

        parent_ri = self._resolve_ri(req.to)
        originator = self._originator_for_path(req.to)
        body = {"m2m:cnt": {"rn": rn}}
        headers = self._headers(originator, req.request_id, resource_type=ResourceType.CONTAINER)
        resp = self._do_request("POST", self._url_for_ri(parent_ri), headers, body)
        if resp.status_code >= 300:
            raise OneM2MError(ResponseStatusCode.BAD_REQUEST,
                               f"container creation failed ({resp.status_code}): {resp.text}")

        cnt = resp.json()["m2m:cnt"]
        self._ri_by_path[full_path] = cnt["ri"]
        self._content_log[full_path] = []
        content = {"ty": int(ResourceType.CONTAINER), "ri": cnt["ri"], "rn": cnt["rn"],
                   "pi": cnt.get("pi"), "ct": cnt.get("ct"), "lt": cnt.get("lt")}
        return ResponsePrimitive(req.request_id, ResponseStatusCode.CREATED, content)

    def _create_content_instance(self, req: RequestPrimitive) -> ResponsePrimitive:
        if req.to not in self._ri_by_path:
            raise OneM2MError(ResponseStatusCode.BAD_REQUEST,
                               f"contentInstance parent '{req.to}' is not a known container "
                               f"-- create it first")
        parent_ri = self._ri_by_path[req.to]
        originator = self._originator_for_path(req.to)
        body = {"m2m:cin": {"con": req.content}}
        headers = self._headers(originator, req.request_id, resource_type=ResourceType.CONTENT_INSTANCE)
        resp = self._do_request("POST", self._url_for_ri(parent_ri), headers, body)
        if resp.status_code >= 300:
            raise OneM2MError(ResponseStatusCode.BAD_REQUEST,
                               f"contentInstance creation failed ({resp.status_code}): {resp.text}")

        cin = resp.json()["m2m:cin"]
        content = {"ty": int(ResourceType.CONTENT_INSTANCE), "ri": cin["ri"], "rn": cin.get("rn"),
                   "pi": cin.get("pi"), "ct": cin.get("ct"), "lt": cin.get("lt"), "con": cin["con"]}
        self._content_log.setdefault(req.to, []).append(content)
        return ResponsePrimitive(req.request_id, ResponseStatusCode.CREATED, content)

    # -- RETRIEVE ------------------------------------------------------------
    def _retrieve(self, req: RequestPrimitive) -> ResponsePrimitive:
        ri = self._ri_by_path.get(req.to)
        if ri is None:
            raise OneM2MError(ResponseStatusCode.NOT_FOUND, f"'{req.to}' not found")
        originator = self._originator_for_path(req.to)
        headers = self._headers(originator, req.request_id)
        resp = self._do_request("GET", self._url_for_ri(ri), headers)
        if resp.status_code >= 300:
            raise OneM2MError(ResponseStatusCode.NOT_FOUND,
                               f"retrieve failed ({resp.status_code}): {resp.text}")
        # Unwrap whichever single "m2m:xxx" key the CSE returned.
        body = resp.json()
        payload = next(iter(body.values())) if body else {}
        return ResponsePrimitive(req.request_id, ResponseStatusCode.OK, payload)

    # -- convenience accessor matching MN_CSE's interface --------------------
    def latest_content_instances(self, ae_name: str, container_name: str,
                                  limit: int = 1) -> List[Dict[str, Any]]:
        """See module docstring: served from this client's own log of
        what it has pushed, to avoid depending on a CSE-specific
        discovery query dialect for a read this process already has
        the answer to locally."""
        path = f"{ae_name}/{container_name}"
        return list(self._content_log.get(path, []))[-limit:]

    def ping(self) -> bool:
        """Health check: True if the CSEBase responds at all. Useful
        for a caller to fail fast with a clear message instead of
        hitting a ConnectionError deep inside the first real primitive."""
        try:
            resp = requests.get(self._cse_root_url(),
                                 headers=self._headers(self.admin_originator, "ping-0"),
                                 timeout=self.timeout_s)
            return resp.status_code < 300
        except requests.exceptions.RequestException:
            return False
