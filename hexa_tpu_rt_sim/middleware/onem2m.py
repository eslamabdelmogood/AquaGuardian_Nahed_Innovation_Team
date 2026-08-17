"""
middleware/onem2m.py
=====================
Standardization & middleware layer, modeled on the oneM2M architecture
(TS-0001 Functional Architecture, TS-0004 Service Layer Core Protocol).

This is a software simulation of the oneM2M resource tree and its
Create/Retrieve/Update/Delete/Notify (CRUD+N) primitives -- it does not
open real sockets or implement a wire protocol (mqtt/coap/http
bindings), but every resource type, addressing scheme, operation code,
and response status code below follows the standard's naming and
semantics so the shapes are drop-in compatible with a real oneM2M
stack if this simulator is ever pointed at one (e.g. via an mcabase
CoAP/HTTP binding module dropped in behind `MN_CSE.handle_primitive`).

Roles (TS-0001 section 6.3, node types):
  ADN-AE  -- Application Dedicated Node - Application Entity. Runs on
             constrained edge hardware with no local CSE of its own
             (that's what makes it an "Application Dedicated" node, as
             opposed to an ASN which hosts its own CSE). This is
             HEXA-TPU-RT's edge hardware: it originates raw sensor
             streams and reflex event signals but has no local
             persistence/routing -- it registers with, and is entirely
             dependent on, a remote CSE.
  MN-CSE  -- Middle Node - Common Services Entity. Sits between the
             constrained edge and the infrastructure (IN-CSE) tier.
             Hosts the actual resource tree (<CSEBase>/<AE>/<container>/
             <contentInstance>), handles the ADN-AE's registration and
             every subsequent primitive, and is the interoperability
             boundary: any standard-compliant oneM2M client (not just
             this project's own MEC layer) can retrieve BDO-SKIN data
             from here without knowing anything about HEXA-TPU-RT.

Resource tree shape used here (TS-0001 section 9.6):

    <CSEBase>
      L <AE> (ADN-AE, one per registered edge node)
          L <container> "sensorData"      -- raw FBG sensor windows
          |   L <contentInstance> ...
          L <container> "reflexEvents"    -- reflex-layer event signals
              L <contentInstance> ...

Not modeled (out of scope for a simulator): security (TS-0003
access-control-policy resources), discovery, and the actual
mcabase transport binding -- those are protocol-binding /
network-layer concerns, not architecture-layer ones.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------
# oneM2M operation codes (TS-0004 Table 6.3.2.1) and response status
# codes (TS-0004 Table 6.3.4.2.1-1) -- the subset this simulator needs.
# --------------------------------------------------------------------

class Operation(IntEnum):
    CREATE = 1
    RETRIEVE = 2
    UPDATE = 3
    DELETE = 4
    NOTIFY = 5


class ResponseStatusCode(IntEnum):
    OK = 2000
    CREATED = 2001
    UPDATED = 2004
    DELETED = 2002
    BAD_REQUEST = 4000
    NOT_FOUND = 4004
    ALREADY_EXISTS = 4105
    INTERNAL_SERVER_ERROR = 5000


class ResourceType(IntEnum):
    # Numeric values follow TS-0001 Table 9.6.1.3-1 (subset used here).
    AE = 2
    CONTAINER = 3
    CONTENT_INSTANCE = 4
    CSE_BASE = 5
    SUBSCRIPTION = 23


# --------------------------------------------------------------------
# Resource tree
# --------------------------------------------------------------------

def _resource_id() -> str:
    return uuid.uuid4().hex[:12]


def _now_iso() -> str:
    return time.strftime("%Y%m%dT%H%M%S", time.gmtime())


@dataclass
class Resource:
    resource_type: ResourceType
    resource_name: str
    parent_id: Optional[str] = None
    resource_id: str = field(default_factory=_resource_id)
    creation_time: str = field(default_factory=_now_iso)
    last_modified_time: str = field(default_factory=_now_iso)
    labels: List[str] = field(default_factory=list)
    children: Dict[str, str] = field(default_factory=dict)   # name -> resource_id
    content: Any = None                                       # payload, for leaf resources

    def to_primitive_content(self) -> Dict[str, Any]:
        """The subset of a oneM2M resource representation a
        responsePrimitive would carry -- short-names mirror TS-0004
        Annex C (e.g. 'ri' = resourceID, 'rn' = resourceName, 'ty' =
        resourceType, 'pi' = parentID, 'ct'/'lt' = creation/modified
        time, 'con' = content for a contentInstance)."""
        d = {
            "ty": int(self.resource_type),
            "ri": self.resource_id,
            "rn": self.resource_name,
            "pi": self.parent_id,
            "ct": self.creation_time,
            "lt": self.last_modified_time,
        }
        if self.content is not None:
            d["con"] = self.content
        return d


@dataclass
class RequestPrimitive:
    """Standard oneM2M requestPrimitive (TS-0004 Table 6.3.1-1),
    trimmed to the short-names this simulator exercises."""
    operation: Operation
    to: str                      # target resource address (path)
    fr: str                      # originator (AE-ID or CSE-ID)
    request_id: str = field(default_factory=_resource_id)
    resource_type: Optional[ResourceType] = None
    content: Any = None


@dataclass
class ResponsePrimitive:
    request_id: str
    response_status_code: ResponseStatusCode
    content: Any = None

    @property
    def ok(self) -> bool:
        return self.response_status_code < 3000


class OneM2MError(Exception):
    def __init__(self, rsc: ResponseStatusCode, message: str):
        super().__init__(f"[{int(rsc)}] {message}")
        self.rsc = rsc


class MN_CSE:
    """Middle Node - Common Services Entity. Hosts the resource tree
    rooted at <CSEBase> and processes standard CRUD/Notify primitives
    from registered ADN-AE originators."""

    def __init__(self, cse_id: str = "mn-cse-hexa"):
        self.cse_id = cse_id
        self._resources: Dict[str, Resource] = {}
        self.base = Resource(ResourceType.CSE_BASE, cse_id, parent_id=None)
        self._resources[self.base.resource_id] = self.base
        self._registered_aes: Dict[str, str] = {}   # ae_id -> resource_id
        self.request_log: List[Dict[str, Any]] = []  # standards-compliant audit trail

    # -- addressing ----------------------------------------------------
    def _resolve_path(self, path: str) -> Optional[Resource]:
        """Resolve a '/'-separated structured resourceName path rooted
        at CSEBase, e.g. 'hexa-tpu-edge-01/sensorData'."""
        parts = [p for p in path.split("/") if p]
        node = self.base
        for part in parts:
            child_id = node.children.get(part)
            if child_id is None:
                return None
            node = self._resources[child_id]
        return node

    def _create_child(self, parent: Resource, rtype: ResourceType,
                       rname: str, content: Any = None) -> Resource:
        r = Resource(rtype, rname, parent_id=parent.resource_id, content=content)
        self._resources[r.resource_id] = r
        parent.children[rname] = r.resource_id
        parent.last_modified_time = _now_iso()
        return r

    # -- the single standard entry point, mirroring a real CSE's mcc/
    # mca service access point ------------------------------------------
    def handle_primitive(self, req: RequestPrimitive) -> ResponsePrimitive:
        self.request_log.append({"op": req.operation.name, "to": req.to, "fr": req.fr,
                                  "rqi": req.request_id, "ts": _now_iso()})
        try:
            if req.operation == Operation.CREATE:
                return self._handle_create(req)
            if req.operation == Operation.RETRIEVE:
                return self._handle_retrieve(req)
            if req.operation == Operation.NOTIFY:
                return self._handle_notify(req)
            raise OneM2MError(ResponseStatusCode.BAD_REQUEST,
                               f"unsupported operation {req.operation}")
        except OneM2MError as e:
            return ResponsePrimitive(req.request_id, e.rsc, content={"error": str(e)})

    def _handle_create(self, req: RequestPrimitive) -> ResponsePrimitive:
        parent = self._resolve_path(req.to) if req.to else self.base
        if parent is None:
            raise OneM2MError(ResponseStatusCode.NOT_FOUND, f"parent '{req.to}' not found")

        if req.resource_type == ResourceType.AE:
            ae_name = req.content.get("rn") if isinstance(req.content, dict) else None
            if not ae_name:
                raise OneM2MError(ResponseStatusCode.BAD_REQUEST, "AE registration missing 'rn'")
            if ae_name in parent.children:
                raise OneM2MError(ResponseStatusCode.ALREADY_EXISTS,
                                   f"AE '{ae_name}' already registered")
            ae = self._create_child(parent, ResourceType.AE, ae_name, content=req.content)
            ae_id = f"C{ae.resource_id}"
            self._registered_aes[ae_id] = ae.resource_id
            resp_content = ae.to_primitive_content()
            resp_content["aei"] = ae_id       # assigned AE-ID, TS-0004 registration response
            return ResponsePrimitive(req.request_id, ResponseStatusCode.CREATED, resp_content)

        if req.resource_type == ResourceType.CONTAINER:
            cname = req.content.get("rn") if isinstance(req.content, dict) else None
            if not cname:
                raise OneM2MError(ResponseStatusCode.BAD_REQUEST, "container missing 'rn'")
            if cname in parent.children:
                existing = self._resources[parent.children[cname]]
                return ResponsePrimitive(req.request_id, ResponseStatusCode.OK,
                                          existing.to_primitive_content())
            c = self._create_child(parent, ResourceType.CONTAINER, cname)
            return ResponsePrimitive(req.request_id, ResponseStatusCode.CREATED,
                                      c.to_primitive_content())

        if req.resource_type == ResourceType.CONTENT_INSTANCE:
            if parent.resource_type != ResourceType.CONTAINER:
                raise OneM2MError(ResponseStatusCode.BAD_REQUEST,
                                   "contentInstance parent must be a container")
            ci_name = f"ci_{len(parent.children):06d}"
            ci = self._create_child(parent, ResourceType.CONTENT_INSTANCE, ci_name,
                                     content=req.content)
            return ResponsePrimitive(req.request_id, ResponseStatusCode.CREATED,
                                      ci.to_primitive_content())

        raise OneM2MError(ResponseStatusCode.BAD_REQUEST,
                           f"unsupported resourceType {req.resource_type}")

    def _handle_retrieve(self, req: RequestPrimitive) -> ResponsePrimitive:
        r = self._resolve_path(req.to)
        if r is None:
            raise OneM2MError(ResponseStatusCode.NOT_FOUND, f"'{req.to}' not found")
        return ResponsePrimitive(req.request_id, ResponseStatusCode.OK, r.to_primitive_content())

    def _handle_notify(self, req: RequestPrimitive) -> ResponsePrimitive:
        # Subscriptions/notifications are out of scope for this
        # simulator's data path (MEC layer polls/retrieves directly);
        # accepted and acknowledged per the standard's async pattern.
        return ResponsePrimitive(req.request_id, ResponseStatusCode.OK, {"notified": True})

    def latest_content_instances(self, ae_name: str, container_name: str,
                                  limit: int = 1) -> List[Dict[str, Any]]:
        """Convenience accessor used by the MEC layer to pull the most
        recent contentInstances -- equivalent to a RETRIEVE with a
        filterCriteria (la/ol) in the real protocol."""
        container = self._resolve_path(f"{ae_name}/{container_name}")
        if container is None:
            return []
        ci_ids = list(container.children.values())[-limit:]
        return [self._resources[i].to_primitive_content() for i in ci_ids]


class ADN_AE:
    """Application Dedicated Node - Application Entity: the edge
    hardware's oneM2M identity. Registers itself with an MN-CSE, then
    encapsulates raw sensor streams and reflex event signals as
    contentInstances under its own <container> resources -- the
    standard-compliant interoperability boundary the spec asks for:
    any oneM2M-conformant consumer downstream (this project's MEC
    layer, or a real IN-CSE) can retrieve this data without any
    HEXA-TPU-RT-specific knowledge."""

    SENSOR_CONTAINER = "sensorData"
    REFLEX_CONTAINER = "reflexEvents"

    def __init__(self, app_name: str, cse: MN_CSE, app_type: str = "NBDO-SKIN-edge"):
        self.app_name = app_name
        self.cse = cse
        self.app_type = app_type
        self.ae_id: Optional[str] = None
        self._registered = False

    def register(self) -> ResponsePrimitive:
        """AE registration primitive (TS-0004 section 7.3.2.1): the
        constrained edge node announces itself to the CSE before it
        can create any child resources."""
        req = RequestPrimitive(
            operation=Operation.CREATE, to="", fr=self.app_name,
            resource_type=ResourceType.AE,
            content={"rn": self.app_name, "api": self.app_type, "rr": True},
        )
        resp = self.cse.handle_primitive(req)
        if resp.ok:
            self.ae_id = resp.content["aei"]
            self._registered = True
            # Pre-create the two data containers this ADN-AE will feed.
            for cname in (self.SENSOR_CONTAINER, self.REFLEX_CONTAINER):
                self.cse.handle_primitive(RequestPrimitive(
                    operation=Operation.CREATE, to=self.app_name, fr=self.ae_id,
                    resource_type=ResourceType.CONTAINER, content={"rn": cname},
                ))
        return resp

    def _require_registered(self):
        if not self._registered:
            raise OneM2MError(ResponseStatusCode.BAD_REQUEST,
                               f"ADN-AE '{self.app_name}' is not registered with an MN-CSE")

    def push_sensor_window(self, window) -> ResponsePrimitive:
        """Encapsulates one SensorWindow (models/sensor_events.py) as a
        oneM2M contentInstance -- the raw sensor stream, standardized
        into a JSON-serializable envelope any oneM2M consumer can read."""
        self._require_registered()
        payload = {
            "kind": "sensorWindow",
            "index": window.index,
            "scenario": window.scenario,
            "mean_temp_c": window.mean_temp_c,
            "mean_strain_kpa": window.mean_strain_kpa,
            "anomaly_active": window.anomaly_active,
            "anomaly_severity": window.anomaly_severity,
            "anomaly_cell": window.anomaly_cell,
            "is_critical": window.is_critical,
        }
        req = RequestPrimitive(
            operation=Operation.CREATE,
            to=f"{self.app_name}/{self.SENSOR_CONTAINER}",
            fr=self.ae_id, resource_type=ResourceType.CONTENT_INSTANCE,
            content=json.loads(json.dumps(payload)),   # normalize (tuples -> lists, per JSON)
        )
        return self.cse.handle_primitive(req)

    def push_reflex_event(self, window_index: int, channels_processed: int,
                           triggered: bool, detail: Optional[Dict[str, Any]] = None
                           ) -> ResponsePrimitive:
        """Encapsulates a reflex-layer event signal -- the low-latency
        always-on monitoring result computed at the edge, standardized
        the same way as the raw sensor stream."""
        self._require_registered()
        payload = {
            "kind": "reflexEvent",
            "window_index": window_index,
            "channels_processed": channels_processed,
            "triggered": triggered,
            "detail": detail or {},
        }
        req = RequestPrimitive(
            operation=Operation.CREATE,
            to=f"{self.app_name}/{self.REFLEX_CONTAINER}",
            fr=self.ae_id, resource_type=ResourceType.CONTENT_INSTANCE,
            content=payload,
        )
        return self.cse.handle_primitive(req)
