#!/usr/bin/env bash
# deploy/run_acme_cse.sh
#
# Installs (if needed) and starts a real, standard-compliant oneM2M CSE
# -- ACME (https://acmecse.net, pip install acmecse) -- for
# middleware/onem2m_http.py to register against over HTTP. This is the
# interoperable mode the ESTIMED hackathon evaluates (see README.md's
# "Phase 8" section): tinyIoT or Mobius work as drop-in alternatives,
# since HttpCSEClient speaks the standard TS-0004 HTTP binding, not
# anything ACME-specific -- only base_url/cse_id would need to change.
#
# ACME needs a non-interactive config file to run with --headless
# (its normal onboarding wizard is interactive). This script generates
# one automatically from the package's own default template plus the
# [basic.config] values ACME's onboarding wizard would otherwise ask
# for interactively, then starts the CSE with CoAP/MQTT/WS/remote-CSE
# bindings disabled (HTTP-only, standalone IN-CSE -- everything this
# project's middleware/onem2m_http.py actually exercises).
#
# Usage:
#   ./deploy/run_acme_cse.sh [runtime-dir] [http-port]
#
# Then point the pipeline at it:
#   EdgeToMecPipeline(cse_http_url="http://127.0.0.1:<http-port>")

set -euo pipefail

RUNTIME_DIR="${1:-$HOME/acme_cse_runtime}"
HTTP_PORT="${2:-8080}"

if ! python3 -c "import acmecse" 2>/dev/null; then
    echo "Installing acmecse..."
    python3 -m pip install acmecse --break-system-packages 2>/dev/null \
        || python3 -m pip install acmecse
fi

mkdir -p "$RUNTIME_DIR"
INI_PATH="$RUNTIME_DIR/acme.ini"

if [ ! -f "$INI_PATH" ]; then
    DEFAULT_INI="$(python3 -c "import acmecse, os; print(os.path.join(os.path.dirname(acmecse.__file__), 'init', 'acme.ini.default'))")"
    cp "$DEFAULT_INI" "$INI_PATH"

    # Inject the [basic.config] section the interactive onboarding
    # wizard would normally collect (CSE type/ID/name, http port, db
    # backend) -- see README.md's Phase 8 section for why this is
    # needed to run headless.
    python3 - "$INI_PATH" "$HTTP_PORT" <<'PY'
import sys
path, http_port = sys.argv[1], sys.argv[2]
with open(path) as f:
    content = f.read()

basic_config = f"""
[basic.config]
adminID=CAdmin
cseHost=127.0.0.1
cseID=id-in
cseName=cse-in
cseType=IN
databaseType=memory
httpPort={http_port}

"""
marker = "[cse]"
content = content.replace(marker, basic_config + marker, 1)
with open(path, "w") as f:
    f.write(content)
PY
    echo "Generated headless config at $INI_PATH"
fi

echo "Starting ACME oneM2M CSE (HTTP-only, IN-CSE, in-memory DB) on port $HTTP_PORT..."
echo "CSEBase will be reachable at: http://127.0.0.1:$HTTP_PORT/id-in"
exec python3 -m acmecse --headless --no-coap --no-mqtt --no-ws --no-remote-cse \
    --db-type memory --http-port "$HTTP_PORT" --log-level warn -dir "$RUNTIME_DIR"
