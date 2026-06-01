#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "serve_glm51_fp8_compile_mtp.sh is a compatibility wrapper; use serve_glm51_fp8.sh. Forwarding..." >&2
exec "$SCRIPT_DIR/serve_glm51_fp8.sh" "$@"
