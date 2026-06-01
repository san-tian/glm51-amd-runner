#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  download_tinker_adapter_archive.sh --tinker-url URL --name NAME [options]

Options:
  --adapter-root PATH         adapter/archive root. Default: /data0/glm51_adapters
  --adapter-dir PATH          extracted adapter dir. Default: <adapter-root>/<name>
  --archive-path PATH         archive file. Default: <adapter-root>/<name>.tar.gz
  --force                    re-download and re-extract even if adapter files exist

Environment:
  GPU_LEASE_BASE_URL          default: https://eval-service.macaron.im
  GPU_LEASE_API_KEY           required; provide from env or secret store

The script converts tinker:// to a signed HTTP(S) archive URL through the GPU
Lease Manager async jobs API, downloads the archive, and extracts only adapter
and metadata files.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TINKER_URL=""
NAME=""
ADAPTER_ROOT="${GLM5_ADAPTER_ROOT:-/data0/glm51_adapters}"
ADAPTER_DIR=""
ARCHIVE_PATH=""
FORCE="0"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --tinker-url) TINKER_URL="${2:?}"; shift 2 ;;
    --name) NAME="${2:?}"; shift 2 ;;
    --adapter-root) ADAPTER_ROOT="${2:?}"; shift 2 ;;
    --adapter-dir) ADAPTER_DIR="${2:?}"; shift 2 ;;
    --archive-path) ARCHIVE_PATH="${2:?}"; shift 2 ;;
    --force) FORCE="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [ -z "$TINKER_URL" ] || [ -z "$NAME" ]; then
  usage
  exit 2
fi
if [[ "$TINKER_URL" != tinker://*/weights/* ]]; then
  echo "bad --tinker-url, expected tinker://<run-id>/weights/<name>: $TINKER_URL" >&2
  exit 2
fi

ADAPTER_DIR="${ADAPTER_DIR:-$ADAPTER_ROOT/$NAME}"
ARCHIVE_PATH="${ARCHIVE_PATH:-$ADAPTER_ROOT/$NAME.tar.gz}"

adapter_ready() {
  [ -f "$ADAPTER_DIR/adapter_config.json" ] || return 1
  if [ -f "$ADAPTER_DIR/adapter_model.safetensors" ]; then
    return 0
  fi
  local shard_count
  shard_count="$(find "$ADAPTER_DIR" -maxdepth 1 -type f -name '*_adapter.pt' 2>/dev/null | wc -l | tr -d ' ')"
  [ "${shard_count:-0}" -gt 0 ]
}

if [ "$FORCE" != "1" ] && adapter_ready; then
  echo "Adapter already exists: $ADAPTER_DIR"
  echo "$ADAPTER_DIR"
  exit 0
fi

export GPU_LEASE_BASE_URL="${GPU_LEASE_BASE_URL:-https://eval-service.macaron.im}"
export GPU_LEASE_API_KEY="${GPU_LEASE_API_KEY:-}"

command -v curl >/dev/null || { echo "curl not found" >&2; exit 1; }
command -v tar >/dev/null || { echo "tar not found" >&2; exit 1; }
mkdir -p "$ADAPTER_ROOT"

CONVERT_LOG="$(mktemp /tmp/tinker-http-archive.XXXXXX)"
cleanup() {
  rm -f "$CONVERT_LOG"
}
trap cleanup EXIT

python3 "$SCRIPT_DIR/tinker_to_http_archive.py" "$TINKER_URL" | tee "$CONVERT_LOG"
DOWNLOAD_URL="$(python3 - "$CONVERT_LOG" <<'PY'
import json
from pathlib import Path
import sys

text = Path(sys.argv[1]).read_text()
start = text.rfind("{")
if start < 0:
    raise SystemExit("converter output did not contain a JSON summary")
summary = json.loads(text[start:])
url = summary.get("download_url")
if not isinstance(url, str) or not url.startswith(("http://", "https://")):
    raise SystemExit("converter JSON did not contain an HTTP(S) download_url")
print(url)
PY
)"

echo "Downloading adapter archive to $ARCHIVE_PATH"
curl -L --fail --retry 5 --retry-delay 5 -C - \
  -o "$ARCHIVE_PATH" \
  "$DOWNLOAD_URL"

echo "Archive preview:"
tar -tf "$ARCHIVE_PATH" | sed -n '1,80p'

rm -rf "$ADAPTER_DIR"
mkdir -p "$ADAPTER_DIR"

tar -xzf "$ARCHIVE_PATH" \
  -C "$ADAPTER_ROOT" \
  --wildcards \
  "$NAME/*_adapter.pt" \
  "$NAME/adapter_config.json" \
  "$NAME/metadata.json" \
  "$NAME/training_meta.json"

if [ ! -f "$ADAPTER_DIR/adapter_config.json" ]; then
  echo "missing extracted adapter_config.json under $ADAPTER_DIR" >&2
  exit 1
fi

SHARD_COUNT="$(find "$ADAPTER_DIR" -maxdepth 1 -type f -name '*_adapter.pt' | wc -l | tr -d ' ')"
if [ ! -f "$ADAPTER_DIR/adapter_model.safetensors" ] && [ "${SHARD_COUNT:-0}" -ne 32 ]; then
  echo "expected 32 adapter shards under $ADAPTER_DIR, got ${SHARD_COUNT:-0}" >&2
  exit 1
fi

echo "adapter_shard_count=$SHARD_COUNT"
du -sh "$ADAPTER_DIR"
echo "$ADAPTER_DIR"
