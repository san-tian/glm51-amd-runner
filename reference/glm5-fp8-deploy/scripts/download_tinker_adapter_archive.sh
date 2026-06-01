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
  --force                    re-extract even if adapter files exist; reuse a readable archive when present

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

archive_ready() {
  [ -s "$ARCHIVE_PATH" ] || return 1
  tar -tzf "$ARCHIVE_PATH" >/dev/null 2>&1
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

WORK_TMP_ROOT="$ADAPTER_ROOT/.tmp"
mkdir -p "$WORK_TMP_ROOT"
SAFE_NAME="$(printf '%s' "$NAME" | tr -c 'A-Za-z0-9_.-' '_')"
[ -n "$SAFE_NAME" ] || SAFE_NAME="adapter"
CONVERT_LOG="$(mktemp "$WORK_TMP_ROOT/tinker-http-archive.XXXXXX")"
TAR_LIST="$(mktemp "$WORK_TMP_ROOT/tinker-archive-list.XXXXXX")"
MEMBER_LIST="$(mktemp "$WORK_TMP_ROOT/tinker-archive-members.XXXXXX")"
STAGE_DIR=""
FINAL_TMP=""
cleanup() {
  rm -f "$CONVERT_LOG" "$TAR_LIST" "$MEMBER_LIST"
  [ -z "${STAGE_DIR:-}" ] || rm -rf "$STAGE_DIR"
  [ -z "${FINAL_TMP:-}" ] || rm -rf "$FINAL_TMP"
}
trap cleanup EXIT

if archive_ready; then
  echo "Using existing readable adapter archive: $ARCHIVE_PATH"
else
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
fi

echo "Archive preview:"
tar -tf "$ARCHIVE_PATH" | sed -n '1,80p'

tar -tzf "$ARCHIVE_PATH" > "$TAR_LIST"
ARCHIVE_ADAPTER_PREFIX="$(python3 - "$NAME" "$TAR_LIST" "$MEMBER_LIST" <<'PY'
from pathlib import Path
import posixpath
import sys

name = sys.argv[1].strip("/")
tar_list = Path(sys.argv[2])
member_list = Path(sys.argv[3])

def normalize(member: str) -> str:
    member = member.strip()
    while member.startswith("./"):
        member = member[2:]
    return member.rstrip("/")

entries = [(raw, normalize(raw)) for raw in tar_list.read_text().splitlines() if normalize(raw)]
for _raw, norm in entries:
    parts = norm.split("/")
    if norm.startswith("/") or any(part == ".." for part in parts):
        raise SystemExit(f"unsafe archive member path: {norm}")

prefixes = []
for _raw, norm in entries:
    if norm == "adapter_config.json":
        prefixes.append("")
    elif norm.endswith("/adapter_config.json"):
        prefixes.append(norm[: -len("/adapter_config.json")])

if not prefixes:
    raise SystemExit("archive does not contain adapter_config.json")

def score(prefix: str):
    base = posixpath.basename(prefix)
    if prefix == name:
        rank = 0
    elif base == name:
        rank = 1
    else:
        rank = 2
    return (rank, len(prefix), prefix)

prefix = sorted(set(prefixes), key=score)[0]
wanted = {"adapter_config.json", "adapter_model.safetensors", "metadata.json", "training_meta.json"}
selected = []
for raw, norm in entries:
    if prefix:
        if norm == prefix:
            continue
        if not norm.startswith(prefix + "/"):
            continue
        rel = norm[len(prefix) + 1 :]
    else:
        rel = norm
    if not rel or "/" in rel:
        continue
    base = posixpath.basename(rel)
    if base in wanted or base.endswith("_adapter.pt"):
        selected.append(raw)

if "adapter_config.json" not in {posixpath.basename(normalize(raw)) for raw in selected}:
    raise SystemExit("adapter_config.json was not selected for extraction")
if not any(posixpath.basename(normalize(raw)).endswith("_adapter.pt") or posixpath.basename(normalize(raw)) == "adapter_model.safetensors" for raw in selected):
    raise SystemExit("archive contains no adapter_model.safetensors or *_adapter.pt files")

member_list.write_text("".join(f"{member}\n" for member in selected))
print(prefix)
PY
)"

if [ -n "$ARCHIVE_ADAPTER_PREFIX" ]; then
  echo "Archive adapter directory: $ARCHIVE_ADAPTER_PREFIX"
else
  echo "Archive adapter directory: <archive root>"
fi
echo "Selected adapter members:"
sed -n '1,80p' "$MEMBER_LIST"

mkdir -p "$(dirname "$ADAPTER_DIR")"
STAGE_DIR="$(mktemp -d "$WORK_TMP_ROOT/extract.${SAFE_NAME}.XXXXXX")"
tar -xzf "$ARCHIVE_PATH" -C "$STAGE_DIR" -T "$MEMBER_LIST"

if [ -n "$ARCHIVE_ADAPTER_PREFIX" ]; then
  EXTRACTED_DIR="$STAGE_DIR/$ARCHIVE_ADAPTER_PREFIX"
else
  EXTRACTED_DIR="$STAGE_DIR"
fi
if [ ! -d "$EXTRACTED_DIR" ]; then
  echo "missing extracted adapter directory under $EXTRACTED_DIR" >&2
  exit 1
fi

FINAL_TMP="$(dirname "$ADAPTER_DIR")/.${SAFE_NAME}.adapter.$$"
rm -rf "$FINAL_TMP"
mv "$EXTRACTED_DIR" "$FINAL_TMP"

if [ ! -f "$FINAL_TMP/adapter_config.json" ]; then
  echo "missing extracted adapter_config.json under $FINAL_TMP" >&2
  exit 1
fi

SHARD_COUNT="$(find "$FINAL_TMP" -maxdepth 1 -type f -name '*_adapter.pt' | wc -l | tr -d ' ')"
if [ ! -f "$FINAL_TMP/adapter_model.safetensors" ] && [ "${SHARD_COUNT:-0}" -ne 32 ]; then
  echo "expected 32 adapter shards under $FINAL_TMP, got ${SHARD_COUNT:-0}" >&2
  exit 1
fi

rm -rf "$ADAPTER_DIR"
mv "$FINAL_TMP" "$ADAPTER_DIR"
FINAL_TMP=""

echo "adapter_shard_count=$SHARD_COUNT"
du -sh "$ADAPTER_DIR"
echo "$ADAPTER_DIR"
