#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  preflight_glm51_deploy.sh --tinker-url URL [options]
  preflight_glm51_deploy.sh --ssh "ssh ..." --tinker-url URL [options]

Options:
  --ssh CMD                   run this same preflight on a remote server over ssh
  --remote-root PATH          temporary remote skill directory. Default: /tmp/glm5-fp8-deploy-skill-...
  --backend auto|sglang|vllm  serve backend. Default: auto
  --tinker-url URL            tinker://.../weights/<name>
  --base PATH                 GLM-5.1 BF16 base snapshot. Default: auto-detect PFS/local cache
  --adapter-root PATH         HTTP archive adapter root when PFS adapter is unavailable. Default: /data0/glm51_adapters
  --min-root-tmp-gib N        minimum free GiB required on /root/tmp. Default: 2500
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORIGINAL_ARGS=("$@")
SSH_CMD=""
REMOTE_ROOT=""
BACKEND="auto"
TINKER_URL=""
BASE=""
ADAPTER_ROOT="${GLM5_ADAPTER_ROOT:-/data0/glm51_adapters}"
MIN_ROOT_TMP_GIB="2500"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --ssh) SSH_CMD="${2:?}"; shift 2 ;;
    --remote-root) REMOTE_ROOT="${2:?}"; shift 2 ;;
    --backend) BACKEND="${2:?}"; shift 2 ;;
    --tinker-url) TINKER_URL="${2:?}"; shift 2 ;;
    --base) BASE="${2:?}"; shift 2 ;;
    --adapter-root) ADAPTER_ROOT="${2:?}"; shift 2 ;;
    --min-root-tmp-gib) MIN_ROOT_TMP_GIB="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [ -n "$SSH_CMD" ]; then
  if [ -z "$REMOTE_ROOT" ]; then
    REMOTE_ROOT="/tmp/glm5-fp8-deploy-skill-${USER:-user}-preflight-$$"
  fi
  REMOTE_SCRIPT_DIR="$REMOTE_ROOT/scripts"
  # shellcheck disable=SC2086
  $SSH_CMD "rm -rf '$REMOTE_ROOT' && mkdir -p '$REMOTE_SCRIPT_DIR'"
  tar -C "$(dirname "$SCRIPT_DIR")" -cf - scripts | $SSH_CMD "tar -C '$REMOTE_ROOT' -xf -"
  FILTERED_ARGS=()
  skip_next=0
  for arg in "${ORIGINAL_ARGS[@]}"; do
    if [ "$skip_next" = "1" ]; then
      skip_next=0
      continue
    fi
    case "$arg" in
      --ssh|--remote-root) skip_next=1 ;;
      *) FILTERED_ARGS+=("$arg") ;;
    esac
  done
  printf -v REMOTE_ARG_STRING '%q ' "${FILTERED_ARGS[@]}"
  # shellcheck disable=SC2086
  exec $SSH_CMD "bash '$REMOTE_SCRIPT_DIR/preflight_glm51_deploy.sh' $REMOTE_ARG_STRING"
fi

if [ -z "$TINKER_URL" ]; then
  usage
  exit 2
fi
if [[ "$TINKER_URL" != tinker://*/weights/* ]]; then
  echo "bad --tinker-url, expected tinker://<run-id>/weights/<name>: $TINKER_URL" >&2
  exit 2
fi

TINKER_BODY="${TINKER_URL#tinker://}"
TINKER_RUN_ID="${TINKER_BODY%%/weights/*}"
TINKER_WEIGHT_PATH="${TINKER_BODY#*/weights/}"
TINKER_WEIGHT_NAME="${TINKER_WEIGHT_PATH##*/}"
PFS_LORA="/vePFS-Mindverse/share/tinker_runtime_checkpoints/persistent_cache/admin/${TINKER_RUN_ID}/${TINKER_WEIGHT_NAME}"
FALLBACK_LORA="$ADAPTER_ROOT/$TINKER_WEIGHT_NAME"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

check_path() {
  local path="$1"
  local desc="$2"
  [ -e "$path" ] || fail "missing ${desc}: ${path}"
  echo "OK: ${desc}: ${path}"
}

base_snapshot_ready() {
  local path="$1"
  [ -d "$path" ] && [ -f "$path/config.json" ] && [ -f "$path/model.safetensors.index.json" ]
}

detect_base_snapshot() {
  local candidate
  local root
  local candidates=()
  if [ -n "${GLM5_BASE:-}" ]; then
    candidates+=("$GLM5_BASE")
  fi
  candidates+=(
    "/vePFS-Mindverse/share/huggingface/hub/models--zai-org--GLM-5.1/snapshots/5c155b85c12729158100ed0d470811050d224e8f"
    "/root/.cache/huggingface/hub/models--zai-org--GLM-5.1/snapshots/26e1bd6e011feb778d25ae34b09b07074139d92d"
    "${HOME:-/root}/.cache/huggingface/hub/models--zai-org--GLM-5.1/snapshots/26e1bd6e011feb778d25ae34b09b07074139d92d"
  )
  for candidate in "${candidates[@]}"; do
    if base_snapshot_ready "$candidate"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  local roots=(
    "/vePFS-Mindverse/share/huggingface/hub/models--zai-org--GLM-5.1/snapshots"
    "/root/.cache/huggingface/hub/models--zai-org--GLM-5.1/snapshots"
    "${HOME:-/root}/.cache/huggingface/hub/models--zai-org--GLM-5.1/snapshots"
  )
  if [ -n "${HF_HOME:-}" ]; then
    roots+=("$HF_HOME/hub/models--zai-org--GLM-5.1/snapshots")
  fi
  for root in "${roots[@]}"; do
    [ -d "$root" ] || continue
    for candidate in "$root"/*; do
      [ -d "$candidate" ] || continue
      if base_snapshot_ready "$candidate"; then
        printf '%s\n' "$candidate"
        return 0
      fi
    done
  done
  return 1
}

sglang_available() {
  python3 - <<'PY' >/dev/null 2>&1
import importlib.util
for name in ("sglang", "sglang.launch_server"):
    if importlib.util.find_spec(name) is None:
        raise SystemExit(1)
PY
}

vllm_available() {
  [ -x /usr/local/bin/vllm ] || command -v vllm >/dev/null 2>&1
}

select_backend() {
  case "$BACKEND" in
    auto)
      if sglang_available; then
        printf '%s\n' sglang
      elif vllm_available; then
        printf '%s\n' vllm
      else
        fail "missing serve backend: neither python3 -m sglang.launch_server nor vllm is available"
      fi
      ;;
    sglang)
      sglang_available || fail "missing Python module: sglang.launch_server"
      printf '%s\n' sglang
      ;;
    vllm)
      vllm_available || fail "missing vLLM entrypoint: /usr/local/bin/vllm or command vllm"
      printf '%s\n' vllm
      ;;
    *)
      echo "bad --backend: $BACKEND" >&2
      exit 2
      ;;
  esac
}

echo "== Host =="
hostname || true
pwd || true

echo "== GPU =="
command -v nvidia-smi >/dev/null || fail "nvidia-smi not found"
nvidia-smi
GPU_COUNT="$(/usr/bin/python3.12 - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
[ "$GPU_COUNT" -ge 8 ] || fail "expected at least 8 visible GPUs, got ${GPU_COUNT}"
echo "OK: visible GPU count ${GPU_COUNT}"

echo "== Disk =="
if [ ! -d /root/tmp ]; then
  mkdir -p /root/tmp || fail "failed to create /root/tmp"
  echo "OK: created /root/tmp"
fi
DF_PATHS=(/root/tmp /tmp)
if [ -d /vePFS-Mindverse/share ]; then
  DF_PATHS+=(/vePFS-Mindverse/share)
fi
if [ -d "$(dirname "$ADAPTER_ROOT")" ]; then
  DF_PATHS+=("$(dirname "$ADAPTER_ROOT")")
fi
df -h "${DF_PATHS[@]}"
ROOT_TMP_FREE_GIB="$(df -BG --output=avail /root/tmp | tail -1 | tr -dc '0-9')"
[ "${ROOT_TMP_FREE_GIB:-0}" -ge "$MIN_ROOT_TMP_GIB" ] || fail "/root/tmp free ${ROOT_TMP_FREE_GIB}GiB < required ${MIN_ROOT_TMP_GIB}GiB"
echo "OK: /root/tmp free ${ROOT_TMP_FREE_GIB}GiB"

echo "== Runtime =="
check_path /usr/bin/python3.12 "Python 3.12 executable"
command -v python3 >/dev/null || fail "python3 not found for serve backend or tinker URL conversion"
SELECTED_BACKEND="$(select_backend)"
echo "OK: selected backend: $SELECTED_BACKEND"
if [ "$SELECTED_BACKEND" = "sglang" ]; then
  echo "OK: sglang launch_server import"
else
  if [ -x /usr/local/bin/vllm ]; then
    echo "OK: vLLM entrypoint: /usr/local/bin/vllm"
  else
    echo "OK: vLLM entrypoint: $(command -v vllm)"
  fi
fi
/usr/bin/python3.12 - <<'PY'
import torch, transformers, safetensors
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.device_count())
print("transformers", transformers.__version__)
print("safetensors", safetensors.__version__)
PY

echo "== Model And Adapter =="
if [ -z "$BASE" ]; then
  BASE="$(detect_base_snapshot || true)"
fi
[ -n "$BASE" ] || fail "could not auto-detect GLM-5.1 BF16 base snapshot; pass --base"
check_path "$BASE" "base snapshot"
check_path "$BASE/config.json" "base config"
check_path "$BASE/model.safetensors.index.json" "base safetensors index"

if [ -f "$PFS_LORA/adapter_config.json" ]; then
  LORA="$PFS_LORA"
  echo "OK: PFS adapter available: $LORA"
elif [ -f "$FALLBACK_LORA/adapter_config.json" ]; then
  LORA="$FALLBACK_LORA"
  echo "OK: downloaded adapter available: $LORA"
else
  LORA="$FALLBACK_LORA"
  echo "PFS adapter unavailable; deploy will download the tinker archive to: $LORA"
  command -v curl >/dev/null || fail "curl not found for HTTP archive download"
  command -v tar >/dev/null || fail "tar not found for HTTP archive extraction"
  command -v python3 >/dev/null || fail "python3 not found for tinker archive URL conversion"
  export GPU_LEASE_API_KEY="${GPU_LEASE_API_KEY:-}"
  [ -n "$GPU_LEASE_API_KEY" ] || fail "GPU_LEASE_API_KEY is required when the adapter must be downloaded"
  mkdir -p "$ADAPTER_ROOT" || fail "failed to create adapter root: $ADAPTER_ROOT"
  echo "OK: HTTP archive fallback prerequisites present"
fi

if [ -f "$LORA/adapter_config.json" ]; then
  if [ -f "$LORA/adapter_model.safetensors" ]; then
    echo "OK: PEFT adapter: $LORA/adapter_model.safetensors"
  elif ls "$LORA"/mp_rank_*_adapter.pt >/dev/null 2>&1; then
    SHARD_COUNT="$(find "$LORA" -maxdepth 1 -type f -name '*_adapter.pt' | wc -l | tr -d ' ')"
    echo "OK: Megatron/MBridge adapter shards present: $SHARD_COUNT"
  else
    fail "missing adapter_model.safetensors or mp_rank_*_adapter.pt under $LORA"
  fi
fi

echo "== Derived Paths =="
echo "tinker_url=$TINKER_URL"
echo "run_id=$TINKER_RUN_ID"
echo "name=$TINKER_WEIGHT_NAME"
echo "base=$BASE"
echo "lora=$LORA"
echo "adapter_root=$ADAPTER_ROOT"
echo "backend=$SELECTED_BACKEND"
echo "merged=/root/tmp/glm5_local_merge/${TINKER_WEIGHT_NAME}_merged_bf16"
echo "fp8_local=/root/tmp/glm5_local_quant/${TINKER_WEIGHT_NAME}_fp8"
if [ "$SELECTED_BACKEND" = "sglang" ]; then
  echo "serve_url=http://127.0.0.1:30000/v1"
else
  echo "serve_url=http://127.0.0.1:8000/v1"
fi
echo "served_model_name=$TINKER_WEIGHT_NAME"
echo "OK: preflight passed"
