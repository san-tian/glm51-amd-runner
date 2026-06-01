#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  merge_quant_serve_glm51_fp8.sh --tinker-url URL [options]
  merge_quant_serve_glm51_fp8.sh --ssh "ssh ..." --tinker-url URL [options]
  merge_quant_serve_glm51_fp8.sh --base PATH --lora PATH --merged PATH --fp8 PATH --served-model-name NAME [options]

Options:
  --ssh CMD                   run this same deploy flow on a remote server over ssh
  --remote-root PATH          temporary remote skill directory. Default: /tmp/glm5-fp8-deploy-skill-...
  --backend auto|sglang|vllm  serve backend. Default: auto
  --tinker-url URL            resolve from PFS adapter cache or HTTP archive fallback
  --gpus CSV                  GPU ids for local merge workers. Default: 0,1,2,3,4,5,6,7
  --workers-per-gpu N         local merge/quant workers per GPU. Default: 8
  --adapter-root PATH         HTTP archive adapter root when PFS adapter is unavailable. Default: /data0/glm51_adapters
  --host HOST                 serve host. Default: 0.0.0.0
  --port PORT                 serve port. Default: 30000 for SGLang, 8000 for vLLM
  --tp N                      tensor parallel size for serving. Default: 8
  --mem-fraction-static F     SGLang static memory fraction. Default: 0.85
  --max-model-len N           vLLM context length. Default: 131072
  --pythonpath PATH           vLLM PYTHONPATH prepend. Default: /mnt/workspace/user/intern/chengy/mint
  --cache-root PATH           vLLM VLLM_CACHE_ROOT
  --clear-aot-cache           clear vLLM VLLM_CACHE_ROOT before launch. Requires --cache-root
  --enable-mtp                enable vLLM in-target MTP speculative decoding. Default.
  --disable-mtp               disable vLLM MTP speculative decoding
  --num-speculative-tokens N  vLLM MTP speculative token count. Default: 3
  --skip-serve                stop after merge+quant+validation
  --force-download-adapter    re-download and re-extract the HTTP archive adapter
  --force-merge               recreate merged output
  --force-quant               recreate FP8 output
  --save-to-pfs               after local quant completes, move FP8 output to --fp8
  --disable-custom-all-reduce pass through to SGLang if custom all-reduce fails during CUDA graph capture
  --no-expand-sparse-experts  pass through to merge script for control experiments only
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORIGINAL_ARGS=("$@")
SSH_CMD=""
REMOTE_ROOT=""
BACKEND="auto"
TINKER_URL=""
BASE=""
LORA=""
MERGED=""
FP8=""
SERVED_MODEL_NAME=""
GPUS="0,1,2,3,4,5,6,7"
WORKERS_PER_GPU="8"
ADAPTER_ROOT="${GLM5_ADAPTER_ROOT:-/data0/glm51_adapters}"
HOST="0.0.0.0"
PORT=""
TP="8"
MEM_FRACTION_STATIC="0.85"
MAX_MODEL_LEN="131072"
PYTHONPATH_PREPEND="/mnt/workspace/user/intern/chengy/mint"
CACHE_ROOT=""
CLEAR_AOT_CACHE="0"
ENABLE_MTP="1"
NUM_SPECULATIVE_TOKENS="3"
SKIP_SERVE="0"
DOWNLOAD_ADAPTER="0"
FORCE_DOWNLOAD_ADAPTER="0"
FORCE_MERGE="0"
FORCE_QUANT="0"
SAVE_TO_PFS="0"
NO_EXPAND="0"
DISABLE_CUSTOM_ALL_REDUCE="${GLM5_SGLANG_DISABLE_CUSTOM_ALL_REDUCE:-0}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --ssh) SSH_CMD="${2:?}"; shift 2 ;;
    --remote-root) REMOTE_ROOT="${2:?}"; shift 2 ;;
    --backend) BACKEND="${2:?}"; shift 2 ;;
    --tinker-url) TINKER_URL="${2:?}"; shift 2 ;;
    --base) BASE="${2:?}"; shift 2 ;;
    --lora) LORA="${2:?}"; shift 2 ;;
    --merged) MERGED="${2:?}"; shift 2 ;;
    --fp8) FP8="${2:?}"; shift 2 ;;
    --served-model-name) SERVED_MODEL_NAME="${2:?}"; shift 2 ;;
    --gpus) GPUS="${2:?}"; shift 2 ;;
    --workers-per-gpu) WORKERS_PER_GPU="${2:?}"; shift 2 ;;
    --adapter-root) ADAPTER_ROOT="${2:?}"; shift 2 ;;
    --host) HOST="${2:?}"; shift 2 ;;
    --port) PORT="${2:?}"; shift 2 ;;
    --tp) TP="${2:?}"; shift 2 ;;
    --mem-fraction-static) MEM_FRACTION_STATIC="${2:?}"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="${2:?}"; shift 2 ;;
    --pythonpath) PYTHONPATH_PREPEND="${2:?}"; shift 2 ;;
    --cache-root) CACHE_ROOT="${2:?}"; shift 2 ;;
    --clear-aot-cache) CLEAR_AOT_CACHE="1"; shift ;;
    --enable-mtp) ENABLE_MTP="1"; shift ;;
    --disable-mtp) ENABLE_MTP="0"; shift ;;
    --num-speculative-tokens) NUM_SPECULATIVE_TOKENS="${2:?}"; shift 2 ;;
    --skip-serve) SKIP_SERVE="1"; shift ;;
    --force-download-adapter) FORCE_DOWNLOAD_ADAPTER="1"; shift ;;
    --force-merge) FORCE_MERGE="1"; shift ;;
    --force-quant) FORCE_QUANT="1"; shift ;;
    --save-to-pfs) SAVE_TO_PFS="1"; shift ;;
    --disable-custom-all-reduce) DISABLE_CUSTOM_ALL_REDUCE="1"; shift ;;
    --no-expand-sparse-experts) NO_EXPAND="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

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

if [ -n "$SSH_CMD" ]; then
  if [ -z "$REMOTE_ROOT" ]; then
    REMOTE_ROOT="/tmp/glm5-fp8-deploy-skill-${USER:-user}-deploy-$$"
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
  exec $SSH_CMD "bash '$REMOTE_SCRIPT_DIR/merge_quant_serve_glm51_fp8.sh' $REMOTE_ARG_STRING"
fi

if [ -n "$TINKER_URL" ]; then
  if [[ "$TINKER_URL" != tinker://*/weights/* ]]; then
    echo "bad --tinker-url, expected tinker://<run-id>/weights/<name>: $TINKER_URL" >&2
    exit 2
  fi
  TINKER_BODY="${TINKER_URL#tinker://}"
  TINKER_RUN_ID="${TINKER_BODY%%/weights/*}"
  TINKER_WEIGHT_PATH="${TINKER_BODY#*/weights/}"
  TINKER_WEIGHT_NAME="${TINKER_WEIGHT_PATH##*/}"
  if [ -z "$TINKER_RUN_ID" ] || [ -z "$TINKER_WEIGHT_NAME" ]; then
    echo "bad --tinker-url, empty run id or weight name: $TINKER_URL" >&2
    exit 2
  fi
  if [ -z "$BASE" ]; then
    BASE="$(detect_base_snapshot || true)"
  fi
  BASE="${BASE:-/vePFS-Mindverse/share/huggingface/hub/models--zai-org--GLM-5.1/snapshots/5c155b85c12729158100ed0d470811050d224e8f}"
  PFS_LORA="/vePFS-Mindverse/share/tinker_runtime_checkpoints/persistent_cache/admin/${TINKER_RUN_ID}/${TINKER_WEIGHT_NAME}"
  if [ -z "$LORA" ]; then
    if [ "$FORCE_DOWNLOAD_ADAPTER" = "1" ]; then
      LORA="$ADAPTER_ROOT/$TINKER_WEIGHT_NAME"
      DOWNLOAD_ADAPTER="1"
    elif [ -f "$PFS_LORA/adapter_config.json" ]; then
      LORA="$PFS_LORA"
    else
      LORA="$ADAPTER_ROOT/$TINKER_WEIGHT_NAME"
      DOWNLOAD_ADAPTER="1"
    fi
  fi
  MERGED="${MERGED:-/root/tmp/glm5_local_merge/${TINKER_WEIGHT_NAME}_merged_bf16}"
  FP8="${FP8:-/vePFS-Mindverse/share/tmp/glm5_local_fp8/${TINKER_WEIGHT_NAME}_fp8}"
  SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$TINKER_WEIGHT_NAME}"
fi

if [ -z "$BASE" ] || [ -z "$LORA" ] || [ -z "$MERGED" ] || [ -z "$FP8" ] || [ -z "$SERVED_MODEL_NAME" ]; then
  usage
  exit 2
fi

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
        echo "missing serve backend: neither python3 -m sglang.launch_server nor vllm is available" >&2
        exit 1
      fi
      ;;
    sglang)
      sglang_available || { echo "missing Python module: sglang.launch_server" >&2; exit 1; }
      printf '%s\n' sglang
      ;;
    vllm)
      vllm_available || { echo "missing vLLM entrypoint: /usr/local/bin/vllm or command vllm" >&2; exit 1; }
      printf '%s\n' vllm
      ;;
    *)
      echo "bad --backend: $BACKEND" >&2
      exit 2
      ;;
  esac
}

if [ "$SKIP_SERVE" = "1" ]; then
  case "$BACKEND" in
    auto)
      if sglang_available; then
        BACKEND="sglang"
      elif vllm_available; then
        BACKEND="vllm"
      else
        BACKEND="none"
      fi
      ;;
    sglang|vllm) ;;
    *)
      echo "bad --backend: $BACKEND" >&2
      exit 2
      ;;
  esac
else
  BACKEND="$(select_backend)"
fi
if [ -z "$PORT" ] && [ "$BACKEND" != "none" ]; then
  if [ "$BACKEND" = "sglang" ]; then
    PORT="30000"
  else
    PORT="8000"
  fi
fi
echo "selected_backend=$BACKEND"
if [ "$BACKEND" != "none" ]; then
  echo "serve_port=$PORT"
fi

export HF_HOME="${HF_HOME:-/vePFS-Mindverse/share/huggingface}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export GLM5_FP8_SHARDWISE="${GLM5_FP8_SHARDWISE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ "$DOWNLOAD_ADAPTER" = "1" ] || [ "$FORCE_DOWNLOAD_ADAPTER" = "1" ]; then
  if [ -z "$TINKER_URL" ]; then
    echo "--force-download-adapter requires --tinker-url" >&2
    exit 2
  fi
  ADAPTER_NAME="${TINKER_WEIGHT_NAME:-$(basename "$LORA")}"
  DOWNLOAD_ARGS=(
    --tinker-url "$TINKER_URL"
    --name "$ADAPTER_NAME"
    --adapter-root "$ADAPTER_ROOT"
    --adapter-dir "$LORA"
  )
  if [ "$FORCE_DOWNLOAD_ADAPTER" = "1" ]; then
    DOWNLOAD_ARGS+=(--force)
  fi
  "$SCRIPT_DIR/download_tinker_adapter_archive.sh" "${DOWNLOAD_ARGS[@]}"
fi

if [ ! -d "$BASE" ]; then
  echo "missing base snapshot: $BASE" >&2
  echo "If this host has no /vePFS-Mindverse/share, pass --base to a local GLM-5.1 BF16 snapshot." >&2
  exit 1
fi
if [ ! -f "$LORA/adapter_config.json" ]; then
  echo "missing adapter_config.json under LoRA adapter dir: $LORA" >&2
  exit 1
fi
mkdir -p "$(dirname "$MERGED")"
if [ "$SAVE_TO_PFS" = "1" ]; then
  mkdir -p "$(dirname "$FP8")"
fi

EFFECTIVE_LORA="$LORA"
if [ -f "$LORA/adapter_model.safetensors" ]; then
  echo "Using PEFT adapter: $LORA"
elif ls "$LORA"/mp_rank_*_adapter.pt >/dev/null 2>&1; then
  if [ -d /vePFS-Mindverse/share/tmp ]; then
    DEFAULT_PEFT_ROOT="/vePFS-Mindverse/share/tmp/glm5_lora_peft"
  else
    DEFAULT_PEFT_ROOT="$ADAPTER_ROOT/peft"
  fi
  PEFT_ROOT="${GLM5_LORA_PEFT_ROOT:-$DEFAULT_PEFT_ROOT}"
  PEFT_NAME="${TINKER_WEIGHT_NAME:-$(basename "$LORA")}"
  PEFT_LORA="$PEFT_ROOT/${PEFT_NAME}_peft"
  if [ ! -f "$PEFT_LORA/adapter_model.safetensors" ]; then
    mkdir -p "$PEFT_ROOT"
    /usr/bin/python3.12 "$SCRIPT_DIR/convert_megatron_lora_to_peft.py" \
      --input-path "$LORA" \
      --output-path "$PEFT_LORA" \
      --base-model-path "$BASE"
  else
    echo "Converted PEFT adapter exists: $PEFT_LORA"
  fi
  EFFECTIVE_LORA="$PEFT_LORA"
else
  echo "missing adapter_model.safetensors or mp_rank_*_adapter.pt under $LORA" >&2
  exit 1
fi

MERGE_ARGS=(
  --base-model-path "$BASE"
  --lora-path "$EFFECTIVE_LORA"
  --output-path "$MERGED"
  --gpus "$GPUS"
  --workers-per-gpu "$WORKERS_PER_GPU"
)
if [ "$FORCE_MERGE" = "1" ]; then
  MERGE_ARGS+=(--force)
fi
if [ "$NO_EXPAND" = "1" ]; then
  MERGE_ARGS+=(--no-expand-sparse-experts)
fi

if [ ! -f "$MERGED/merge_summary.json" ] || [ "$FORCE_MERGE" = "1" ]; then
  /usr/bin/python3.12 "$SCRIPT_DIR/merge_glm5_lora_into_base_local.py" "${MERGE_ARGS[@]}"
else
  echo "Merged checkpoint exists: $MERGED"
fi

LOCAL_FP8="${GLM5_LOCAL_QUANT_ROOT:-/root/tmp/glm5_local_quant}/$(basename "$FP8")"
EFFECTIVE_FP8="$LOCAL_FP8"
if [ "$SAVE_TO_PFS" = "1" ]; then
  EFFECTIVE_FP8="$FP8"
fi

if [ "$FORCE_QUANT" = "1" ]; then
  rm -rf "$LOCAL_FP8"
  if [ "$SAVE_TO_PFS" = "1" ]; then
    rm -rf "$FP8"
  fi
fi
if [ ! -f "$EFFECTIVE_FP8/fp8_quant_meta.json" ]; then
  QUANT_ARGS=(
    --base-model-path "$MERGED"
    --export-dir "$FP8"
    --trust-remote-code
    --gpus "$GPUS"
    --workers-per-gpu "$WORKERS_PER_GPU"
    --local-quant-root "${GLM5_LOCAL_QUANT_ROOT:-/root/tmp/glm5_local_quant}"
  )
  if [ "$SAVE_TO_PFS" = "1" ]; then
    QUANT_ARGS+=(--save-to-pfs)
  fi
  /usr/bin/python3.12 "$SCRIPT_DIR/quantize_glm5_finegrained_fp8_parallel.py" "${QUANT_ARGS[@]}"
else
  echo "FP8 checkpoint exists: $EFFECTIVE_FP8"
fi

VALIDATE_ARGS=(
  --model-path "$EFFECTIVE_FP8"
)
if [ "$BACKEND" = "vllm" ] && [ "$ENABLE_MTP" = "1" ]; then
  VALIDATE_ARGS+=(--require-mtp)
fi
/usr/bin/python3.12 "$SCRIPT_DIR/validate_glm51_fp8_checkpoint.py" "${VALIDATE_ARGS[@]}"

if [ "$SKIP_SERVE" = "1" ]; then
  exit 0
fi

SERVE_ARGS=(
  --backend "$BACKEND"
  --model-path "$EFFECTIVE_FP8"
  --served-model-name "$SERVED_MODEL_NAME"
  --host "$HOST"
  --port "$PORT"
  --tp "$TP"
)
if [ "$BACKEND" = "sglang" ]; then
  SERVE_ARGS+=(--mem-fraction-static "$MEM_FRACTION_STATIC")
fi
if [ "$BACKEND" = "sglang" ] && [ "$DISABLE_CUSTOM_ALL_REDUCE" = "1" ]; then
  SERVE_ARGS+=(--disable-custom-all-reduce)
fi
if [ "$BACKEND" = "vllm" ]; then
  SERVE_ARGS+=(--max-model-len "$MAX_MODEL_LEN")
  SERVE_ARGS+=(--pythonpath "$PYTHONPATH_PREPEND")
  SERVE_ARGS+=(--num-speculative-tokens "$NUM_SPECULATIVE_TOKENS")
  if [ -n "$CACHE_ROOT" ]; then
    SERVE_ARGS+=(--cache-root "$CACHE_ROOT")
  fi
  if [ "$CLEAR_AOT_CACHE" = "1" ]; then
    SERVE_ARGS+=(--clear-aot-cache)
  fi
  if [ "$ENABLE_MTP" = "1" ]; then
    SERVE_ARGS+=(--enable-mtp)
  else
    SERVE_ARGS+=(--disable-mtp)
  fi
fi

exec "$SCRIPT_DIR/serve_glm51_fp8.sh" \
  "${SERVE_ARGS[@]}"
