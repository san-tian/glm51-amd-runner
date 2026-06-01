#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  serve_glm51_fp8.sh --model-path PATH --served-model-name NAME [options]

Options:
  --backend auto|sglang|vllm  serve backend. Default: auto
  --port PORT                 backend port. Default: 30000 for SGLang, 8000 for vLLM
  --host HOST                 bind host. Default: 0.0.0.0
  --tp N                      tensor parallel size. Default: 8
  --log PATH                  log path. Default: /tmp/<served-model-name>-<backend>-serve.log
  --no-kill-existing          do not stop an existing server on this port

SGLang options:
  --mem-fraction-static F     SGLang static memory fraction. Default: 0.85
  --disable-custom-all-reduce pass through when SGLang custom all-reduce fails during CUDA graph capture
  Prometheus metrics and MFU metrics are enabled by default.

vLLM options:
  --max-model-len N           context length. Default: 131072
  --pythonpath PATH           prepend to PYTHONPATH. Default: /mnt/workspace/user/intern/chengy/mint
  --cache-root PATH           set VLLM_CACHE_ROOT. Default: unset, use vLLM default cache
  --clear-aot-cache           remove VLLM_CACHE_ROOT before launch. Requires --cache-root
  --enable-mtp                enable in-target MTP speculative decoding. Default.
  --disable-mtp               serve without speculative decoding
  --num-speculative-tokens N  MTP speculative token count. Default: 3

Stable SGLang command:
  python3 -m sglang.launch_server --model-path <model_path> --served-model-name <model_name> --host 0.0.0.0 --port 30000 --tp 8 --reasoning-parser glm45 --tool-call-parser glm47 --mem-fraction-static 0.85 --enable-metrics --enable-mfu-metrics

Stable vLLM command:
  /usr/local/bin/vllm serve <model_path> --host 0.0.0.0 --port 8000 --tensor-parallel-size 8 --trust-remote-code --served-model-name <model_name> --max-model-len 131072 --tool-call-parser glm47 --reasoning-parser glm45 --enable-auto-tool-choice --chat-template-content-format string --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
EOF
}

MODEL_PATH=""
SERVED_MODEL_NAME=""
BACKEND="auto"
HOST="0.0.0.0"
PORT=""
TP="8"
MEM_FRACTION_STATIC="0.85"
MAX_MODEL_LEN="131072"
LOG=""
PYTHONPATH_PREPEND="/mnt/workspace/user/intern/chengy/mint"
CACHE_ROOT=""
CLEAR_AOT_CACHE="0"
KILL_EXISTING="1"
ENABLE_MTP="1"
NUM_SPECULATIVE_TOKENS="3"
DISABLE_CUSTOM_ALL_REDUCE="${GLM5_SGLANG_DISABLE_CUSTOM_ALL_REDUCE:-0}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --model-path) MODEL_PATH="${2:?}"; shift 2 ;;
    --served-model-name) SERVED_MODEL_NAME="${2:?}"; shift 2 ;;
    --backend) BACKEND="${2:?}"; shift 2 ;;
    --host) HOST="${2:?}"; shift 2 ;;
    --port) PORT="${2:?}"; shift 2 ;;
    --tp) TP="${2:?}"; shift 2 ;;
    --mem-fraction-static) MEM_FRACTION_STATIC="${2:?}"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="${2:?}"; shift 2 ;;
    --log) LOG="${2:?}"; shift 2 ;;
    --pythonpath) PYTHONPATH_PREPEND="${2:?}"; shift 2 ;;
    --cache-root) CACHE_ROOT="${2:?}"; shift 2 ;;
    --clear-aot-cache) CLEAR_AOT_CACHE="1"; shift ;;
    --enable-mtp) ENABLE_MTP="1"; shift ;;
    --disable-mtp) ENABLE_MTP="0"; shift ;;
    --num-speculative-tokens) NUM_SPECULATIVE_TOKENS="${2:?}"; shift 2 ;;
    --no-kill-existing) KILL_EXISTING="0"; shift ;;
    --disable-custom-all-reduce) DISABLE_CUSTOM_ALL_REDUCE="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [ -z "$MODEL_PATH" ] || [ -z "$SERVED_MODEL_NAME" ]; then
  usage
  exit 2
fi

test -d "$MODEL_PATH"
test -f "$MODEL_PATH/model.safetensors.index.json"
test -f "$MODEL_PATH/config.json"

sglang_available() {
  python3 - <<'PY' >/dev/null 2>&1
import importlib.util
for name in ("sglang", "sglang.launch_server"):
    if importlib.util.find_spec(name) is None:
        raise SystemExit(1)
PY
}

detect_vllm_bin() {
  if [ -x /usr/local/bin/vllm ]; then
    printf '%s\n' /usr/local/bin/vllm
    return 0
  fi
  command -v vllm 2>/dev/null || true
}

select_backend() {
  case "$BACKEND" in
    auto)
      if sglang_available; then
        printf '%s\n' sglang
      elif [ -n "$(detect_vllm_bin)" ]; then
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
      [ -n "$(detect_vllm_bin)" ] || { echo "missing vLLM entrypoint: /usr/local/bin/vllm or command vllm" >&2; exit 1; }
      printf '%s\n' vllm
      ;;
    *)
      echo "bad --backend: $BACKEND" >&2
      exit 2
      ;;
  esac
}

BACKEND="$(select_backend)"
VLLM_BIN="$(detect_vllm_bin)"
if [ -z "$PORT" ]; then
  if [ "$BACKEND" = "sglang" ]; then
    PORT="30000"
  else
    PORT="8000"
  fi
fi

if [ -z "$LOG" ]; then
  LOG="/tmp/${SERVED_MODEL_NAME}-${BACKEND}-serve.log"
fi
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

date -Is
echo "MODEL_PATH=$MODEL_PATH"
echo "SERVED_MODEL_NAME=$SERVED_MODEL_NAME"
echo "BACKEND=$BACKEND"
echo "HOST=$HOST"
echo "PORT=$PORT"
echo "TP=$TP"

export HF_HOME="${HF_HOME:-/vePFS-Mindverse/share/huggingface}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

if [ "$BACKEND" = "sglang" ]; then
  echo "MEM_FRACTION_STATIC=$MEM_FRACTION_STATIC"
  echo "SGLANG_METRICS=enabled"
  echo "SGLANG_MFU_METRICS=enabled"
  echo "DISABLE_CUSTOM_ALL_REDUCE=$DISABLE_CUSTOM_ALL_REDUCE"
  echo "server_entrypoint=python3 -m sglang.launch_server"

  if [ "$KILL_EXISTING" = "1" ]; then
    OLD_PIDS="$(pgrep -f "(^| )python3 -m sglang.launch_server .* --port ${PORT}( |$)" || true)"
    if [ -n "$OLD_PIDS" ]; then
      echo "Stopping existing SGLang server on port $PORT: $OLD_PIDS"
      kill $OLD_PIDS
      for _ in $(seq 1 90); do
        if ! pgrep -f "(^| )python3 -m sglang.launch_server .* --port ${PORT}( |$)" >/dev/null; then
          break
        fi
        sleep 2
      done
    fi
  fi

  SGLANG_ARGS=(
    -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --host "$HOST"
    --port "$PORT"
    --tp "$TP"
    --reasoning-parser glm45
    --tool-call-parser glm47
    --mem-fraction-static "$MEM_FRACTION_STATIC"
    --enable-metrics
    --enable-mfu-metrics
  )
  if [ "$DISABLE_CUSTOM_ALL_REDUCE" = "1" ]; then
    SGLANG_ARGS+=(--disable-custom-all-reduce)
  fi

  exec python3 "${SGLANG_ARGS[@]}"
fi

echo "MAX_MODEL_LEN=$MAX_MODEL_LEN"
echo "compile_mode=default_vllm_compile_no_enforce_eager"
if [ "$ENABLE_MTP" = "1" ]; then
  echo "mtp=enabled"
  echo "num_speculative_tokens=$NUM_SPECULATIVE_TOKENS"
else
  echo "mtp=disabled"
fi
echo "vllm_entrypoint=$VLLM_BIN"
echo "mint_child_real_python=/usr/bin/python3.12"
echo "mint_import_patches=disabled"

export PYTHONPATH="${PYTHONPATH_PREPEND}:${PYTHONPATH:-}"
export VLLM_USE_FLASHINFER_MOE_FP8=0
export MINT_VLLM_REAL_PYTHON_EXECUTABLE="/usr/bin/python3.12"
unset MINT_ENABLE_VLLM_IMPORT_PATCHES

if [ -n "$CACHE_ROOT" ]; then
  export VLLM_CACHE_ROOT="$CACHE_ROOT"
  if [ "$CLEAR_AOT_CACHE" = "1" ]; then
    rm -rf "$CACHE_ROOT"
  fi
else
  unset VLLM_CACHE_ROOT
  if [ "$CLEAR_AOT_CACHE" = "1" ]; then
    echo "--clear-aot-cache requires --cache-root" >&2
    exit 2
  fi
fi

if [ "$KILL_EXISTING" = "1" ]; then
  OLD_PIDS="$(pgrep -f "(^|[ /])vllm serve .* --port ${PORT}( |$)" || true)"
  if [ -n "$OLD_PIDS" ]; then
    echo "Stopping existing vLLM serve on port $PORT: $OLD_PIDS"
    kill $OLD_PIDS
    for _ in $(seq 1 90); do
      if ! pgrep -f "(^|[ /])vllm serve .* --port ${PORT}( |$)" >/dev/null; then
        break
      fi
      sleep 2
    done
  fi
fi

VLLM_ARGS=(
  serve "$MODEL_PATH"
  --host "$HOST"
  --port "$PORT"
  --tensor-parallel-size "$TP"
  --trust-remote-code
  --served-model-name "$SERVED_MODEL_NAME"
  --max-model-len "$MAX_MODEL_LEN"
  --tool-call-parser glm47
  --reasoning-parser glm45
  --enable-auto-tool-choice
  --chat-template-content-format string
)

if [ "$ENABLE_MTP" = "1" ]; then
  VLLM_ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${NUM_SPECULATIVE_TOKENS}}")
fi

exec "$VLLM_BIN" "${VLLM_ARGS[@]}"
