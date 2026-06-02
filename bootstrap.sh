# 这个粘贴块会自己准备 /local_nvme，不需要你先开 tmux。
# 如果 /local_nvme 不可写，会自动选择两块“空白、未挂载、无分区”的 NVMe 做 RAID0 + XFS 并挂载到 /local_nvme。
# 破坏性操作：被选中的两块 NVMe 会被清空。
# 注意：严格模式放在子 bash 里，失败只退出这个粘贴任务，不会退出当前 SSH 登录 shell。

########################################
# 0. 先改这里：所有常用环境变量集中在最前面
########################################
 
# 是否在 serve ready 后自动新开 tmux window 跑 GSM8K lm_eval：0=不跑，1=跑。
export RUN_LMEVAL="1"
export MODEL_ID="zai-org/GLM-5.1-FP8"
export SGLANG_PORT="7777"
# SGLang serve 参数。调参优先改这一整块，保持每行的反斜杠续行。
# 官方 AMD GPU recipe 的入口是 python3 -m sglang.launch_server。
# PR355 示例使用 --tensor-parallel-size 8、fp8_e4m3 KV cache、mem fraction 0.8、page-size 1。
# GLM parser 参数保留，但 PR 示例未覆盖 GLM-5.1-FP8，需实测。
read -r -d '' SGLANG_SERVE_ARGS <<'SGLANG_ARGS' || true
python3 -m sglang.launch_server \
  --model-path "$MODEL_DIR" \
  --host 0.0.0.0 \
  --port "$SGLANG_PORT" \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.8 \
  --page-size 1 \
  --reasoning-parser glm45 --tool-call-parser glm47 \
  --enable-metrics --enable-mfu-metrics --disable-custom-all-reduce
SGLANG_ARGS
export SGLANG_SERVE_ARGS

# 私有 / gated 模型需要 HF token 时，在远端手动注入 secret 文件，不要把 token 明文写进 Obsidian。
# 默认位置会在控制面确定后变成：${CONTROL_PLANE_DIR}/secrets/hf_token.env
# 文件内容格式：HF_TOKEN=hf_...
export HF_TOKEN="${HF_TOKEN:-}"
export GLM51_SECRETS_FILE="${GLM51_SECRETS_FILE:-}"
export HF_TOKEN_FILE="${HF_TOKEN_FILE:-}"

# 设置systemd开机启动
export INSTALL_SYSTEMD_AUTOSTART="1"
export AUTOSTART_SERVICE_NAME="glm51-autostart"
export AUTOSTART_CHECK_INTERVAL_SECONDS="60"
# 本地 NVMe 丢失且 resume 不存在时，是否允许 autostart 调控制面 bootstrap.sh 自动重建/重下载。
# 接受重下载时保持 1；改成 0 则只报错，不自动重建。
export ALLOW_REDOWNLOAD_ON_LOCAL_NVME_LOSS="1"

# 持久日志根目录；留空自动选择第一个存在且可写的 /data、/data2、/mnt、/opt/glm51。
# 例如强制用 /data2：export CONTROL_PLANE_ROOT="/data2"
export PERSIST_LOG_ROOT="${PERSIST_LOG_ROOT:-}"

# 实验日志分组；RUN_ID 会按北京时间/LOG_TIMEZONE + 安全化实验名生成。
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-glm51-fp8-atom-pr355-oot}"
export LOG_TIMEZONE="${LOG_TIMEZONE:-Asia/Shanghai}"

# 首次诊断日志复制到持久日志目录的最大等待秒数；复制失败只告警，不中断主流程。
export FIRST_RUN_LOG_COPY_TIMEOUT_SECONDS="${FIRST_RUN_LOG_COPY_TIMEOUT_SECONDS:-10}"
# 默认不把 first-run log 复制到 /local_nvme；持久盘日志才是权威。确实需要本地方便副本时改成 1。
export COPY_FIRST_RUN_LOG_TO_LOCAL_NVME="${COPY_FIRST_RUN_LOG_TO_LOCAL_NVME:-0}"

# generated 控制脚本放控制面 generated 目录；不要写入 HOST_WORKDIR(/local_nvme/...)。
export CONTROL_DIR="${CONTROL_DIR:-}"
export GENERATED_SCRIPT_DIR="${GENERATED_SCRIPT_DIR:-}"
export AUTO_ATTACH_TMUX="${AUTO_ATTACH_TMUX:-0}"
# autostart-observe 是循环刷新状态的 tmux 窗口。默认关闭，避免抢占 SSH/tmux 焦点。
export AUTOSTART_CREATE_OBSERVE_WINDOW="${AUTOSTART_CREATE_OBSERVE_WINDOW:-0}"

# BASE_IMAGE 用于下载/校验 Hugging Face 模型，并作为 Tinker merge/quant 的 ROCm 运行时；serve 使用 SGLANG_IMAGE。
export BASE_IMAGE="rocm/atom-dev@sha256:9be7af4ec2b5eed8826521db5719e9610ce03f784fb49cc15effb1f2584192eb"
export GLM51_SCRIPT_VERSION="markdown-20260602-sglang-quark-loader-patch-v3.13"
# 若已有包含 ATOM PR355 代码的自定义镜像，在这里覆盖 SGLANG_IMAGE；官方镜像未必包含 atom 包。
export SGLANG_IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang-rocm:v0.5.12.post1-rocm720-mi30x-20260529}"
export SGLANG_CONTAINER="sglang-glm51-fp8-atom-pr355"
export LMEVAL_IMAGE="lm-eval-harness:latest"
export LMEVAL_CONTAINER="lm-eval-glm51-atom-pr355"

export MOUNT_POINT="/local_nvme"
export MD_DEV="/dev/md0"
export RAID_DEVICES="2"
# 如果要强制指定两块盘，取消下一行注释并改盘名：
# export NVME_DEVS="/dev/nvme1n1 /dev/nvme2n1"

export HOST_WORKDIR="${MOUNT_POINT}/amd_profiling"
export CONTAINER_WORKDIR="${MOUNT_POINT}/amd_profiling"
export HOST_TMPDIR="${HOST_WORKDIR}/host-tmp"

export MODEL_REVISION="f396cf805182f4ca10fa675e1a99815b3ca384db"
export MODEL_DIR="${CONTAINER_WORKDIR}/models/GLM-5.1-FP8"
export SERVED_MODEL_NAME="${MODEL_ID}"

# Tinker LoRA merge + quant。
# auto：TINKER_URL 是真实 tinker://.../weights/... 时走 LoRA merge/quant；为空或占位符时部署 base 模型。
# 1：强制走 LoRA merge/quant，TINKER_URL 不是真实 tinker:// 会直接失败。
# 0：强制跳过 LoRA，部署 base 模型。
# merge/quant 严格调用 `Infra实验/reference/glm5-fp8-deploy/scripts` 中的原始脚本；这里只配置路径和参数。
export RUN_TINKER_MERGE_QUANT="${RUN_TINKER_MERGE_QUANT:-auto}"
export GPU_LEASE_BASE_URL="https://eval-service.macaron.im"
export GPU_LEASE_API_KEY="${GPU_LEASE_API_KEY:-}"
export TINKER_URL="${TINKER_URL:-<这里填 tinker://.../weights/...>}"
export LORA_DIR="${CONTAINER_WORKDIR}/loras/tinker-lora"
export MERGED_MODEL_DIR="${CONTAINER_WORKDIR}/models/GLM-5.1-FP8-tinker-merged"
export QUANT_MODEL_DIR="${CONTAINER_WORKDIR}/models/GLM-5.1-FP8-tinker-merged-fp8-dynamic"
export QUANT_SCHEME="${QUANT_SCHEME:-FP8_DYNAMIC}"
export GLM5_REFERENCE_SCRIPTS_DIR="${GLM5_REFERENCE_SCRIPTS_DIR:-}"
export GLM5_MERGE_QUANT_GPUS="${GLM5_MERGE_QUANT_GPUS:-0,1,2,3,4,5,6,7}"
export GLM5_MERGE_QUANT_WORKERS_PER_GPU="${GLM5_MERGE_QUANT_WORKERS_PER_GPU:-8}"
export GLM5_LOCAL_QUANT_ROOT="${GLM5_LOCAL_QUANT_ROOT:-${CONTAINER_WORKDIR}/glm5_local_quant}"


# ATOM PR355 OOT plugin 设置。
# 默认采用方案 B：脚本在宿主持久控制盘自动同步 ROCm/ATOM 到固定 commit，然后只读 bind mount 到容器。
export ATOM_REPO_URL="${ATOM_REPO_URL:-https://github.com/ROCm/ATOM.git}"
export ATOM_REF="${ATOM_REF:-9427621d3dfdfac1c7820bb435b6d034083686ee}"
# ATOM_REPO_ROOT 留空时稍后自动选择：/data/glm51-control/ATOM、/data2/glm51-control/ATOM、/opt/glm51/ATOM。
export ATOM_REPO_ROOT="${ATOM_REPO_ROOT:-}"
# 留空表示使用 ATOM_REPO_ROOT；稍后会归一化为 ATOM_REPO_HOST="${ATOM_REPO_HOST:-$ATOM_REPO_ROOT}"。
export ATOM_REPO_HOST="${ATOM_REPO_HOST:-}"
export ATOM_REPO_CONTAINER="${ATOM_REPO_CONTAINER:-/opt/ATOM}"
# 容器内 PYTHONPATH。默认包含常见 SGLang 源码路径和 ATOM 容器路径；不同镜像可在粘贴前覆盖。
export ATOM_PLUGIN_PYTHONPATH="${ATOM_PLUGIN_PYTHONPATH:-/sgl-workspace/sglang/python:/opt/ATOM}"
export SGLANG_EXTERNAL_MODEL_PACKAGE="${SGLANG_EXTERNAL_MODEL_PACKAGE:-atom.plugin.sglang.models}"
export AITER_QUICK_REDUCE_QUANTIZATION="${AITER_QUICK_REDUCE_QUANTIZATION:-INT4}"
export SGLANG_AITER_FP8_PREFILL_ATTN="${SGLANG_AITER_FP8_PREFILL_ATTN:-0}"
# SGLang 0.5.12 的通用 quark loader 会把 GLM q_a/kv_a A-proj 强行映射到
# fused_qkv_a_proj_with_mqa，连 weight_scale_inv 也一起改名，导致 scale 找不到
# runtime 参数。默认在容器启动时 patch 这条通用映射，让 GLM 模型类自己的
# 成对 fuse 逻辑同时处理 weight 和 scale；不改 checkpoint。
export PATCH_SGLANG_QUARK_FUSED_QKV_A_PROJ="${PATCH_SGLANG_QUARK_FUSED_QKV_A_PROJ:-1}"


export LMEVAL_WORKDIR="${CONTAINER_WORKDIR}/lm_eval"



# Docker 缓存损坏时是否自动做轻量修复：1=是，0=否。
export AUTO_FIX_DOCKER_CACHE="1"

# 空间治理：1=启动前自动清理 Docker/cache/local_nvme 下 eval 输出；0=只检查不清理。
export AUTO_FREE_SPACE="1"
# 如果清完仍然空间不足，是否允许删除已有模型目录重新下载：1=允许，0=保留模型。
export AUTO_DELETE_MODEL_IF_LOW_SPACE="1"
# 空间阈值。HOST 是脚本/日志/状态的最低余量；下载模型和 build 会用更高阈值单独检查。
export MIN_HOST_FREE_GB="5"
export MIN_MODEL_DOWNLOAD_FREE_GB="120"
export MIN_DOCKER_BUILD_FREE_GB="40"

# 下载前是否杀掉旧的 HF/Python 下载进程并清理当前模型目录的残留 lock/tmp。
export AUTO_KILL_STALE_DOWNLOADS="1"
# Hugging Face snapshot_download 的文件下载并发数；下载和 build 仍然是顺序执行，不会同时跑。
export HF_DOWNLOAD_MAX_WORKERS="8"

# 启动前是否自动杀掉并删除已有 Docker containers 来释放 GPU/端口/overlay 空间：1=是，0=否。
export AUTO_KILL_EXISTING_CONTAINERS="1"

# 启动前是否自动杀掉同名 tmux session 和旧 socket：1=是，0=否。
export AUTO_KILL_EXISTING_TMUX_SESSION="1"



export TMUX_TMPDIR="${TMUX_TMPDIR:-}"
export TMUX_SOCKET="${TMUX_SOCKET:-}"
export TMUX_SESSION="glm51"
export PREP_WINDOW="prep-download-build"
export SERVE_WINDOW="sglang-serve"
export LMEVAL_WINDOW="lm-eval"

echo "[markdown] GLM51_SCRIPT_VERSION=${GLM51_SCRIPT_VERSION}"

echo "[markdown] installing bootstrap/control plane under selected control dir and running with sudo"
GLM51_OPT_DIR="${GLM51_OPT_DIR:-}"
control_root_writable_outer() {
  local dir="$1"
  [ -n "$dir" ] || return 1
  sudo install -d -m 0755 "$dir" 2>/dev/null || return 1
  sudo sh -c 'touch "$1/.glm51-control-rw" && rm -f "$1/.glm51-control-rw"' sh "$dir" 2>/dev/null
}
choose_control_dir_outer() {
  local candidate dir
  if [ -n "${CONTROL_PLANE_DIR:-}" ]; then
    if control_root_writable_outer "$CONTROL_PLANE_DIR"; then
      printf '%s
' "$CONTROL_PLANE_DIR"
      return 0
    fi
    echo "[markdown] warning: CONTROL_PLANE_DIR is not writable: $CONTROL_PLANE_DIR" >&2
  fi
  if [ -n "${CONTROL_PLANE_ROOT:-}" ]; then
    dir="${CONTROL_PLANE_ROOT%/}/glm51-control"
    if control_root_writable_outer "$dir"; then
      printf '%s
' "$dir"
      return 0
    fi
    echo "[markdown] warning: CONTROL_PLANE_ROOT is not writable for control plane: $CONTROL_PLANE_ROOT" >&2
  fi
  for candidate in /data /data2 /mnt; do
    [ -d "$candidate" ] || continue
    dir="${candidate}/glm51-control"
    if control_root_writable_outer "$dir"; then
      printf '%s
' "$dir"
      return 0
    fi
  done
  if control_root_writable_outer /opt/glm51; then
    printf '%s
' /opt/glm51
    return 0
  fi
  return 1
}
if ! CONTROL_PLANE_DIR="$(choose_control_dir_outer)"; then
  echo "ERROR: no writable control plane found in /data, /data2, /mnt, or /opt/glm51" >&2
  exit 1
fi
case "$CONTROL_PLANE_DIR" in
  */glm51-control) CONTROL_PLANE_ROOT="${CONTROL_PLANE_DIR%/glm51-control}" ;;
  /opt/glm51) CONTROL_PLANE_ROOT="/opt" ;;
  *) CONTROL_PLANE_ROOT="$(dirname "$CONTROL_PLANE_DIR")" ;;
esac
export CONTROL_PLANE_ROOT CONTROL_PLANE_DIR
CONTROL_DIR="$CONTROL_PLANE_DIR"
export CONTROL_DIR
GENERATED_SCRIPT_DIR="${GENERATED_SCRIPT_DIR:-${CONTROL_PLANE_DIR}/generated}"
if [ -z "${ATOM_REPO_ROOT:-}" ]; then
  ATOM_REPO_ROOT="${CONTROL_PLANE_DIR}/ATOM"
fi
ATOM_REPO_HOST="${ATOM_REPO_HOST:-$ATOM_REPO_ROOT}"
GLM51_OPT_DIR="$CONTROL_PLANE_DIR"
BOOTSTRAP_PATH="${CONTROL_PLANE_DIR}/bootstrap.sh"
GLM51_ENV_FILE="${CONTROL_PLANE_DIR}/glm51.env"
GLM51_SECRETS_FILE="${GLM51_SECRETS_FILE:-${CONTROL_PLANE_DIR}/secrets/glm51-secrets.env}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-${CONTROL_PLANE_DIR}/secrets/hf_token.env}"
TMUX_TMPDIR="${TMUX_TMPDIR:-${CONTROL_PLANE_DIR}/tmux}"
TMUX_SOCKET="${TMUX_SOCKET:-${TMUX_TMPDIR}/glm51.sock}"
export GLM51_OPT_DIR BOOTSTRAP_PATH GLM51_ENV_FILE GLM51_SECRETS_FILE HF_TOKEN_FILE TMUX_TMPDIR TMUX_SOCKET
CONTROL_PLANE_TMPDIR="${CONTROL_PLANE_DIR}/tmp"
if ! sudo install -d -m 0700 "$CONTROL_PLANE_TMPDIR"; then
  echo "ERROR: cannot create control-plane temp dir: $CONTROL_PLANE_TMPDIR" >&2
  exit 1
fi
if ! sudo chown "$(id -u):$(id -g)" "$CONTROL_PLANE_TMPDIR" 2>/dev/null; then
  echo "ERROR: cannot chown control-plane temp dir: $CONTROL_PLANE_TMPDIR" >&2
  exit 1
fi
export CONTROL_PLANE_TMPDIR TMPDIR="$CONTROL_PLANE_TMPDIR"
if ! touch "$TMPDIR/.glm51-tmpdir-rw.$$" 2>/dev/null; then
  echo "ERROR: control-plane temp dir is not writable: $TMPDIR" >&2
  exit 1
fi
rm -f "$TMPDIR/.glm51-tmpdir-rw.$$" 2>/dev/null || true
if ! BOOTSTRAP_TMP="$(mktemp "${CONTROL_PLANE_TMPDIR}/bootstrap.XXXXXXXXXX")"; then
  echo "ERROR: cannot create bootstrap temp file under $CONTROL_PLANE_TMPDIR" >&2
  exit 1
fi
cat > "$BOOTSTRAP_TMP" <<'BOOTSTRAP'
#!/usr/bin/env bash
set -Eeuo pipefail

CONTROL_PLANE_ROOT="${CONTROL_PLANE_ROOT:-}"
CONTROL_PLANE_DIR="${CONTROL_PLANE_DIR:-${CONTROL_DIR:-}}"
GLM51_ENV_FILE="${GLM51_ENV_FILE:-${CONTROL_PLANE_DIR:-/opt/glm51}/glm51.env}"
if [ -r "$GLM51_ENV_FILE" ]; then
  set -a
  . "$GLM51_ENV_FILE"
  set +a
fi
GLM51_SECRETS_FILE="${GLM51_SECRETS_FILE:-${CONTROL_PLANE_DIR:-/opt/glm51}/secrets/glm51-secrets.env}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-${CONTROL_PLANE_DIR:-/opt/glm51}/secrets/hf_token.env}"
for secret_file in "$GLM51_SECRETS_FILE" "$HF_TOKEN_FILE"; do
  if [ -r "$secret_file" ]; then
    set -a
    . "$secret_file"
    set +a
  fi
done
export GLM51_SECRETS_FILE HF_TOKEN_FILE
export ALLOW_REDOWNLOAD_ON_LOCAL_NVME_LOSS="${ALLOW_REDOWNLOAD_ON_LOCAL_NVME_LOSS:-1}"
control_root_writable() {
  local dir="$1" test_file
  [ -n "$dir" ] || return 1
  mkdir -p "$dir" 2>/dev/null || return 1
  test_file="${dir}/.glm51-control-rw.$$"
  touch "$test_file" 2>/dev/null || return 1
  rm -f "$test_file" 2>/dev/null || true
}
choose_control_dir() {
  local candidate dir
  if [ -n "${CONTROL_PLANE_DIR:-}" ]; then
    if control_root_writable "$CONTROL_PLANE_DIR"; then
      printf '%s
' "$CONTROL_PLANE_DIR"
      return 0
    fi
    printf '[bootstrap] warning: CONTROL_PLANE_DIR is not writable, falling back: %s
' "$CONTROL_PLANE_DIR" >&2
  fi
  if [ -n "${CONTROL_PLANE_ROOT:-}" ]; then
    dir="${CONTROL_PLANE_ROOT%/}/glm51-control"
    if control_root_writable "$dir"; then
      printf '%s
' "$dir"
      return 0
    fi
    printf '[bootstrap] warning: CONTROL_PLANE_ROOT is not writable, falling back: %s
' "$CONTROL_PLANE_ROOT" >&2
  fi
  for candidate in /data /data2 /mnt; do
    [ -d "$candidate" ] || continue
    dir="${candidate}/glm51-control"
    if control_root_writable "$dir"; then
      printf '%s
' "$dir"
      return 0
    fi
  done
  printf '%s
' "/opt/glm51"
}
export CONTROL_PLANE_DIR="$(choose_control_dir)"
case "$CONTROL_PLANE_DIR" in
  */glm51-control) export CONTROL_PLANE_ROOT="${CONTROL_PLANE_DIR%/glm51-control}" ;;
  /opt/glm51) export CONTROL_PLANE_ROOT="/opt" ;;
  *) export CONTROL_PLANE_ROOT="$(dirname "$CONTROL_PLANE_DIR")" ;;
esac
export CONTROL_DIR="$CONTROL_PLANE_DIR"
export GLM51_OPT_DIR="$CONTROL_PLANE_DIR"
export CONTROL_PLANE_TMPDIR="${CONTROL_PLANE_DIR}/tmp"
mkdir -p "$CONTROL_PLANE_TMPDIR" 2>/dev/null || { echo "ERROR: cannot create control-plane temp dir: $CONTROL_PLANE_TMPDIR" >&2; exit 1; }
export TMPDIR="$CONTROL_PLANE_TMPDIR"
touch "$TMPDIR/.glm51-tmpdir-rw.$$" 2>/dev/null || { echo "ERROR: control-plane temp dir is not writable: $TMPDIR" >&2; exit 1; }
rm -f "$TMPDIR/.glm51-tmpdir-rw.$$" 2>/dev/null || true
export GLM51_ENV_FILE="${GLM51_ENV_FILE:-${CONTROL_PLANE_DIR}/glm51.env}"
export GLM51_SECRETS_FILE="${GLM51_SECRETS_FILE:-${CONTROL_PLANE_DIR}/secrets/glm51-secrets.env}"
export HF_TOKEN_FILE="${HF_TOKEN_FILE:-${CONTROL_PLANE_DIR}/secrets/hf_token.env}"
for secret_file in "$GLM51_SECRETS_FILE" "$HF_TOKEN_FILE"; do
  if [ -r "$secret_file" ]; then
    set -a
    . "$secret_file"
    set +a
  fi
done
export GENERATED_SCRIPT_DIR="${GENERATED_SCRIPT_DIR:-${CONTROL_PLANE_DIR}/generated}"
export ATOM_REPO_URL="${ATOM_REPO_URL:-https://github.com/ROCm/ATOM.git}"
export ATOM_REF="${ATOM_REF:-9427621d3dfdfac1c7820bb435b6d034083686ee}"
export ATOM_REPO_ROOT="${ATOM_REPO_ROOT:-${CONTROL_DIR}/ATOM}"
export ATOM_REPO_HOST="${ATOM_REPO_HOST:-$ATOM_REPO_ROOT}"
export ATOM_REPO_CONTAINER="${ATOM_REPO_CONTAINER:-/opt/ATOM}"
export ATOM_PLUGIN_PYTHONPATH="${ATOM_PLUGIN_PYTHONPATH:-/sgl-workspace/sglang/python:/opt/ATOM}"

export GLM51_SCRIPT_VERSION="${GLM51_SCRIPT_VERSION:-markdown-20260602-sglang-quark-loader-patch-v3.13}"
export CONTROL_PLANE_DIR="${CONTROL_PLANE_DIR:-${CONTROL_DIR:-/opt/glm51}}"
export CONTROL_DIR="${CONTROL_DIR:-$CONTROL_PLANE_DIR}"
export GLM51_OPT_DIR="${GLM51_OPT_DIR:-$CONTROL_PLANE_DIR}"
export CONTROL_PLANE_TMPDIR="${CONTROL_PLANE_TMPDIR:-${CONTROL_PLANE_DIR}/tmp}"
mkdir -p "$CONTROL_PLANE_TMPDIR" 2>/dev/null || { echo "ERROR: cannot create control-plane temp dir: $CONTROL_PLANE_TMPDIR" >&2; exit 1; }
export TMPDIR="$CONTROL_PLANE_TMPDIR"
touch "$TMPDIR/.glm51-tmpdir-rw.$$" 2>/dev/null || { echo "ERROR: control-plane temp dir is not writable: $TMPDIR" >&2; exit 1; }
rm -f "$TMPDIR/.glm51-tmpdir-rw.$$" 2>/dev/null || true

export MOUNT_POINT="${MOUNT_POINT:-/local_nvme}"
export MD_DEV="${MD_DEV:-/dev/md0}"
export RAID_DEVICES="${RAID_DEVICES:-2}"
export HOST_WORKDIR="${HOST_WORKDIR:-${MOUNT_POINT}/amd_profiling}"
export CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-${MOUNT_POINT}/amd_profiling}"
export HOST_TMPDIR="${HOST_TMPDIR:-${HOST_WORKDIR}/host-tmp}"
export TMUX_TMPDIR="${TMUX_TMPDIR:-${CONTROL_PLANE_DIR}/tmux}"
export INSTALL_SYSTEMD_AUTOSTART="${INSTALL_SYSTEMD_AUTOSTART:-1}"
export AUTOSTART_SERVICE_NAME="${AUTOSTART_SERVICE_NAME:-glm51-autostart}"
export AUTOSTART_CHECK_INTERVAL_SECONDS="${AUTOSTART_CHECK_INTERVAL_SECONDS:-60}"
export AUTO_FREE_SPACE="${AUTO_FREE_SPACE:-1}"
export AUTO_DELETE_MODEL_IF_LOW_SPACE="${AUTO_DELETE_MODEL_IF_LOW_SPACE:-1}"
export MIN_HOST_FREE_GB="${MIN_HOST_FREE_GB:-5}"
export HF_DOWNLOAD_MAX_WORKERS="${HF_DOWNLOAD_MAX_WORKERS:-8}"
export HF_TOKEN="${HF_TOKEN:-}"
export PERSIST_LOG_ROOT="${PERSIST_LOG_ROOT:-}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-glm51-fp8-atom-pr355-oot}"
export LOG_TIMEZONE="${LOG_TIMEZONE:-Asia/Shanghai}"
export RUN_ID="${RUN_ID:-}"
export FIRST_RUN_LOG_COPY_TIMEOUT_SECONDS="${FIRST_RUN_LOG_COPY_TIMEOUT_SECONDS:-10}"
export COPY_FIRST_RUN_LOG_TO_LOCAL_NVME="${COPY_FIRST_RUN_LOG_TO_LOCAL_NVME:-0}"
export PATCH_SGLANG_QUARK_FUSED_QKV_A_PROJ="${PATCH_SGLANG_QUARK_FUSED_QKV_A_PROJ:-1}"

persist_root_writable() {
  local dir="$1" test_file
  [ -n "$dir" ] || return 1
  [ -d "$dir" ] || return 1
  test_file="${dir}/.glm51-persist-log-rw.$$"
  touch "$test_file" 2>/dev/null || return 1
  rm -f "$test_file" 2>/dev/null || true
}

choose_persist_log_root() {
  local candidate
  if [ -n "${PERSIST_LOG_ROOT:-}" ]; then
    if persist_root_writable "$PERSIST_LOG_ROOT"; then
      printf '%s
' "$PERSIST_LOG_ROOT"
      return 0
    fi
    printf '[bootstrap] warning: PERSIST_LOG_ROOT is not writable, falling back: %s
' "$PERSIST_LOG_ROOT" >&2
  fi
  for candidate in "${CONTROL_PLANE_DIR:-}" /data /data2 /mnt /opt/glm51; do
    if persist_root_writable "$candidate"; then
      printf '%s
' "$candidate"
      return 0
    fi
  done
  printf '%s
' "/opt/glm51"
}


safe_experiment_name() {
  local raw safe
  raw="${1:-experiment}"
  safe="${raw//[^A-Za-z0-9_.-]/-}"
  [ -n "$safe" ] || safe="experiment"
  printf '%s
' "$safe"
}

ensure_run_logging_metadata() {
  local safe_name tz_label ts
  safe_name="$(safe_experiment_name "${EXPERIMENT_NAME:-glm51-fp8-atom-pr355-oot}")"
  tz_label="$(printf '%s' "${LOG_TIMEZONE:-Asia/Shanghai}" | tr -cd 'A-Za-z0-9')"
  [ -n "$tz_label" ] || tz_label="UTC"
  if [ -z "${RUN_ID:-}" ]; then
    if ts="$(TZ="${LOG_TIMEZONE:-Asia/Shanghai}" date '+%Y%m%d-%H%M%S' 2>/dev/null)"; then
      RUN_ID="${ts}-${tz_label}-${safe_name}"
    else
      ts="$(date -u '+%Y%m%d-%H%M%S')"
      RUN_ID="${ts}-UTC-${safe_name}"
    fi
  fi
  RUNS_DIR="${PERSIST_LOG_DIR}/runs"
  RUN_DIR="${RUNS_DIR}/${RUN_ID}"
  mkdir -p "$RUN_DIR" 2>/dev/null || true
  ln -sfn "runs/${RUN_ID}" "${PERSIST_LOG_DIR}/latest" 2>/dev/null || true
  export EXPERIMENT_NAME LOG_TIMEZONE RUN_ID RUNS_DIR RUN_DIR
}

setup_persistent_logging() {
  local selected fallback
  fallback="${CONTROL_PLANE_DIR:-/opt/glm51}/logs"
  selected="$(choose_persist_log_root)"
  PERSIST_LOG_ROOT="$selected"
  PERSIST_LOG_DIR="${PERSIST_LOG_DIR:-${PERSIST_LOG_ROOT}/glm51-logs}"
  if ! mkdir -p "$PERSIST_LOG_DIR" 2>/dev/null || ! persist_root_writable "$PERSIST_LOG_DIR"; then
    PERSIST_LOG_ROOT="${CONTROL_PLANE_DIR:-/opt/glm51}"
    PERSIST_LOG_DIR="$fallback"
    mkdir -p "$PERSIST_LOG_DIR" 2>/dev/null || true
    printf '[bootstrap] warning: persistent log dir unavailable, using OS disk fallback: %s
' "$PERSIST_LOG_DIR" >&2
  fi
  export PERSIST_LOG_ROOT PERSIST_LOG_DIR
  ensure_run_logging_metadata
  BOOTSTRAP_LOG="${PERSIST_LOG_DIR}/bootstrap.log"
  RUN_BOOTSTRAP_LOG="${RUN_DIR}/bootstrap.log"
  FIRST_RUN_CHECK_LOG_PERSIST="${PERSIST_LOG_DIR}/first-run-check.log"
  RUN_FIRST_RUN_CHECK_LOG_PERSIST="${RUN_DIR}/first-run-check.log"
  export BOOTSTRAP_LOG RUN_BOOTSTRAP_LOG FIRST_RUN_CHECK_LOG_PERSIST RUN_FIRST_RUN_CHECK_LOG_PERSIST
  exec > >(tee -a "$BOOTSTRAP_LOG" "$RUN_BOOTSTRAP_LOG") 2>&1
  echo "[bootstrap] experiment_name=${EXPERIMENT_NAME} run_id=${RUN_ID} log_timezone=${LOG_TIMEZONE} persistent_log_root=${PERSIST_LOG_ROOT} persistent_log_dir=${PERSIST_LOG_DIR} run_dir=${RUN_DIR}"
}

setup_persistent_logging

echo "[bootstrap] GLM51_SCRIPT_VERSION=${GLM51_SCRIPT_VERSION}"
echo "[bootstrap] pid=$$ user=$(id -un 2>/dev/null || true) pwd=$(pwd)"

print_first_run_preflight() {
  echo "[first-run-check] begin detailed first-run diagnostics"
  echo "[first-run-check] script_version=${GLM51_SCRIPT_VERSION} mount_point=${MOUNT_POINT} host_workdir=${HOST_WORKDIR}"

  echo "[first-run-check] host identity"
  hostname || true
  uname -a || true
  id || true
  date || true

  echo "[first-run-check] Azure instance compute metadata (first 4096 bytes; failure ignored)"
  (curl -fsS --max-time 3 -H Metadata:true 'http://169.254.169.254/metadata/instance/compute?api-version=2021-02-01' 2>&1 | head -c 4096 || true)
  echo

  echo "[first-run-check] Azure scheduled events (first 4096 bytes; failure ignored)"
  (curl -fsS --max-time 3 -H Metadata:true 'http://169.254.169.254/metadata/scheduledevents?api-version=2020-07-01' 2>&1 | head -c 4096 || true)
  echo

  echo "[first-run-check] block devices"
  lsblk -o NAME,TYPE,SIZE,FSTYPE,MOUNTPOINT,MODEL,SERIAL 2>/dev/null || lsblk -o NAME,TYPE,SIZE,FSTYPE,MOUNTPOINT,MODEL 2>/dev/null || true

  echo "[first-run-check] mounts"
  findmnt / 2>/dev/null || true
  findmnt "$MOUNT_POINT" 2>/dev/null || true

  echo "[first-run-check] filesystem free space"
  df -h / "$MOUNT_POINT" 2>/dev/null || true

  echo "[first-run-check] docker summary"
  if command -v docker >/dev/null 2>&1; then
    sudo docker info --format 'Server Version: {{.ServerVersion}}  Docker Root Dir: {{.DockerRootDir}}' 2>/dev/null || true
  else
    echo "[first-run-check] docker missing"
  fi

  echo "[first-run-check] docker/containerd systemd state"
  if command -v systemctl >/dev/null 2>&1; then
    systemctl is-active docker containerd 2>/dev/null || true
    (systemctl status docker containerd --no-pager -l 2>&1 || true) | head -n 80 || true
  else
    echo "[first-run-check] systemctl missing"
  fi

  echo "[first-run-check] ROCm tools"
  if command -v rocm-smi >/dev/null 2>&1; then
    echo "[first-run-check] rocm-smi available"
    rocm-smi 2>&1 | head -n 80 || true
  else
    echo "[first-run-check] rocm-smi missing"
  fi
  if command -v rocminfo >/dev/null 2>&1; then
    echo "[first-run-check] rocminfo available"
    rocminfo 2>&1 | head -n 80 || true
  else
    echo "[first-run-check] rocminfo missing"
  fi

  echo "[first-run-check] GPU device nodes"
  ls -ld /dev/kfd /dev/dri /dev/dri/* 2>/dev/null || true

  echo "[first-run-check] tmux/curl versions"
  if command -v tmux >/dev/null 2>&1; then
    tmux -V || true
  else
    echo "[first-run-check] tmux missing"
  fi
  if command -v curl >/dev/null 2>&1; then
    curl --version 2>/dev/null | head -n 1 || true
  else
    echo "[first-run-check] curl missing"
  fi
  echo "[first-run-check] end detailed first-run diagnostics"
  true
}

print_post_mount_storage_status() {
  echo "[first-run-check] post-mount storage status"
  findmnt "$MOUNT_POINT" 2>/dev/null || true
  df -h / "$MOUNT_POINT" 2>/dev/null || true
  lsblk -o NAME,TYPE,SIZE,FSTYPE,MOUNTPOINT,MODEL,SERIAL 2>/dev/null || lsblk -o NAME,TYPE,SIZE,FSTYPE,MOUNTPOINT,MODEL 2>/dev/null || true
  true
}

copy_first_run_log_best_effort() {
  local src="$1" dst="$2" label="$3" timeout_seconds
  timeout_seconds="${FIRST_RUN_LOG_COPY_TIMEOUT_SECONDS:-10}"
  echo "[preflight] copying first-run log to ${label}: ${dst}"
  if [ -r "$src" ]; then
    wc -c "$src" 2>/dev/null || ls -lh "$src" 2>/dev/null || true
  else
    echo "[preflight] warning: first-run log source is not readable: $src" >&2
    return 0
  fi
  mkdir -p "$(dirname "$dst")" 2>/dev/null || true
  if command -v timeout >/dev/null 2>&1; then
    if ! timeout "$timeout_seconds" cp "$src" "$dst" 2>/dev/null; then
      echo "[preflight] warning: first-run log copy failed or timed out after ${timeout_seconds}s: ${label} -> ${dst}" >&2
    fi
  else
    echo "[preflight] warning: timeout command missing; copying first-run log without timeout: ${label} -> ${dst}" >&2
    cp "$src" "$dst" 2>/dev/null || echo "[preflight] warning: first-run log copy failed: ${label} -> ${dst}" >&2
  fi
}

mount_ready() {
  findmnt "$MOUNT_POINT" >/dev/null 2>&1 && [ -d "$MOUNT_POINT" ] && touch "$MOUNT_POINT/.rw-test" 2>/dev/null && rm -f "$MOUNT_POINT/.rw-test"
}

mount_md_if_present() {
  [ -b "$MD_DEV" ] || return 1
  sudo mkdir -p "$MOUNT_POINT" || return 1
  sudo mount "$MD_DEV" "$MOUNT_POINT" 2>/dev/null || true
  mount_ready
}

md_member_candidates() {
  local dev name type
  while read -r name type; do
    dev="/dev/${name}"
    [ "$type" = "disk" ] || continue
    [[ "$name" == nvme*n1 ]] || continue
    lsblk -nr -o TYPE "$dev" 2>/dev/null | grep -q '^part$' && continue
    sudo mdadm --examine "$dev" >/dev/null 2>&1 || continue
    printf '%s
' "$dev"
  done < <(lsblk -dn -o NAME,TYPE 2>/dev/null)
}

assemble_existing_md() {
  local members=()
  mount_md_if_present && return 0
  mapfile -t members < <(md_member_candidates)
  [ "${#members[@]}" -gt 0 ] || return 1
  echo "[preflight] assembling $MD_DEV from md superblock members: ${members[*]}"
  sudo mdadm --assemble "$MD_DEV" "${members[@]}" --run >/dev/null 2>&1 || true
  mount_md_if_present
}

blank_nvme_candidates() {
  local dev name type fstype mountpoint sig
  while read -r name type fstype mountpoint; do
    dev="/dev/${name}"
    [ "$type" = "disk" ] || continue
    [[ "$name" == nvme*n1 ]] || continue
    [ -z "$fstype" ] || continue
    [ -z "$mountpoint" ] || continue
    lsblk -nr -o TYPE "$dev" 2>/dev/null | grep -q '^part$' && continue
    sig="$(sudo wipefs -n "$dev" 2>/dev/null || true)"
    [ -z "$sig" ] || continue
    sudo blkid "$dev" >/dev/null 2>&1 && continue
    sudo mdadm --examine "$dev" >/dev/null 2>&1 && continue
    printf '%s
' "$dev"
  done < <(lsblk -dn -o NAME,TYPE,FSTYPE,MOUNTPOINT 2>/dev/null)
}

validate_blank_nvme() {
  local dev="$1" sig
  lsblk -dn -o TYPE "$dev" 2>/dev/null | grep -qx disk || { echo "ERROR: not a disk: $dev" >&2; return 1; }
  [[ "$(basename "$dev")" == nvme*n1 ]] || { echo "ERROR: refusing non-NVMe disk: $dev" >&2; return 1; }
  lsblk -nr -o TYPE "$dev" 2>/dev/null | grep -q '^part$' && { echo "ERROR: refusing to format $dev because it has partitions." >&2; return 1; }
  lsblk -dn -o MOUNTPOINT "$dev" 2>/dev/null | grep -q . && { echo "ERROR: refusing to format $dev because it is mounted." >&2; return 1; }
  lsblk -dn -o FSTYPE "$dev" 2>/dev/null | grep -q . && { echo "ERROR: refusing to format $dev because it has a filesystem/RAID type." >&2; return 1; }
  sig="$(sudo wipefs -n "$dev" 2>/dev/null || true)"
  [ -z "$sig" ] || { echo "ERROR: refusing to format $dev because wipefs sees existing signatures." >&2; return 1; }
  sudo blkid "$dev" >/dev/null 2>&1 && { echo "ERROR: refusing to format $dev because blkid sees existing content." >&2; return 1; }
  sudo mdadm --examine "$dev" >/dev/null 2>&1 && { echo "ERROR: refusing to format $dev because mdadm sees an existing md signature." >&2; return 1; }
  return 0
}

rebuild_blank_nvme_raid() {
  local disks=()
  if [ -n "${NVME_DEVS:-}" ]; then
    read -r -a disks <<< "$NVME_DEVS"
  else
    mapfile -t disks < <(blank_nvme_candidates | head -n "$RAID_DEVICES")
  fi
  if [ "${#disks[@]}" -ne "$RAID_DEVICES" ]; then
    echo "ERROR: need exactly $RAID_DEVICES blank, unmounted, unpartitioned NVMe disks; found: ${disks[*]:-(none)}" >&2
    echo "No md superblock could be assembled; refusing destructive rebuild without safe blank disks." >&2
    return 1
  fi
  local dev
  for dev in "${disks[@]}"; do
    validate_blank_nvme "$dev" || return 1
  done
  echo "[preflight] creating RAID0 $MD_DEV from blank NVMe disks: ${disks[*]}"
  sudo mdadm --stop "$MD_DEV" >/dev/null 2>&1 || true
  sudo mdadm --create "$MD_DEV" --level=0 --raid-devices="$RAID_DEVICES" --force "${disks[@]}"
  echo "[preflight] formatting $MD_DEV as XFS"
  sudo mkfs.xfs -f "$MD_DEV"
  sudo mkdir -p "$MOUNT_POINT" || return 1
  sudo mount "$MD_DEV" "$MOUNT_POINT"
  sudo chown "$(id -u):$(id -g)" "$MOUNT_POINT" || true
  mount_ready
}

mount_local_nvme_if_needed() {
  local mnt_source="" mnt_target=""
  mnt_source="$(findmnt -n -o SOURCE -T "$MOUNT_POINT" 2>/dev/null || true)"
  mnt_target="$(findmnt -n -o TARGET -T "$MOUNT_POINT" 2>/dev/null || true)"
  if [ "$mnt_target" = "$MOUNT_POINT" ]; then
    case "$mnt_source" in
      "$MD_DEV"|/dev/md*|/dev/nvme*|/dev/mapper/*)
        if mount_ready; then
          echo "[preflight] $MOUNT_POINT already mounted from $mnt_source and writable"
          return 0
        fi
        ;;
    esac
  fi

  command -v mkfs.xfs >/dev/null 2>&1 || { echo "ERROR: mkfs.xfs is missing. Cannot format NVMe as XFS." >&2; exit 1; }
  command -v mdadm >/dev/null 2>&1 || { echo "ERROR: mdadm is missing. Cannot create/assemble RAID0." >&2; exit 1; }

  if assemble_existing_md; then
    echo "[preflight] $MOUNT_POINT mounted and writable after targeted md assemble"
    return 0
  fi

  echo "[preflight] no usable md superblock assembled; checking safe blank NVMe rebuild path"
  rebuild_blank_nvme_raid || exit 1
}
gb_to_kb() {
  awk -v gb="$1" 'BEGIN { printf "%.0f", gb * 1024 * 1024 }'
}

free_kb_for_path() {
  df -Pk "$1" 2>/dev/null | awk 'NR==2 {print $4}'
}

print_bootstrap_storage() {
  echo "[preflight] storage diagnostics"
  lsblk -o NAME,TYPE,SIZE,FSTYPE,MOUNTPOINT,MODEL,SERIAL 2>/dev/null || lsblk -o NAME,TYPE,SIZE,FSTYPE,MOUNTPOINT,MODEL 2>/dev/null || true
  findmnt "$MOUNT_POINT" 2>/dev/null || true
  df -h / "$MOUNT_POINT" "$HOST_WORKDIR" 2>/dev/null || true
  df -ih / "$MOUNT_POINT" "$HOST_WORKDIR" 2>/dev/null || true
  if command -v docker >/dev/null 2>&1; then
    sudo docker info --format 'Server Version: {{.ServerVersion}}  Docker Root Dir: {{.DockerRootDir}}' 2>/dev/null || true
    sudo docker system df 2>/dev/null || true
  fi
}

print_bootstrap_platform_hints() {
  echo "[preflight] platform diagnostics"
  command -v rocm-smi >/dev/null 2>&1 && rocm-smi 2>/dev/null || true
  command -v rocminfo >/dev/null 2>&1 && rocminfo 2>/dev/null | sed -n '1,80p' || true
  echo "[preflight] Azure scheduled events check: curl -fsS -H Metadata:true 'http://169.254.169.254/metadata/scheduledevents?api-version=2020-07-01'"
  curl -fsS -H Metadata:true 'http://169.254.169.254/metadata/scheduledevents?api-version=2020-07-01' 2>/dev/null || true
}

free_space_aggressively() {
  [ "$AUTO_FREE_SPACE" = "1" ] || return 0
  echo "[preflight] aggressive space cleanup enabled"

  if command -v docker >/dev/null 2>&1; then
    sudo docker rm -f $(sudo docker ps -aq) >/dev/null 2>&1 || true
    sudo docker builder prune -af >/dev/null 2>&1 || true
    sudo docker container prune -f >/dev/null 2>&1 || true
    sudo docker image prune -af >/dev/null 2>&1 || true
    sudo docker volume prune -f >/dev/null 2>&1 || true
    sudo docker system prune -af --volumes >/dev/null 2>&1 || true
  fi

  rm -rf "${HOST_WORKDIR}/docker-contexts" \
         "${HOST_WORKDIR}/.glm51_resume_state" \
         "${HOST_WORKDIR}/lm_eval/results_"* \
         "${HOST_WORKDIR}/lm_eval/"*.log \
         "${HOST_WORKDIR}/hf-cache" \
         "${HOST_WORKDIR}/.cache" 2>/dev/null || true
}

ensure_bootstrap_space() {
  local min_kb avail_kb model_host
  min_kb="$(gb_to_kb "$MIN_HOST_FREE_GB")"
  avail_kb="$(free_kb_for_path "$HOST_WORKDIR")"
  if [ -n "$avail_kb" ] && [ "$avail_kb" -ge "$min_kb" ]; then
    return 0
  fi

  print_bootstrap_storage
  free_space_aggressively
  avail_kb="$(free_kb_for_path "$HOST_WORKDIR")"
  if [ -n "$avail_kb" ] && [ "$avail_kb" -ge "$min_kb" ]; then
    print_bootstrap_storage
    return 0
  fi

  if [ "${AUTO_DELETE_MODEL_IF_LOW_SPACE:-1}" = "1" ]; then
    model_host="${MODEL_DIR/#$CONTAINER_WORKDIR/$HOST_WORKDIR}"
    echo "[preflight] still low on space; deleting model directory: $model_host"
    rm -rf "$model_host" 2>/dev/null || true
  fi

  print_bootstrap_storage
  avail_kb="$(free_kb_for_path "$HOST_WORKDIR")"
  if [ -z "$avail_kb" ] || [ "$avail_kb" -lt "$min_kb" ]; then
    echo "ERROR: not enough free space in $HOST_WORKDIR after cleanup" >&2
    exit 1
  fi
}

FIRST_RUN_CHECK_LOG="${FIRST_RUN_CHECK_LOG:-/tmp/glm51-first-run-check.log}"
FIRST_RUN_CHECK_LOG_PERSIST="${FIRST_RUN_CHECK_LOG_PERSIST:-${PERSIST_LOG_DIR:-/opt/glm51/logs}/first-run-check.log}"
RUN_FIRST_RUN_CHECK_LOG_PERSIST="${RUN_FIRST_RUN_CHECK_LOG_PERSIST:-${RUN_DIR:-${PERSIST_LOG_DIR:-/opt/glm51/logs}}/first-run-check.log}"
mkdir -p "$(dirname "$FIRST_RUN_CHECK_LOG_PERSIST")" "$(dirname "$RUN_FIRST_RUN_CHECK_LOG_PERSIST")" 2>/dev/null || true
: > "$FIRST_RUN_CHECK_LOG" 2>/dev/null || true
: > "$FIRST_RUN_CHECK_LOG_PERSIST" 2>/dev/null || true
: > "$RUN_FIRST_RUN_CHECK_LOG_PERSIST" 2>/dev/null || true
print_first_run_preflight 2>&1 | tee -a "$FIRST_RUN_CHECK_LOG" "$FIRST_RUN_CHECK_LOG_PERSIST" "$RUN_FIRST_RUN_CHECK_LOG_PERSIST" || true
mount_local_nvme_if_needed
print_post_mount_storage_status 2>&1 | tee -a "$FIRST_RUN_CHECK_LOG" "$FIRST_RUN_CHECK_LOG_PERSIST" "$RUN_FIRST_RUN_CHECK_LOG_PERSIST" || true
echo "[preflight] after post-mount storage status completed"
echo "[preflight] creating generated script dir on persistent control disk"
sudo mkdir -p "$GENERATED_SCRIPT_DIR"
sudo chown "$(id -u):$(id -g)" "$GENERATED_SCRIPT_DIR" || true
chmod 0755 "$GENERATED_SCRIPT_DIR" || true
echo "[preflight] creating host workdir/tmux dir"
mkdir -p "$HOST_WORKDIR" "$TMUX_TMPDIR"
chmod 700 "$TMUX_TMPDIR"
cd "$HOST_WORKDIR"
echo "[preflight] copying first-run logs to persistent storage with timeout=${FIRST_RUN_LOG_COPY_TIMEOUT_SECONDS:-10}s"
copy_first_run_log_best_effort "$FIRST_RUN_CHECK_LOG" "$FIRST_RUN_CHECK_LOG_PERSIST" "persistent total log"
copy_first_run_log_best_effort "$FIRST_RUN_CHECK_LOG" "$RUN_FIRST_RUN_CHECK_LOG_PERSIST" "persistent run log"
if [ "${COPY_FIRST_RUN_LOG_TO_LOCAL_NVME:-0}" = "1" ]; then
  echo "[preflight] optional COPY_FIRST_RUN_LOG_TO_LOCAL_NVME=1; copying convenience first-run log to local NVMe"
  copy_first_run_log_best_effort "$FIRST_RUN_CHECK_LOG" "${HOST_WORKDIR}/first-run-check.log" "optional local NVMe workdir log"
else
  echo "[preflight] skip optional /local_nvme first-run log copy; authoritative logs: ${FIRST_RUN_CHECK_LOG_PERSIST} and ${RUN_FIRST_RUN_CHECK_LOG_PERSIST}"
fi
echo "[preflight] checking bootstrap free space"
ensure_bootstrap_space
echo "[preflight] writing generated resume/serve/lmeval scripts to ${GENERATED_SCRIPT_DIR}"
mkdir -p "$GENERATED_SCRIPT_DIR"
echo "$GLM51_SCRIPT_VERSION" > "${GENERATED_SCRIPT_DIR}/.glm51_script_version"
rm -f \
  "${GENERATED_SCRIPT_DIR}/glm51_resume.sh" \
  "${GENERATED_SCRIPT_DIR}/glm51_serve.sh" \
  "${GENERATED_SCRIPT_DIR}/glm51_download_only.sh" \
  "${GENERATED_SCRIPT_DIR}/glm51_build_only.sh" \
  "${GENERATED_SCRIPT_DIR}/glm51_lmeval.sh"
# Do not remove or rewrite ${HOST_WORKDIR}/glm51_*.sh here; /local_nvme I/O can block startup.

cat > "${GENERATED_SCRIPT_DIR}/glm51_resume.sh" <<'BASH'
#!/usr/bin/env bash
set -Eeuo pipefail

export GLM51_SCRIPT_VERSION="${GLM51_SCRIPT_VERSION:-markdown-20260602-sglang-quark-loader-patch-v3.13}"
echo "[resume] GLM51_SCRIPT_VERSION=${GLM51_SCRIPT_VERSION} script=${BASH_SOURCE[0]:-$0} pid=$$ pwd=$(pwd)"

########################################
# 0. 可调环境变量
########################################

# BASE_IMAGE 用于下载/校验 Hugging Face 模型，并作为 Tinker merge/quant 的 ROCm 运行时；serve 使用 SGLANG_IMAGE。
export BASE_IMAGE="${BASE_IMAGE:-rocm/atom-dev@sha256:9be7af4ec2b5eed8826521db5719e9610ce03f784fb49cc15effb1f2584192eb}"
export GLM51_SCRIPT_VERSION="${GLM51_SCRIPT_VERSION:-markdown-20260602-sglang-quark-loader-patch-v3.13}"
export SGLANG_IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang-rocm:v0.5.12.post1-rocm720-mi30x-20260529}"
export SGLANG_CONTAINER="${SGLANG_CONTAINER:-sglang-glm51-fp8-atom-pr355}"
export LMEVAL_IMAGE="${LMEVAL_IMAGE:-lm-eval-harness:latest}"
export LMEVAL_CONTAINER="${LMEVAL_CONTAINER:-lm-eval-glm51-atom-pr355}"
export CONTROL_DIR="${CONTROL_DIR:-/data/glm51-control}"
export GENERATED_SCRIPT_DIR="${GENERATED_SCRIPT_DIR:-${CONTROL_DIR}/generated}"

export HOST_WORKDIR="${HOST_WORKDIR:-/local_nvme/amd_profiling}"
export CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-/local_nvme/amd_profiling}"
export HOST_TMPDIR="${HOST_TMPDIR:-${HOST_WORKDIR}/host-tmp}"
export MODEL_ID="${MODEL_ID:-zai-org/GLM-5.1-FP8}"
export MODEL_REVISION="${MODEL_REVISION:-f396cf805182f4ca10fa675e1a99815b3ca384db}"
export MODEL_DIR="${MODEL_DIR:-${CONTAINER_WORKDIR}/models/GLM-5.1-FP8}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL_ID}}"
export RUN_TINKER_MERGE_QUANT="${RUN_TINKER_MERGE_QUANT:-auto}"
export GPU_LEASE_BASE_URL="${GPU_LEASE_BASE_URL:-https://eval-service.macaron.im}"
export GPU_LEASE_API_KEY="${GPU_LEASE_API_KEY:-}"
export TINKER_URL="${TINKER_URL:-<这里填 tinker://.../weights/...>}"
export LORA_DIR="${LORA_DIR:-${CONTAINER_WORKDIR}/loras/tinker-lora}"
export MERGED_MODEL_DIR="${MERGED_MODEL_DIR:-${CONTAINER_WORKDIR}/models/GLM-5.1-FP8-tinker-merged}"
export QUANT_MODEL_DIR="${QUANT_MODEL_DIR:-${CONTAINER_WORKDIR}/models/GLM-5.1-FP8-tinker-merged-fp8-dynamic}"
export QUANT_SCHEME="${QUANT_SCHEME:-FP8_DYNAMIC}"
export GLM5_REFERENCE_SCRIPTS_DIR="${GLM5_REFERENCE_SCRIPTS_DIR:-${CONTROL_DIR}/glm5-fp8-deploy/scripts}"
export GLM5_MERGE_QUANT_GPUS="${GLM5_MERGE_QUANT_GPUS:-0,1,2,3,4,5,6,7}"
export GLM5_MERGE_QUANT_WORKERS_PER_GPU="${GLM5_MERGE_QUANT_WORKERS_PER_GPU:-8}"
export GLM5_LOCAL_QUANT_ROOT="${GLM5_LOCAL_QUANT_ROOT:-${CONTAINER_WORKDIR}/glm5_local_quant}"
export SGLANG_PORT="${SGLANG_PORT:-7777}"
export ATOM_REPO_URL="${ATOM_REPO_URL:-https://github.com/ROCm/ATOM.git}"
export ATOM_REF="${ATOM_REF:-9427621d3dfdfac1c7820bb435b6d034083686ee}"
export ATOM_REPO_ROOT="${ATOM_REPO_ROOT:-${CONTROL_DIR}/ATOM}"
export ATOM_REPO_HOST="${ATOM_REPO_HOST:-$ATOM_REPO_ROOT}"
export ATOM_REPO_CONTAINER="${ATOM_REPO_CONTAINER:-/opt/ATOM}"
export ATOM_PLUGIN_PYTHONPATH="${ATOM_PLUGIN_PYTHONPATH:-/sgl-workspace/sglang/python:/opt/ATOM}"
export SGLANG_EXTERNAL_MODEL_PACKAGE="${SGLANG_EXTERNAL_MODEL_PACKAGE:-atom.plugin.sglang.models}"
export AITER_QUICK_REDUCE_QUANTIZATION="${AITER_QUICK_REDUCE_QUANTIZATION:-INT4}"
export SGLANG_AITER_FP8_PREFILL_ATTN="${SGLANG_AITER_FP8_PREFILL_ATTN:-0}"
export PATCH_SGLANG_QUARK_FUSED_QKV_A_PROJ="${PATCH_SGLANG_QUARK_FUSED_QKV_A_PROJ:-1}"
export INSTALL_SYSTEMD_AUTOSTART="${INSTALL_SYSTEMD_AUTOSTART:-1}"
export AUTOSTART_SERVICE_NAME="${AUTOSTART_SERVICE_NAME:-glm51-autostart}"
export AUTOSTART_CHECK_INTERVAL_SECONDS="${AUTOSTART_CHECK_INTERVAL_SECONDS:-60}"
export ALLOW_REDOWNLOAD_ON_LOCAL_NVME_LOSS="${ALLOW_REDOWNLOAD_ON_LOCAL_NVME_LOSS:-1}"

if [ -z "${SGLANG_SERVE_ARGS:-}" ]; then
  read -r -d '' SGLANG_SERVE_ARGS <<'SGLANG_ARGS' || true
python3 -m sglang.launch_server \
  --model-path "$MODEL_DIR" \
  --host 0.0.0.0 \
  --port "$SGLANG_PORT" \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.8 \
  --page-size 1 \
  --reasoning-parser glm45 --tool-call-parser glm47
SGLANG_ARGS
fi
export SGLANG_SERVE_ARGS

export LMEVAL_WORKDIR="${LMEVAL_WORKDIR:-${CONTAINER_WORKDIR}/lm_eval}"

# 私有 / gated 模型通过远端 secret 文件注入；公开模型留空即可。不要把 token 长期写进笔记。
export GLM51_SECRETS_FILE="${GLM51_SECRETS_FILE:-${CONTROL_DIR}/secrets/glm51-secrets.env}"
export HF_TOKEN_FILE="${HF_TOKEN_FILE:-${CONTROL_DIR}/secrets/hf_token.env}"
for secret_file in "$GLM51_SECRETS_FILE" "$HF_TOKEN_FILE"; do
  if [ -r "$secret_file" ]; then
    set -a
    . "$secret_file"
    set +a
  fi
done
export HF_TOKEN="${HF_TOKEN:-}"

# Docker 缓存损坏时是否自动做轻量修复。
export AUTO_FIX_DOCKER_CACHE="${AUTO_FIX_DOCKER_CACHE:-1}"

# 空间治理：1=启动前自动清理 Docker/cache/local_nvme 下 eval 输出；0=只检查不清理。
export AUTO_FREE_SPACE="${AUTO_FREE_SPACE:-1}"
# 如果清完仍然空间不足，是否允许删除已有模型目录重新下载。
export AUTO_DELETE_MODEL_IF_LOW_SPACE="${AUTO_DELETE_MODEL_IF_LOW_SPACE:-1}"
export MIN_HOST_FREE_GB="${MIN_HOST_FREE_GB:-5}"
export MIN_MODEL_DOWNLOAD_FREE_GB="${MIN_MODEL_DOWNLOAD_FREE_GB:-120}"
export MIN_DOCKER_BUILD_FREE_GB="${MIN_DOCKER_BUILD_FREE_GB:-40}"
export HF_DOWNLOAD_MAX_WORKERS="${HF_DOWNLOAD_MAX_WORKERS:-8}"
export AUTO_KILL_STALE_DOWNLOADS="${AUTO_KILL_STALE_DOWNLOADS:-1}"

# 启动前是否自动杀掉并删除已有 Docker containers 来释放 GPU/端口/overlay 空间。
export AUTO_KILL_EXISTING_CONTAINERS="${AUTO_KILL_EXISTING_CONTAINERS:-1}"

# 启动前是否自动杀掉同名 tmux session 和旧 socket。注意：glm51_resume.sh 自己运行在 tmux 内时不会杀掉当前 session；
# 外层粘贴启动器会在创建新 session 前执行这一步。
export AUTO_KILL_EXISTING_TMUX_SESSION="${AUTO_KILL_EXISTING_TMUX_SESSION:-1}"

# 由最前面的环境变量块控制：0=不跑；1=serve ready 后自动在单独 tmux window 里跑 GSM8K lm_eval。
export RUN_LMEVAL="${RUN_LMEVAL:-0}"

# tmux orchestration. SGLang serve runs foreground in its own window, not docker -d.
export TMUX_TMPDIR="${TMUX_TMPDIR:-/local_nvme/tmux}"
export TMUX_SOCKET="${TMUX_SOCKET:-${TMUX_TMPDIR}/glm51.sock}"
export TMUX_SESSION="${TMUX_SESSION:-glm51}"
export GLM51_OPT_DIR="${GLM51_OPT_DIR:-/opt/glm51}"
export CONTROL_DIR="${CONTROL_DIR:-/data/glm51-control}"
export GENERATED_SCRIPT_DIR="${GENERATED_SCRIPT_DIR:-${CONTROL_DIR}/generated}"
export PREP_WINDOW="${PREP_WINDOW:-prep-download-build}"
export SERVE_WINDOW="${SERVE_WINDOW:-sglang-serve}"
export LMEVAL_WINDOW="${LMEVAL_WINDOW:-lm-eval}"

########################################
# 1. 工具函数
########################################

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

print_tmux_observe_guide() {
  echo
  echo "[observe] tmux attach:"
  echo "  sudo tmux -S $TMUX_SOCKET attach -t $TMUX_SESSION"
  echo "[observe] list windows:"
  echo "  sudo tmux -S $TMUX_SOCKET list-windows -t $TMUX_SESSION"
  echo "[observe] capture first-run-check/bootstrap log:"
  echo "  sudo tmux -S $TMUX_SOCKET capture-pane -pt ${TMUX_SESSION}:0 -S -3000 | grep -F '[first-run-check]' || true"
  echo "  sudo tmux -S $TMUX_SOCKET capture-pane -pt ${TMUX_SESSION}:resume -S -3000 | grep -F '[first-run-check]' || true"
  echo "[observe] capture recent prep log:"
  echo "  sudo tmux -S $TMUX_SOCKET capture-pane -pt ${TMUX_SESSION}:$PREP_WINDOW -S -200"
  echo "[observe] capture recent serve log:"
  echo "  sudo tmux -S $TMUX_SOCKET capture-pane -pt ${TMUX_SESSION}:$SERVE_WINDOW -S -200"
  echo "  sudo tail -200 /data/glm51-control/logs/sglang-serve.log"
  echo "  sudo tail -200 /data/glm51-control/logs/latest/sglang-serve.log"
  echo "[observe] capture recent lm_eval log:"
  echo "  sudo tmux -S $TMUX_SOCKET capture-pane -pt ${TMUX_SESSION}:$LMEVAL_WINDOW -S -200"
  echo "[observe] endpoint check:"
  echo "  curl -fsS http://127.0.0.1:${SGLANG_PORT}/v1/models"
  echo "[observe] state dir:"
  echo "  ${STATE_DIR}"
  echo
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "missing command: $1"
}

assert_dir_writable() {
  local dir="$1"
  local label="$2"
  local test_file="${dir}/.rw-test.$$"
  local err_file_base="${RUNTIME_DIR:-${dir}}"
  local err_file="${err_file_base}/rw_test.err"

  if ! [ -d "$dir" ]; then
    echo "ERROR: $label does not exist or is not a directory: $dir" >&2
    echo "Diagnostics:" >&2
    findmnt -T "$dir" 2>/dev/null >&2 || true
    ls -ld "$dir" "$(dirname "$dir")" 2>/dev/null >&2 || true
    exit 1
  fi

  mkdir -p "$err_file_base" 2>/dev/null || true
  if ! touch "$test_file" 2>"$err_file"; then
    echo "ERROR: $label is not writable: $dir" >&2
    echo "touch error:" >&2
    sed -n '1,40p' "$err_file" >&2 || true
    echo "Diagnostics:" >&2
    findmnt -T "$dir" 2>/dev/null >&2 || true
    ls -ld "$dir" "$(dirname "$dir")" 2>/dev/null >&2 || true
    exit 1
  fi

  rm -f "$test_file" || true
}

print_storage_diagnostics() {
  log "storage diagnostics"
  lsblk -o NAME,TYPE,SIZE,FSTYPE,MOUNTPOINT,MODEL 2>/dev/null || true
  findmnt "${HOST_WORKDIR:-$MOUNT_POINT}" 2>/dev/null || true
  df -h / "$MOUNT_POINT" "$HOST_WORKDIR" 2>/dev/null || true
  df -ih / "$MOUNT_POINT" "$HOST_WORKDIR" 2>/dev/null || true
  du -sh "${HOST_WORKDIR}"/* 2>/dev/null | sort -h | tail -30 || true
  if command -v docker >/dev/null 2>&1; then
    sudo docker info --format 'Docker Root Dir: {{.DockerRootDir}}' 2>/dev/null || true
    sudo docker system df 2>/dev/null || true
  fi
}

print_platform_diagnostics() {
  log "platform diagnostics (light; detailed first-run check is printed by outer bootstrap)"
  hostname || true
  uname -r || true
  ls -ld /dev/kfd /dev/dri /dev/dri/* 2>/dev/null || true
  if command -v rocm-smi >/dev/null 2>&1; then
    rocm-smi 2>&1 | head -n 40 || true
  else
    echo "rocm-smi missing"
  fi
  if command -v rocminfo >/dev/null 2>&1; then
    rocminfo 2>&1 | head -n 40 || true
  else
    echo "rocminfo missing"
  fi
}

gb_to_kb() {
  awk -v gb="$1" 'BEGIN { printf "%.0f", gb * 1024 * 1024 }'
}

assert_dir_has_space() {
  local dir="$1"
  local label="$2"
  local min_kb="${3:-1048576}"
  local avail_kb

  avail_kb="$(df -Pk "$dir" 2>/dev/null | awk 'NR==2 {print $4}')"
  if [ -z "$avail_kb" ]; then
    print_storage_diagnostics
    fail "cannot read free space for $label: $dir"
  fi
  if [ "$avail_kb" -lt "$min_kb" ]; then
    print_storage_diagnostics
    fail "$label has too little free space: $dir has ${avail_kb}KB available, need at least ${min_kb}KB"
  fi
}

cleanup_space_aggressively() {
  [ "$AUTO_FREE_SPACE" = "1" ] || return 0
  log "aggressive space cleanup enabled"

  sudo docker rm -f $(sudo docker ps -aq) >/dev/null 2>&1 || true
  sudo docker builder prune -af >/dev/null 2>&1 || true
  sudo docker container prune -f >/dev/null 2>&1 || true
  sudo docker image prune -af >/dev/null 2>&1 || true
  sudo docker volume prune -f >/dev/null 2>&1 || true
  sudo docker system prune -af --volumes >/dev/null 2>&1 || true

  rm -rf "${HOST_WORKDIR}/docker-contexts" \
         "${HOST_WORKDIR}/lm_eval/results_"* \
         "${HOST_WORKDIR}/lm_eval/"*.log \
         "${HOST_WORKDIR}/hf-cache" \
         "${HOST_WORKDIR}/.cache" \
         "${RUNTIME_DIR}" 2>/dev/null || true
  mkdir -p "$RUNTIME_DIR" 2>/dev/null || true
}

ensure_space_or_cleanup() {
  local dir="$1"
  local label="$2"
  local min_gb="$3"
  local allow_model_delete="${4:-0}"
  local min_kb

  min_kb="$(gb_to_kb "$min_gb")"
  if df -Pk "$dir" 2>/dev/null | awk -v min="$min_kb" 'NR==2 {exit !($4 >= min)}'; then
    return 0
  fi

  print_storage_diagnostics
  cleanup_space_aggressively
  if df -Pk "$dir" 2>/dev/null | awk -v min="$min_kb" 'NR==2 {exit !($4 >= min)}'; then
    print_storage_diagnostics
    return 0
  fi

  if [ "$allow_model_delete" = "1" ] && [ "$AUTO_DELETE_MODEL_IF_LOW_SPACE" = "1" ] && [ -d "$MODEL_DIR_HOST" ]; then
    log "still low on space; deleting model directory: $MODEL_DIR_HOST"
    rm -rf "$MODEL_DIR_HOST" 2>/dev/null || true
  fi

  print_storage_diagnostics
  assert_dir_has_space "$dir" "$label" "$min_kb"
}

host_path_for_container_path() {
  local p="$1"
  case "$p" in
    "$CONTAINER_WORKDIR"*) printf '%s%s\n' "$HOST_WORKDIR" "${p#"$CONTAINER_WORKDIR"}" ;;
    *) printf '%s\n' "$p" ;;
  esac
}

MODEL_DIR_HOST="$(host_path_for_container_path "$MODEL_DIR")"
LORA_DIR_HOST="$(host_path_for_container_path "$LORA_DIR")"
MERGED_MODEL_DIR_HOST="$(host_path_for_container_path "$MERGED_MODEL_DIR")"
QUANT_MODEL_DIR_HOST="$(host_path_for_container_path "$QUANT_MODEL_DIR")"
LMEVAL_WORKDIR_HOST="$(host_path_for_container_path "$LMEVAL_WORKDIR")"
STATE_DIR="${HOST_WORKDIR}/.glm51_resume_state"
RUNTIME_DIR="${STATE_DIR}/runtime"
ACTIVE_MODEL_ENV="${STATE_DIR}/active_model.env"

install_glm5_reference_scripts() {
  local scripts_dir="${GLM5_REFERENCE_SCRIPTS_DIR:?GLM5_REFERENCE_SCRIPTS_DIR empty}"
  mkdir -p "$scripts_dir"
  cat > "${scripts_dir}/merge_quant_serve_glm51_fp8.sh" <<'GLM5_REF_merge_quant_serve_glm51_fp8_sh'
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
GLM5_REF_merge_quant_serve_glm51_fp8_sh
  cat > "${scripts_dir}/download_tinker_adapter_archive.sh" <<'GLM5_REF_download_tinker_adapter_archive_sh'
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

echo "Archive preview, best effort first 80 members:"
if timeout 30s tar -tzf "$ARCHIVE_PATH" | sed -n '1,80p'; then
  :
else
  echo "Archive preview timed out or failed; continuing with full archive member scan."
fi

echo "Scanning adapter archive members: $ARCHIVE_PATH"
tar -tzf "$ARCHIVE_PATH" > "$TAR_LIST"
echo "Archive member scan complete: $(wc -l < "$TAR_LIST" | tr -d ' ') entries"
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
GLM5_REF_download_tinker_adapter_archive_sh
  cat > "${scripts_dir}/tinker_to_http_archive.py" <<'GLM5_REF_tinker_to_http_archive_py'
#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request


TERMINAL_SUCCESS = {"succeeded", "completed"}
TERMINAL_FAILURE = {"failed"}
URL_FIELDS = ("oss_http_url", "download_url", "oss_url")
DEFAULT_GPU_LEASE_API_KEY = ""


def request_json(method, url, api_key, payload=None):
    data = None
    headers = {"x-api-key": api_key}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Request failed for {url}: {exc}") from exc

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Non-JSON response from {url}: {body}") from exc


def get_nested(mapping, *path):
    current = mapping
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def extract_job_id(response):
    for path in (("job_id",), ("id",), ("job", "id"), ("data", "job_id")):
        value = get_nested(response, *path)
        if value:
            return str(value)
    raise RuntimeError(f"Could not find job_id in create response: {json.dumps(response, ensure_ascii=False)}")


def extract_result_url(result):
    if not isinstance(result, dict):
        return None
    for field in URL_FIELDS:
        value = result.get(field)
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            return value
    return None


def print_summary(status_response, final_url=None):
    result = status_response.get("result") if isinstance(status_response, dict) else {}
    if not isinstance(result, dict):
        result = {}

    rows = {
        "job_id": status_response.get("job_id") or status_response.get("id"),
        "status": status_response.get("status"),
        "stage": status_response.get("stage"),
        "oss_key": result.get("oss_key"),
        "archive_size_bytes": result.get("archive_size_bytes"),
        "expires_at": result.get("expires_at"),
        "download_url": final_url,
    }
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Convert a tinker:// model URL to a signed HTTP(S) archive URL via GPU Lease Manager async jobs."
    )
    parser.add_argument("model_url", nargs="?", default=os.environ.get("TINKER_URL"))
    parser.add_argument("--base-url", default=os.environ.get("GPU_LEASE_BASE_URL", "https://eval-service.macaron.im"))
    parser.add_argument("--api-key", default=os.environ.get("GPU_LEASE_API_KEY", DEFAULT_GPU_LEASE_API_KEY))
    parser.add_argument("--poll-interval", type=float, default=7.0)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    args = parser.parse_args()

    if not args.model_url:
        print("ERROR: provide a tinker:// URL argument or set TINKER_URL", file=sys.stderr)
        return 2
    if not args.model_url.startswith("tinker://"):
        print("ERROR: model URL must start with tinker://", file=sys.stderr)
        return 2
    if not args.api_key:
        print("ERROR: set GPU_LEASE_API_KEY", file=sys.stderr)
        return 2

    base_url = args.base_url.rstrip("/")
    create_url = f"{base_url}/api/transfer/jobs"
    create_response = request_json("POST", create_url, args.api_key, {"model_url": args.model_url})
    job_id = extract_job_id(create_response)
    print(f"job_id: {job_id}", flush=True)

    status_url = f"{base_url}/api/transfer/jobs/{job_id}"
    deadline = time.monotonic() + args.timeout_seconds
    last_response = None

    while time.monotonic() < deadline:
        last_response = request_json("GET", status_url, args.api_key)
        status = str(last_response.get("status", "")).lower()
        stage = last_response.get("stage")
        print(f"status: {status} stage: {stage}", flush=True)

        if status in TERMINAL_FAILURE:
            print_summary(last_response)
            print("ERROR: transfer job failed", file=sys.stderr)
            if "error" in last_response:
                print(json.dumps(last_response["error"], ensure_ascii=False, indent=2), file=sys.stderr)
            return 1

        if status in TERMINAL_SUCCESS:
            result = last_response.get("result")
            final_url = extract_result_url(result)
            if not final_url:
                print_summary(last_response)
                print("ERROR: succeeded but no HTTP(S) URL found in result. Do not use oss:// as a download URL.", file=sys.stderr)
                return 1
            print_summary(last_response, final_url)
            return 0

        time.sleep(args.poll_interval)

    if last_response is not None:
        print_summary(last_response)
    print(f"ERROR: timed out after {args.timeout_seconds} seconds waiting for {job_id}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
GLM5_REF_tinker_to_http_archive_py
  cat > "${scripts_dir}/merge_glm5_lora_into_base_local.py" <<'GLM5_REF_merge_glm5_lora_into_base_local_py'
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model-path", type=Path)
    p.add_argument("--lora-path", type=Path)
    p.add_argument("--output-path", type=Path)
    p.add_argument("--gpus", default=None)
    p.add_argument("--workers-per-gpu", type=int, default=1)
    p.add_argument("--limit-shards", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--no-expand-sparse-experts",
        action="store_true",
        help="Merge only sparse expert LoRA tensors that are present in adapter_model.safetensors.",
    )
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--worker-mode", action="store_true")
    p.add_argument("--worker-shards-json", type=Path)
    p.add_argument("--worker-result-json", type=Path)
    p.add_argument("--worker-gpu", default=None)
    p.add_argument("--plan-json", type=Path)
    return p.parse_args()


def map_lora_key_to_base_key(lora_key: str) -> str | None:
    if not lora_key.endswith(".lora_A.weight"):
        return None
    prefix = "base_model.model."
    if not lora_key.startswith(prefix):
        return None
    return (
        lora_key.removeprefix(prefix)
        .replace(".shared_expert.", ".shared_experts.")
        .replace(".lora_A.weight", ".weight")
    )


def tensor_shape(model_path: Path, weight_map: dict[str, str], key: str) -> tuple[int, ...]:
    from safetensors import safe_open

    with safe_open(str(model_path / weight_map[key]), framework="pt", device="cpu") as f:
        try:
            return tuple(f.get_slice(key).get_shape())
        except AttributeError:
            return tuple(f.get_tensor(key).shape)


def load_num_experts(base_model_path: Path) -> int | None:
    config_path = base_model_path / "config.json"
    if not config_path.is_file():
        return None
    config = json.loads(config_path.read_text())
    value = config.get("n_routed_experts") or config.get("num_experts")
    return int(value) if value is not None else None


def reconstruct_lm_head_b(
    *,
    lora_path: Path,
    rank: int,
    expected_rows: int,
    original_b: torch.Tensor,
) -> torch.Tensor | None:
    files: dict[int, Path] = {}
    for path in sorted(lora_path.glob("mp_rank_*_adapter.pt")):
        m = re.search(r"mp_rank_(\d+)_(\d+)_adapter\.pt$", path.name)
        if m:
            files.setdefault(int(m.group(1)), path)
    if not files:
        return None

    parts = []
    for tp_rank in sorted(files):
        data = torch.load(files[tp_rank], map_location="cpu")
        state = data.get("adapter_state_dict", data)
        tensor = state.get("output_layer.adapter.linear_out.weight")
        if tensor is None:
            return None
        if tensor.ndim != 2 or tensor.shape[1] < rank:
            raise RuntimeError(f"bad lm_head shard shape in {files[tp_rank]}: {tuple(tensor.shape)}")
        parts.append(tensor[:, :rank].contiguous())

    full = torch.cat(parts, dim=0)
    if tuple(full.shape) != (expected_rows, original_b.shape[1]):
        raise RuntimeError(
            f"bad reconstructed lm_head LoRA-B shape: got={tuple(full.shape)} "
            f"expected={(expected_rows, original_b.shape[1])}"
        )
    if not torch.equal(full[: original_b.shape[0]].to(original_b.dtype), original_b):
        max_diff = float((full[: original_b.shape[0]].float() - original_b.float()).abs().max())
        raise RuntimeError(f"rank0 lm_head LoRA-B shard mismatch; max_diff={max_diff}")
    return full.to(dtype=original_b.dtype)


def reconstruct_lm_head_pair(
    *,
    lora_path: Path,
    rank: int,
    expected_rows: int,
    original_a: torch.Tensor,
    original_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    files: dict[int, Path] = {}
    for path in sorted(lora_path.glob("mp_rank_*_adapter.pt")):
        m = re.search(r"mp_rank_(\d+)_(\d+)_adapter\.pt$", path.name)
        if m:
            files.setdefault(int(m.group(1)), path)
    if not files:
        return None

    a_parts = []
    b_parts = []
    for tp_rank in sorted(files):
        data = torch.load(files[tp_rank], map_location="cpu")
        state = data.get("adapter_state_dict", data)
        a_tensor = state.get("output_layer.adapter.linear_in.weight")
        b_tensor = state.get("output_layer.adapter.linear_out.weight")
        if a_tensor is None or b_tensor is None:
            return None
        if a_tensor.ndim != 2 or b_tensor.ndim != 2:
            raise RuntimeError(
                f"bad lm_head LoRA shard rank in {files[tp_rank]}: "
                f"A={None if a_tensor is None else tuple(a_tensor.shape)} "
                f"B={None if b_tensor is None else tuple(b_tensor.shape)}"
            )
        a_parts.append(a_tensor.contiguous())
        b_parts.append(b_tensor[:, :rank].contiguous())

    full_a = torch.cat(a_parts, dim=0)
    full_b = torch.cat(b_parts, dim=0)
    if tuple(full_a.shape) != (rank, original_a.shape[1]):
        raise RuntimeError(
            f"bad reconstructed lm_head LoRA-A shape: got={tuple(full_a.shape)} "
            f"expected={(rank, original_a.shape[1])}"
        )
    if tuple(full_b.shape) != (expected_rows, rank):
        raise RuntimeError(
            f"bad reconstructed lm_head LoRA-B shape: got={tuple(full_b.shape)} "
            f"expected={(expected_rows, rank)}"
        )
    if not torch.equal(full_a[: original_a.shape[0]].to(original_a.dtype), original_a):
        max_diff = float((full_a[: original_a.shape[0]].float() - original_a.float()).abs().max())
        raise RuntimeError(f"rank0 lm_head LoRA-A shard mismatch; max_diff={max_diff}")
    if not torch.equal(full_b[: original_b.shape[0]].to(original_b.dtype), original_b[:, :rank]):
        max_diff = float((full_b[: original_b.shape[0]].float() - original_b[:, :rank].float()).abs().max())
        raise RuntimeError(f"rank0 lm_head LoRA-B shard mismatch; max_diff={max_diff}")
    return full_a.to(dtype=original_a.dtype), full_b.to(dtype=original_b.dtype)


def expand_sparse_experts(
    base_to_pair: dict[str, tuple[str, str]],
    weight_map: dict[str, str],
    num_experts: int | None,
) -> int:
    if not num_experts:
        return 0
    pat = re.compile(r"^(model\.layers\.(\d+)\.mlp\.experts\.)(\d+)(\..+\.weight)$")
    by_layer: dict[int, set[int]] = defaultdict(set)
    for key in base_to_pair:
        m = pat.match(key)
        if m:
            by_layer[int(m.group(2))].add(int(m.group(3)))

    added = 0
    for layer, experts in sorted(by_layer.items()):
        reps = sorted(experts)
        if len(reps) < 2:
            continue
        stride = min(b - a for a, b in zip(reps, reps[1:]) if b > a)
        if stride <= 1 or reps[0] != 0 or any(rep % stride for rep in reps):
            continue
        items = []
        for key, pair in list(base_to_pair.items()):
            m = pat.match(key)
            if m and int(m.group(2)) == layer:
                items.append((key, pair, m))
        for _, pair, m in items:
            prefix, _, rep_raw, suffix = m.groups()
            rep = int(rep_raw)
            for expert_id in range(rep, min(rep + stride, num_experts)):
                expanded = f"{prefix}{expert_id}{suffix}"
                if expanded not in base_to_pair and expanded in weight_map:
                    base_to_pair[expanded] = pair
                    added += 1
    return added


def build_merge_plan(
    *,
    base_model_path: Path,
    lora_path: Path,
    expand_sparse_expert_lora: bool = True,
) -> tuple[dict[str, tuple[str, str]], dict[str, str], float, dict[str, torch.Tensor], dict[str, object]]:
    from safetensors import safe_open

    config = json.loads((lora_path / "adapter_config.json").read_text())
    rank = int(config["r"])
    scaling = float(config["lora_alpha"]) / rank
    weight_map = json.loads((base_model_path / "model.safetensors.index.json").read_text())["weight_map"]

    adapter_file = lora_path / "adapter_model.safetensors"
    with safe_open(str(adapter_file), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        key_set = set(keys)

    base_to_pair: dict[str, tuple[str, str]] = {}
    missing = []
    for key in keys:
        base_key = map_lora_key_to_base_key(key)
        if base_key is None:
            continue
        b_key = key.replace(".lora_A.weight", ".lora_B.weight")
        if b_key not in key_set:
            missing.append({"lora_A": key, "reason": "missing_lora_B"})
        elif base_key not in weight_map:
            missing.append({"lora_A": key, "reason": f"missing_base:{base_key}"})
        else:
            base_to_pair[base_key] = (key, b_key)
    if missing:
        raise RuntimeError(f"merge plan has {len(missing)} unmapped LoRA entries; first={missing[:5]}")
    if not base_to_pair:
        raise RuntimeError("merge plan found no mergeable LoRA pairs")

    overrides: dict[str, torch.Tensor] = {}
    lm_head_a_key = "base_model.model.lm_head.lora_A.weight"
    lm_head_b_key = "base_model.model.lm_head.lora_B.weight"
    if base_to_pair.get("lm_head.weight", (None, None))[1] == lm_head_b_key:
        base_rows = tensor_shape(base_model_path, weight_map, "lm_head.weight")[0]
        with safe_open(str(adapter_file), framework="pt", device="cpu") as f:
            original_a = f.get_tensor(lm_head_a_key)
            original_b = f.get_tensor(lm_head_b_key)
        if original_a.shape[0] != rank or original_b.shape[0] != base_rows:
            pair = reconstruct_lm_head_pair(
                lora_path=lora_path,
                rank=rank,
                expected_rows=base_rows,
                original_a=original_a,
                original_b=original_b,
            )
            if pair is not None:
                full_a, full_b = pair
                overrides[lm_head_a_key] = full_a
                overrides[lm_head_b_key] = full_b
        elif original_b.shape[0] != base_rows:
            full_b = reconstruct_lm_head_b(
                lora_path=lora_path,
                rank=rank,
                expected_rows=base_rows,
                original_b=original_b,
            )
            if full_b is not None:
                overrides[lm_head_b_key] = full_b

    expanded = 0
    if expand_sparse_expert_lora:
        expanded = expand_sparse_experts(base_to_pair, weight_map, load_num_experts(base_model_path))
    stats = {
        "lora_override_count": len(overrides),
        "lora_override_keys": sorted(overrides),
        "sparse_expert_expanded_param_count": expanded,
    }
    return base_to_pair, weight_map, scaling, overrides, stats


def copy_or_symlink_metadata(base_model_path: Path, out_dir: Path) -> None:
    for src in base_model_path.iterdir():
        if src.name.startswith("model-") and src.name.endswith(".safetensors"):
            continue
        dst = out_dir / src.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            dst.symlink_to(src)
        except OSError:
            if src.is_file():
                dst.write_bytes(src.read_bytes())


def parse_gpu_list(raw: str | None) -> list[str]:
    if raw:
        return [part.strip() for part in raw.split(",") if part.strip()]
    return [str(i) for i in range(torch.cuda.device_count())]


def run_worker(args: argparse.Namespace) -> None:
    if args.worker_shards_json is None or args.worker_result_json is None or args.plan_json is None:
        raise RuntimeError("worker mode requires shard/result/plan json paths")

    from safetensors import safe_open
    from safetensors.torch import save_file

    shards = json.loads(args.worker_shards_json.read_text())
    plan = json.loads(args.plan_json.read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_dir = Path(plan["base_model_path"])
    out_dir = Path(plan["output_path"])
    base_to_pair = {k: tuple(v) for k, v in plan["base_to_pair"].items()}
    scaling = float(plan["scaling"])
    override_file = plan.get("lora_override_file")

    modified_shards = 0
    modified_params = 0
    shard_summaries = []
    with safe_open(plan["lora_adapter_file"], framework="pt", device="cpu") as lora_file:
        override_ctx = safe_open(override_file, framework="pt", device="cpu") if override_file else None
        override_keys = set(override_ctx.keys()) if override_ctx is not None else set()

        def get_lora_tensor(key: str) -> torch.Tensor:
            if override_ctx is not None and key in override_keys:
                return override_ctx.get_tensor(key)
            return lora_file.get_tensor(key)

        for shard_name in shards:
            changed = 0
            updated: dict[str, torch.Tensor] = {}
            with safe_open(str(base_dir / shard_name), framework="pt", device=device) as shard_file:
                for key in shard_file.keys():
                    weight = shard_file.get_tensor(key)
                    pair = base_to_pair.get(key)
                    if pair is not None:
                        a_key, b_key = pair
                        lora_a = get_lora_tensor(a_key).to(device=device, dtype=torch.float32)
                        lora_b = get_lora_tensor(b_key).to(device=device, dtype=torch.float32)
                        delta = torch.matmul(lora_b, lora_a)
                        if delta.shape != weight.shape:
                            raise RuntimeError(
                                f"LoRA delta shape mismatch: base_key={key} base_shape={tuple(weight.shape)} "
                                f"delta_shape={tuple(delta.shape)} a_key={a_key} b_key={b_key}"
                            )
                        weight = weight.to(torch.float32).add_(delta, alpha=scaling).to(weight.dtype)
                        del lora_a, lora_b, delta
                        changed += 1
                    updated[key] = weight.cpu()
                    del weight
            out_path = out_dir / shard_name
            out_dir.mkdir(parents=True, exist_ok=True)
            tmp_out_path = out_dir / f".{shard_name}.tmp.{os.getpid()}"
            if tmp_out_path.exists() or tmp_out_path.is_symlink():
                tmp_out_path.unlink()
            if out_path.exists() or out_path.is_symlink():
                out_path.unlink()
            save_file(updated, str(tmp_out_path))
            tmp_out_path.replace(out_path)
            del updated
            if device == "cuda":
                torch.cuda.empty_cache()
            if changed:
                modified_shards += 1
                modified_params += changed
            shard_summaries.append({"shard": shard_name, "changed_params": changed})

    args.worker_result_json.write_text(json.dumps({
        "worker_gpu": args.worker_gpu,
        "worker_device": device,
        "assigned_shards": shards,
        "modified_shards": modified_shards,
        "modified_params": modified_params,
        "shards": shard_summaries,
    }, ensure_ascii=True, indent=2))


def write_overrides(overrides: dict[str, torch.Tensor], path: Path) -> str | None:
    if not overrides:
        return None
    from safetensors.torch import save_file

    save_file(overrides, str(path))
    return str(path)


def run_controller(args: argparse.Namespace) -> None:
    if args.base_model_path is None or args.lora_path is None or args.output_path is None:
        raise RuntimeError("controller mode requires --base-model-path, --lora-path, and --output-path")

    base_to_pair, weight_map, scaling, overrides, stats = build_merge_plan(
        base_model_path=args.base_model_path,
        lora_path=args.lora_path,
        expand_sparse_expert_lora=not args.no_expand_sparse_experts,
    )
    affected_by_shard: dict[str, list[str]] = defaultdict(list)
    for base_key in sorted(base_to_pair):
        affected_by_shard[weight_map[base_key]].append(base_key)

    shard_names = sorted(p.name for p in args.base_model_path.glob("model-*.safetensors"))
    if args.limit_shards is not None:
        if args.limit_shards <= 0:
            raise RuntimeError("--limit-shards must be positive")
        shard_names = shard_names[: args.limit_shards]
    if not shard_names:
        raise RuntimeError(f"no model shards under {args.base_model_path}")

    summary_base = {
        "base_model_path": str(args.base_model_path),
        "lora_path": str(args.lora_path),
        "output_path": str(args.output_path),
        "scaling": scaling,
        "mapped_param_count": len(base_to_pair),
        "affected_shard_count": len(affected_by_shard),
        "total_shard_count": len(shard_names),
        "plan_stats": stats,
    }
    if args.plan_only:
        print(json.dumps(summary_base, ensure_ascii=True, indent=2))
        return

    out_dir = args.output_path
    if out_dir.exists():
        if any(out_dir.iterdir()) and not args.force:
            raise RuntimeError(f"output path already exists and is non-empty: {out_dir}")
        if any(out_dir.iterdir()) and args.force:
            shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
    copy_or_symlink_metadata(args.base_model_path, out_dir)

    gpus = parse_gpu_list(args.gpus)
    if not gpus:
        raise RuntimeError("no GPUs available; pass --gpus or ensure CUDA is visible")
    worker_count = max(1, min(len(shard_names), len(gpus) * max(1, args.workers_per_gpu)))
    buckets = [[] for _ in range(worker_count)]
    for idx, shard_name in enumerate(shard_names):
        buckets[idx % worker_count].append(shard_name)

    with tempfile.TemporaryDirectory(prefix="glm5_merge_local_") as tmp_raw:
        tmp = Path(tmp_raw)
        override_file = write_overrides(overrides, tmp / "lora_overrides.safetensors")
        plan_json = tmp / "plan.json"
        plan_json.write_text(json.dumps({
            "base_model_path": str(args.base_model_path),
            "output_path": str(out_dir),
            "lora_adapter_file": str(args.lora_path / "adapter_model.safetensors"),
            "lora_override_file": override_file,
            "base_to_pair": base_to_pair,
            "scaling": scaling,
        }, ensure_ascii=True))

        procs = []
        result_paths = []
        for idx, bucket in enumerate(buckets):
            if not bucket:
                continue
            gpu = gpus[idx % len(gpus)]
            shards_json = tmp / f"shards_{idx}.json"
            result_json = tmp / f"result_{idx}.json"
            shards_json.write_text(json.dumps(bucket, ensure_ascii=True))
            result_paths.append(result_json)
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = gpu
            procs.append(subprocess.Popen([
                sys.executable,
                __file__,
                "--worker-mode",
                "--worker-shards-json",
                str(shards_json),
                "--worker-result-json",
                str(result_json),
                "--plan-json",
                str(plan_json),
                "--worker-gpu",
                gpu,
            ], env=env))

        for proc in procs:
            code = proc.wait()
            if code != 0:
                raise RuntimeError(f"merge worker exited with code {code}")
        worker_results = [json.loads(path.read_text()) for path in result_paths]

    changed_shards = {
        item["shard"]
        for result in worker_results
        for item in result["shards"]
        if item["changed_params"] > 0
    }
    untouched = sorted(set(shard_names) - changed_shards)
    for shard_name in untouched:
        dst = out_dir / shard_name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(args.base_model_path / shard_name)

    summary = dict(summary_base)
    summary.update({
        "changed_shards": len(changed_shards),
        "unchanged_shards": len(untouched),
        "worker_results": worker_results,
    })
    (out_dir / "merge_summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2))
    print(json.dumps(summary, ensure_ascii=True, indent=2))


def main() -> None:
    args = parse_args()
    if args.worker_mode:
        run_worker(args)
    else:
        run_controller(args)


if __name__ == "__main__":
    main()
GLM5_REF_merge_glm5_lora_into_base_local_py
  cat > "${scripts_dir}/quantize_glm5_finegrained_fp8_parallel.py" <<'GLM5_REF_quantize_glm5_finegrained_fp8_parallel_py'
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    FineGrainedFP8Config,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model-path")
    p.add_argument("--export-dir")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--gpus", default=None)
    p.add_argument("--workers-per-gpu", type=int, default=8)
    p.add_argument("--local-quant-root", default="/root/tmp/glm5_local_quant")
    p.add_argument(
        "--save-to-pfs",
        action="store_true",
        help="Quantize to local disk first, then move the completed checkpoint to --export-dir.",
    )
    p.add_argument("--worker-mode", action="store_true")
    p.add_argument("--worker-tasks-json", type=Path)
    p.add_argument("--worker-result-json", type=Path)
    p.add_argument("--worker-plan-json", type=Path)
    p.add_argument("--worker-gpu", default=None)
    return p.parse_args()


def resolve_config_source(base_model_path: str) -> str:
    config_path = Path(base_model_path) / "config.json"
    if config_path.is_file():
        return base_model_path

    summary_path = Path(base_model_path) / "merge_summary.json"
    if not summary_path.is_file():
        raise RuntimeError(
            f"missing readable config.json and merge_summary.json under {base_model_path}"
        )
    summary = json.loads(summary_path.read_text())
    source = summary.get("base_model_path")
    if not isinstance(source, str):
        raise RuntimeError(f"merge_summary.json missing base_model_path: {summary_path}")
    if not (Path(source) / "config.json").is_file():
        raise RuntimeError(f"merge_summary base_model_path has no config.json: {source}")
    return source


def materialize_merged_model_metadata(base_model_path: str, config_source: str) -> list[str]:
    copied: list[str] = []
    source_root = Path(config_source)
    target_root = Path(base_model_path)
    for filename in (
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ):
        source_path = source_root / filename
        if not source_path.is_file():
            continue
        target_path = target_root / filename
        if target_path.is_file():
            continue
        if target_path.is_symlink():
            target_path.unlink()
        target_path.write_bytes(source_path.read_bytes())
        copied.append(filename)
    return copied


def repair_missing_merged_shards(base_model_path: str, config_source: str) -> list[str]:
    source_root = Path(config_source)
    target_root = Path(base_model_path)
    index_path = target_root / "model.safetensors.index.json"
    if not index_path.is_file():
        return []
    summary_path = target_root / "merge_summary.json"
    if not summary_path.is_file():
        raise RuntimeError(
            f"missing merge_summary.json for merged checkpoint shard repair: {summary_path}"
        )

    shard_names = sorted(set(json.loads(index_path.read_text())["weight_map"].values()))
    summary = json.loads(summary_path.read_text())
    unchanged_shards = {
        shard_info["shard"]
        for worker in summary.get("worker_results", [])
        for shard_info in worker.get("shards", [])
        if shard_info.get("changed_params") == 0
    }
    repaired: list[str] = []
    for shard_name in shard_names:
        target_path = target_root / shard_name
        if target_path.exists():
            continue
        if shard_name not in unchanged_shards:
            raise RuntimeError(
                f"missing merged shard is not marked unchanged in merge_summary.json: {shard_name}"
            )
        source_path = source_root / shard_name
        if not source_path.exists():
            continue
        if target_path.is_symlink():
            target_path.unlink()
        target_path.symlink_to(source_path)
        repaired.append(shard_name)
    return repaired


def parse_gpu_list(raw: str | None) -> list[str]:
    if raw:
        return [part.strip() for part in raw.split(",") if part.strip()]
    return [str(i) for i in range(torch.cuda.device_count())]


def resolve_local_export_dir(requested_export_dir: Path, local_quant_root: str, save_to_pfs: bool) -> Path:
    local_root = Path(local_quant_root)
    local_dir = local_root / requested_export_dir.name
    if not save_to_pfs and requested_export_dir.is_absolute():
        try:
            requested_export_dir.relative_to(local_root)
            return requested_export_dir
        except ValueError:
            return local_dir
    return local_dir


def promote_local_export(local_export_dir: Path, final_export_dir: Path, save_to_pfs: bool) -> None:
    if not save_to_pfs or local_export_dir == final_export_dir:
        return
    if not (local_export_dir / "fp8_quant_meta.json").is_file():
        raise RuntimeError(f"refusing to promote incomplete quant output: {local_export_dir}")
    if final_export_dir.exists():
        if any(final_export_dir.iterdir()):
            raise RuntimeError(f"final export dir already exists and is non-empty: {final_export_dir}")
        final_export_dir.rmdir()
    final_export_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(local_export_dir), str(final_export_dir))
    meta_path = final_export_dir / "fp8_quant_meta.json"
    meta = json.loads(meta_path.read_text())
    meta["local_export_dir"] = str(local_export_dir)
    meta["export_dir"] = str(final_export_dir)
    meta["promoted_to_pfs"] = True
    meta_path.write_text(json.dumps(meta, ensure_ascii=True, indent=2))


def copy_tokenizer_artifacts(config_source: str, export_dir: Path) -> list[str]:
    source_root = Path(config_source)
    copied: list[str] = []
    for filename in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "spiece.model",
        "sentencepiece.bpe.model",
    ):
        source_path = source_root / filename
        if not source_path.is_file():
            continue
        target_path = export_dir / filename
        target_path.write_bytes(source_path.read_bytes())
        copied.append(filename)
    return copied


def build_fp32_skip_modules(config) -> list[str]:
    # FineGrainedFP8 block quantization requires 128x128 tiling. GLM-5's
    # kv_a_proj_with_mqa has width 576, and the indexer weights_proj stays bf16
    # by design in the upstream HF model. vLLM also fuses q_a + kv_a into one
    # fused_qkv_a_proj, so q_a must stay bf16 as well or vLLM rejects the fused
    # layer as partially quantized.
    layer_ids = range(int(config.num_hidden_layers))
    return [
        "lm_head",
        "model.embed_tokens",
        *[
            f"model.layers.{layer_idx}.input_layernorm"
            for layer_idx in layer_ids
        ],
        *[
            f"model.layers.{layer_idx}.post_attention_layernorm"
            for layer_idx in layer_ids
        ],
        *[
            f"model.layers.{layer_idx}.mlp.gate"
            for layer_idx in layer_ids
        ],
        *[
            f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias"
            for layer_idx in layer_ids
        ],
        *[
            f"model.layers.{layer_idx}.self_attn.indexer.k_norm"
            for layer_idx in layer_ids
        ],
        *[
            f"model.layers.{layer_idx}.self_attn.indexer.k_norm.bias"
            for layer_idx in layer_ids
        ],
        *[
            f"model.layers.{layer_idx}.self_attn.indexers_proj"
            for layer_idx in layer_ids
        ],
        *[
            f"model.layers.{layer_idx}.self_attn.indexer.weights_proj"
            for layer_idx in layer_ids
        ],
        *[
            f"model.layers.{layer_idx}.self_attn.kv_a_layernorm"
            for layer_idx in layer_ids
        ],
        *[
            f"model.layers.{layer_idx}.self_attn.q_a_layernorm"
            for layer_idx in layer_ids
        ],
        *[
            f"model.layers.{layer_idx}.self_attn.q_a_proj"
            for layer_idx in layer_ids
        ],
        *[
            f"model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa"
            for layer_idx in layer_ids
        ],
    ]


def normalize_glm5_config_for_fp8(config) -> None:
    # Transformers' FineGrainedFP8 MoE wrapper currently assumes a generic
    # `num_experts` attribute, while GLM-5 remote config exposes
    # `n_routed_experts` instead. Add the alias before module replacement so
    # the wrapper does not crash during model construction.
    if getattr(config, "model_type", None) == "glm_moe_dsa" and getattr(config, "num_experts", None) is None:
        routed = getattr(config, "n_routed_experts", None)
        if routed is not None:
            config.num_experts = int(routed)


def sanitize_generation_config_for_save(model) -> list[str]:
    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        return []

    fixed: list[str] = []
    if bool(getattr(generation_config, "do_sample", False)):
        return fixed

    sample_only_defaults = {
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 1.0,
        "min_p": None,
        "typical_p": 1.0,
        "epsilon_cutoff": 0.0,
        "eta_cutoff": 0.0,
    }
    for field_name, neutral_value in sample_only_defaults.items():
        current = getattr(generation_config, field_name, None)
        if current != neutral_value:
            setattr(generation_config, field_name, neutral_value)
            fixed.append(field_name)
    return fixed


def quantize_block_fp8_tensor(
    tensor: torch.Tensor,
    *,
    block_size: tuple[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = tensor.shape
    block_m, block_n = block_size
    padded_rows = math.ceil(rows / block_m) * block_m
    padded_cols = math.ceil(cols / block_n) * block_n
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    weight_fp32 = tensor.to(device=device, dtype=torch.float32, non_blocking=True)
    if padded_rows != rows or padded_cols != cols:
        padded = torch.zeros(
            (padded_rows, padded_cols),
            dtype=torch.float32,
            device=device,
        )
        padded[:rows, :cols] = weight_fp32
        weight_fp32 = padded
    reshaped = weight_fp32.reshape(padded_rows // block_m, block_m, padded_cols // block_n, block_n)
    max_abs = reshaped.abs().amax(dim=(1, 3))
    safe_max = torch.where(max_abs > 0, max_abs, torch.ones_like(max_abs))
    scales = fp8_max / safe_max
    scales = torch.where(max_abs > 0, scales, torch.ones_like(scales))
    quantized = torch.clamp(
        reshaped * scales.unsqueeze(1).unsqueeze(3),
        min=torch.finfo(torch.float8_e4m3fn).min,
        max=fp8_max,
    ).to(torch.float8_e4m3fn)
    quantized = quantized.reshape(padded_rows, padded_cols)[:rows, :cols].to("cpu")
    scale_inv = scales.reciprocal().to(dtype=torch.float32, device="cpu")
    del weight_fp32, reshaped, max_abs, safe_max, scales
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return quantized, scale_inv


def rewrite_moe_expert_shards(
    *,
    base_model_path: str,
    config,
    export_dir: Path,
    weight_block_size: tuple[int, int],
) -> list[str]:
    source_root = Path(base_model_path)
    source_index_path = source_root / "model.safetensors.index.json"
    if not source_index_path.is_file():
        raise RuntimeError(f"missing source safetensors index: {source_index_path}")
    export_index_path = export_dir / "model.safetensors.index.json"
    if not export_index_path.is_file():
        raise RuntimeError(f"missing export safetensors index: {export_index_path}")

    source_weight_map = json.loads(source_index_path.read_text())["weight_map"]
    index_payload = json.loads(export_index_path.read_text())
    weight_map = index_payload["weight_map"]
    mlp_layer_types = list(getattr(config, "mlp_layer_types", []))
    if not mlp_layer_types:
        raise RuntimeError("config.mlp_layer_types is empty")

    open_handles: dict[str, object] = {}
    fixed_shards: list[str] = []
    quant_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _load_tensor(name: str) -> torch.Tensor:
        shard_name = source_weight_map.get(name)
        if shard_name is None:
            raise KeyError(f"missing source tensor in merged checkpoint: {name}")
        handle = open_handles.get(shard_name)
        if handle is None:
            handle = safe_open(str(source_root / shard_name), framework="pt", device="cpu")
            open_handles[shard_name] = handle
        return handle.get_tensor(name)

    experts_per_chunk = 32
    num_experts = int(getattr(config, "n_routed_experts"))
    for layer_idx, layer_type in enumerate(mlp_layer_types):
        if layer_type != "sparse":
            continue
        for chunk_start in range(0, num_experts, experts_per_chunk):
            chunk_stop = min(chunk_start + experts_per_chunk, num_experts)
            shard_name = f"model-experts-fix-layer-{layer_idx:03d}-chunk-{chunk_start:03d}.safetensors"
            shard_tensors: dict[str, torch.Tensor] = {}
            for expert_idx in range(chunk_start, chunk_stop):
                expert_prefix = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"
                for proj_name in ("gate_proj", "up_proj", "down_proj"):
                    weight_key = f"{expert_prefix}.{proj_name}.weight"
                    quantized, scale_inv = quantize_block_fp8_tensor(
                        _load_tensor(weight_key),
                        block_size=weight_block_size,
                        device=quant_device,
                    )
                    shard_tensors[weight_key] = quantized.contiguous()
                    shard_tensors[f"{expert_prefix}.{proj_name}.weight_scale_inv"] = scale_inv.contiguous()
                    weight_map[weight_key] = shard_name
                    weight_map[f"{expert_prefix}.{proj_name}.weight_scale_inv"] = shard_name
            save_file(shard_tensors, str(export_dir / shard_name), metadata={"format": "pt"})
            fixed_shards.append(shard_name)
            del shard_tensors
            if quant_device.type == "cuda":
                torch.cuda.empty_cache()

    total_size = sum(path.stat().st_size for path in export_dir.glob("*.safetensors"))
    index_payload.setdefault("metadata", {})["total_size"] = total_size
    export_index_path.write_text(json.dumps(index_payload, ensure_ascii=True, indent=2))
    print(
        json.dumps(
            {
                "phase": "moe_fix_done",
                "fixed_shard_count": len(fixed_shards),
                "fixed_sparse_layer_count": sum(1 for x in mlp_layer_types if x == "sparse"),
                "total_safetensors_bytes": total_size,
            },
            ensure_ascii=True,
        ),
        flush=True,
    )
    return fixed_shards


def build_moe_fix_tasks(config) -> list[dict[str, int]]:
    mlp_layer_types = list(getattr(config, "mlp_layer_types", []))
    if not mlp_layer_types:
        raise RuntimeError("config.mlp_layer_types is empty")
    num_experts = int(getattr(config, "n_routed_experts"))
    tasks: list[dict[str, int]] = []
    experts_per_chunk = 32
    for layer_idx, layer_type in enumerate(mlp_layer_types):
        if layer_type != "sparse":
            continue
        for chunk_start in range(0, num_experts, experts_per_chunk):
            tasks.append(
                {
                    "layer_idx": layer_idx,
                    "chunk_start": chunk_start,
                    "chunk_stop": min(chunk_start + experts_per_chunk, num_experts),
                }
            )
    return tasks


def run_moe_fix_worker(args: argparse.Namespace) -> None:
    if args.worker_tasks_json is None or args.worker_result_json is None or args.worker_plan_json is None:
        raise RuntimeError("worker mode requires --worker-tasks-json, --worker-result-json, and --worker-plan-json")

    tasks = json.loads(args.worker_tasks_json.read_text())
    plan = json.loads(args.worker_plan_json.read_text())
    if not tasks:
        args.worker_result_json.write_text(
            json.dumps(
                {
                    "worker_gpu": args.worker_gpu,
                    "worker_device": "none",
                    "task_count": 0,
                    "fixed_shards": [],
                },
                ensure_ascii=True,
                indent=2,
            )
        )
        return

    source_root = Path(plan["base_model_path"])
    export_dir = Path(plan["export_dir"])
    source_weight_map = json.loads((source_root / "model.safetensors.index.json").read_text())["weight_map"]
    weight_block_size = tuple(plan["weight_block_size"])
    quant_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    open_handles: dict[str, object] = {}
    fixed_shards: list[str] = []

    def _load_tensor(name: str) -> torch.Tensor:
        shard_name = source_weight_map.get(name)
        if shard_name is None:
            raise KeyError(f"missing source tensor in merged checkpoint: {name}")
        handle = open_handles.get(shard_name)
        if handle is None:
            handle = safe_open(str(source_root / shard_name), framework="pt", device="cpu")
            open_handles[shard_name] = handle
        return handle.get_tensor(name)

    try:
        for task in tasks:
            layer_idx = int(task["layer_idx"])
            chunk_start = int(task["chunk_start"])
            chunk_stop = int(task["chunk_stop"])
            shard_name = f"model-experts-fix-layer-{layer_idx:03d}-chunk-{chunk_start:03d}.safetensors"
            shard_tensors: dict[str, torch.Tensor] = {}
            for expert_idx in range(chunk_start, chunk_stop):
                expert_prefix = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"
                for proj_name in ("gate_proj", "up_proj", "down_proj"):
                    weight_key = f"{expert_prefix}.{proj_name}.weight"
                    quantized, scale_inv = quantize_block_fp8_tensor(
                        _load_tensor(weight_key),
                        block_size=weight_block_size,
                        device=quant_device,
                    )
                    shard_tensors[weight_key] = quantized.contiguous()
                    shard_tensors[f"{expert_prefix}.{proj_name}.weight_scale_inv"] = scale_inv.contiguous()
            tmp_path = export_dir / f".{shard_name}.tmp.{os.getpid()}"
            out_path = export_dir / shard_name
            if tmp_path.exists():
                tmp_path.unlink()
            if out_path.exists():
                out_path.unlink()
            save_file(shard_tensors, str(tmp_path), metadata={"format": "pt"})
            tmp_path.replace(out_path)
            fixed_shards.append(shard_name)
            del shard_tensors
            if quant_device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        for handle in open_handles.values():
            handle.__exit__(None, None, None)

    args.worker_result_json.write_text(
        json.dumps(
            {
                "worker_gpu": args.worker_gpu,
                "worker_device": str(quant_device),
                "task_count": len(tasks),
                "fixed_shards": fixed_shards,
            },
            ensure_ascii=True,
            indent=2,
        )
    )


def rewrite_moe_expert_shards_parallel(
    *,
    base_model_path: str,
    config,
    export_dir: Path,
    weight_block_size: tuple[int, int],
    gpus: list[str],
    workers_per_gpu: int,
) -> dict[str, object]:
    source_root = Path(base_model_path)
    source_index_path = source_root / "model.safetensors.index.json"
    if not source_index_path.is_file():
        raise RuntimeError(f"missing source safetensors index: {source_index_path}")
    export_index_path = export_dir / "model.safetensors.index.json"
    if not export_index_path.is_file():
        raise RuntimeError(f"missing export safetensors index: {export_index_path}")

    index_payload = json.loads(export_index_path.read_text())
    weight_map = index_payload["weight_map"]
    tasks = build_moe_fix_tasks(config)
    if not tasks:
        raise RuntimeError("no sparse MoE fix tasks were generated")
    if not gpus:
        raise RuntimeError("no GPUs available; pass --gpus or ensure CUDA is visible")
    worker_count = max(1, min(len(tasks), len(gpus) * max(1, workers_per_gpu)))
    buckets = [[] for _ in range(worker_count)]
    for idx, task in enumerate(tasks):
        buckets[idx % worker_count].append(task)

    with tempfile.TemporaryDirectory(prefix="glm5_quant_moe_") as tmp_raw:
        tmp = Path(tmp_raw)
        plan_json = tmp / "plan.json"
        plan_json.write_text(
            json.dumps(
                {
                    "base_model_path": base_model_path,
                    "export_dir": str(export_dir),
                    "weight_block_size": list(weight_block_size),
                },
                ensure_ascii=True,
            )
        )
        procs = []
        result_paths = []
        for idx, bucket in enumerate(buckets):
            if not bucket:
                continue
            gpu = gpus[idx % len(gpus)]
            tasks_json = tmp / f"tasks_{idx}.json"
            result_json = tmp / f"result_{idx}.json"
            tasks_json.write_text(json.dumps(bucket, ensure_ascii=True))
            result_paths.append(result_json)
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = gpu
            procs.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        __file__,
                        "--worker-mode",
                        "--worker-tasks-json",
                        str(tasks_json),
                        "--worker-result-json",
                        str(result_json),
                        "--worker-plan-json",
                        str(plan_json),
                        "--worker-gpu",
                        gpu,
                    ],
                    env=env,
                )
            )

        for proc in procs:
            code = proc.wait()
            if code != 0:
                raise RuntimeError(f"MoE quant worker exited with code {code}")
        worker_results = [json.loads(path.read_text()) for path in result_paths]

    fixed_shards = [shard for result in worker_results for shard in result["fixed_shards"]]
    for task in tasks:
        layer_idx = int(task["layer_idx"])
        chunk_start = int(task["chunk_start"])
        chunk_stop = int(task["chunk_stop"])
        shard_name = f"model-experts-fix-layer-{layer_idx:03d}-chunk-{chunk_start:03d}.safetensors"
        for expert_idx in range(chunk_start, chunk_stop):
            expert_prefix = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"
            for proj_name in ("gate_proj", "up_proj", "down_proj"):
                weight_key = f"{expert_prefix}.{proj_name}.weight"
                weight_map[weight_key] = shard_name
                weight_map[f"{expert_prefix}.{proj_name}.weight_scale_inv"] = shard_name

    total_size = sum(path.stat().st_size for path in export_dir.glob("*.safetensors"))
    index_payload.setdefault("metadata", {})["total_size"] = total_size
    export_index_path.write_text(json.dumps(index_payload, ensure_ascii=True, indent=2))
    result = {
        "phase": "moe_fix_done",
        "fixed_shard_count": len(fixed_shards),
        "fixed_sparse_layer_count": len({task["layer_idx"] for task in tasks}),
        "worker_count": len(worker_results),
        "workers_per_gpu": workers_per_gpu,
        "gpus": gpus,
        "worker_results": worker_results,
        "total_safetensors_bytes": total_size,
    }
    print(json.dumps(result, ensure_ascii=True), flush=True)
    return result


def scrub_stale_moe_keys(export_dir: Path, config) -> dict[str, object]:
    index_path = export_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise RuntimeError(f"missing export safetensors index: {index_path}")

    index_payload = json.loads(index_path.read_text())
    weight_map = index_payload["weight_map"]
    n_routed_experts = int(getattr(config, "n_routed_experts"))
    expert_key_re = re.compile(r"\.mlp\.experts\.(\d+)\.")

    def _is_overrange_expert_key(key: str) -> bool:
        match = expert_key_re.search(key)
        return match is not None and int(match.group(1)) >= n_routed_experts

    overrange_index_keys = [key for key in weight_map if _is_overrange_expert_key(key)]
    for key in overrange_index_keys:
        del weight_map[key]

    fix_keys = {
        key
        for key, shard_name in weight_map.items()
        if shard_name.startswith("model-experts-fix-layer-")
    }
    if not fix_keys:
        raise RuntimeError("no repaired model-experts-fix shards found in export index")

    scrubbed: list[dict[str, object]] = []
    for shard_path in sorted(export_dir.glob("model-*-of-*.safetensors")):
        with safe_open(str(shard_path), framework="pt", device="cpu") as reader:
            file_keys = list(reader.keys())
            remove_keys = [
                key
                for key in file_keys
                if key in fix_keys or _is_overrange_expert_key(key)
            ]
            if not remove_keys:
                continue
            keep_tensors = {
                key: reader.get_tensor(key)
                for key in file_keys
                if key not in remove_keys
            }

        tmp_path = shard_path.with_suffix(shard_path.suffix + ".tmp")
        backup_path = shard_path.with_suffix(shard_path.suffix + ".pre_moe_scrub")
        if tmp_path.exists():
            tmp_path.unlink()
        if backup_path.exists():
            backup_path.unlink()
        save_file(keep_tensors, str(tmp_path), metadata={"format": "pt"})
        shard_path.rename(backup_path)
        tmp_path.rename(shard_path)
        backup_path.unlink()
        scrubbed.append(
            {
                "shard": shard_path.name,
                "removed_key_count": len(remove_keys),
                "removed_overrange_count": sum(
                    1 for key in remove_keys if _is_overrange_expert_key(key)
                ),
                "removed_redirected_count": sum(1 for key in remove_keys if key in fix_keys),
            }
        )

    total_size = sum(path.stat().st_size for path in export_dir.glob("*.safetensors"))
    index_payload.setdefault("metadata", {})["total_size"] = total_size
    index_payload["metadata"]["scrubbed_overrange_expert_key_count"] = len(overrange_index_keys)
    index_payload["metadata"]["scrubbed_n_routed_experts"] = n_routed_experts
    index_path.write_text(json.dumps(index_payload, ensure_ascii=True, indent=2))

    result = {
        "phase": "moe_stale_key_scrub_done",
        "scrubbed_shard_count": len(scrubbed),
        "scrubbed_shards": scrubbed,
        "removed_index_overrange_key_count": len(overrange_index_keys),
        "n_routed_experts": n_routed_experts,
        "total_safetensors_bytes": total_size,
    }
    print(json.dumps(result, ensure_ascii=True), flush=True)
    return result


def add_mtp_layer_from_source(
    *,
    base_model_path: str,
    export_dir: Path,
    weight_block_size: tuple[int, int],
) -> dict[str, object]:
    source_root = Path(base_model_path)
    source_index_path = source_root / "model.safetensors.index.json"
    export_index_path = export_dir / "model.safetensors.index.json"
    if not source_index_path.is_file() or not export_index_path.is_file():
        raise RuntimeError("missing source/export safetensors index for MTP repair")

    source_weight_map = json.loads(source_index_path.read_text())["weight_map"]
    layer78_keys = sorted(key for key in source_weight_map if key.startswith("model.layers.78."))
    if not layer78_keys:
        return {"phase": "mtp_layer_export_skipped", "reason": "source_has_no_layer_78"}

    index_payload = json.loads(export_index_path.read_text())
    weight_map = index_payload["weight_map"]
    existing_layer78 = [key for key in weight_map if key.startswith("model.layers.78.")]
    if existing_layer78:
        return {
            "phase": "mtp_layer_export_skipped",
            "reason": "export_already_has_layer_78",
            "existing_key_count": len(existing_layer78),
        }

    # Match the official GLM-5.1-FP8 MTP layout: norms and gates stay bf16/fp32,
    # while all 2-D projection matrices are block-fp8 with *_weight_scale_inv.
    keep_bf16_or_fp32_suffixes = (
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        ".eh_proj.weight",
        ".enorm.weight",
        ".hnorm.weight",
        ".shared_head.norm.weight",
        ".mlp.gate.weight",
        ".mlp.gate.e_score_correction_bias",
        ".self_attn.indexer.k_norm.weight",
        ".self_attn.indexer.k_norm.bias",
        ".self_attn.indexer.weights_proj.weight",
        ".self_attn.kv_a_layernorm.weight",
        ".self_attn.q_a_layernorm.weight",
    )
    expert_key_re = re.compile(r"\.mlp\.experts\.(\d+)\.")
    open_handles: dict[str, object] = {}
    quant_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    shard_tensors: dict[str, torch.Tensor] = {}
    quantized_count = 0
    copied_count = 0

    def _load_tensor(name: str) -> torch.Tensor:
        shard_name = source_weight_map[name]
        handle = open_handles.get(shard_name)
        if handle is None:
            handle = safe_open(str(source_root / shard_name), framework="pt", device="cpu")
            open_handles[shard_name] = handle
        return handle.get_tensor(name)

    for key in layer78_keys:
        if key.endswith(keep_bf16_or_fp32_suffixes):
            shard_tensors[key] = _load_tensor(key).contiguous()
            copied_count += 1
            continue
        tensor = _load_tensor(key)
        if tensor.ndim != 2:
            shard_tensors[key] = tensor.contiguous()
            copied_count += 1
            continue
        quantized, scale_inv = quantize_block_fp8_tensor(
            tensor,
            block_size=weight_block_size,
            device=quant_device,
        )
        shard_tensors[key] = quantized.contiguous()
        shard_tensors[f"{key}_scale_inv"] = scale_inv.contiguous()
        quantized_count += 1

    for handle in open_handles.values():
        handle.__exit__(None, None, None)

    shard_name = "model-mtp-layer-078.safetensors"
    save_file(shard_tensors, str(export_dir / shard_name), metadata={"format": "pt"})
    for key in shard_tensors:
        weight_map[key] = shard_name

    config_path = export_dir / "config.json"
    if config_path.is_file():
        config_payload = json.loads(config_path.read_text())
        quant_config = config_payload.setdefault("quantization_config", {})
        modules_to_not_convert = quant_config.setdefault("modules_to_not_convert", [])
        for module_name in (
            "model.layers.78.eh_proj",
            "model.layers.78.enorm",
            "model.layers.78.hnorm",
            "model.layers.78.input_layernorm",
            "model.layers.78.mlp.gate",
            "model.layers.78.mlp.gate.e_score_correction_bias",
            "model.layers.78.post_attention_layernorm",
            "model.layers.78.self_attn.indexer.k_norm",
            "model.layers.78.self_attn.indexer.k_norm.bias",
            "model.layers.78.self_attn.indexers_proj",
            "model.layers.78.self_attn.kv_a_layernorm",
            "model.layers.78.self_attn.q_a_layernorm",
            "model.layers.78.shared_head.norm",
        ):
            if module_name not in modules_to_not_convert:
                modules_to_not_convert.append(module_name)
        for module_name in (
            "model.layers.78.self_attn.kv_a_proj_with_mqa",
            "model.layers.78.self_attn.indexer.weights_proj",
        ):
            if module_name in modules_to_not_convert:
                modules_to_not_convert.remove(module_name)
        config_path.write_text(json.dumps(config_payload, ensure_ascii=True, indent=2))

    total_size = sum(path.stat().st_size for path in export_dir.glob("*.safetensors"))
    index_payload.setdefault("metadata", {})["total_size"] = total_size
    index_payload["metadata"]["mtp_layer_78_exported"] = True
    export_index_path.write_text(json.dumps(index_payload, ensure_ascii=True, indent=2))
    result = {
        "phase": "mtp_layer_export_done",
        "source_layer78_key_count": len(layer78_keys),
        "exported_key_count": len(shard_tensors),
        "quantized_weight_count": quantized_count,
        "copied_weight_count": copied_count,
        "shard": shard_name,
        "total_safetensors_bytes": total_size,
    }
    print(json.dumps(result, ensure_ascii=True), flush=True)
    return result


def module_name_from_tensor_key(key: str) -> str:
    for suffix in (
        ".weight_scale_inv",
        ".weight",
        ".bias",
        ".e_score_correction_bias",
    ):
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return key


def export_shardwise_fp8(
    *,
    base_model_path: str,
    config_source: str,
    config,
    export_dir: Path,
    modules_to_not_convert: list[str],
    weight_block_size: tuple[int, int],
) -> dict[str, object]:
    source_root = Path(base_model_path)
    source_index_path = source_root / "model.safetensors.index.json"
    if not source_index_path.is_file():
        raise RuntimeError(f"missing source safetensors index: {source_index_path}")

    config_payload = config.to_dict()
    config_payload["quantization_config"] = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": list(weight_block_size),
        "modules_to_not_convert": modules_to_not_convert,
    }
    (export_dir / "config.json").write_text(json.dumps(config_payload, ensure_ascii=True, indent=2))

    generation_config_path = Path(config_source) / "generation_config.json"
    if generation_config_path.is_file():
        (export_dir / "generation_config.json").write_bytes(generation_config_path.read_bytes())
    copied_tokenizer_artifacts = copy_tokenizer_artifacts(config_source, export_dir)

    source_weight_map = json.loads(source_index_path.read_text())["weight_map"]
    shard_names = sorted(set(source_weight_map.values()))
    n_routed_experts = int(getattr(config, "n_routed_experts"))
    expert_key_re = re.compile(r"\.mlp\.experts\.(\d+)\.")
    skip_modules = set(modules_to_not_convert)
    quant_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weight_map: dict[str, str] = {}
    quantized_weight_count = 0
    copied_tensor_count = 0
    skipped_expert_tensor_count = 0
    skipped_layer78_tensor_count = 0

    def should_skip_main_key(key: str) -> bool:
        if key.startswith("model.layers.78."):
            return True
        match = expert_key_re.search(key)
        return match is not None

    def should_quantize(key: str, tensor: torch.Tensor) -> bool:
        if not key.endswith(".weight") or tensor.ndim != 2:
            return False
        return module_name_from_tensor_key(key) not in skip_modules

    for shard_idx, shard_name in enumerate(shard_names, start=1):
        shard_tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(source_root / shard_name), framework="pt", device="cpu") as reader:
            for key in reader.keys():
                if should_skip_main_key(key):
                    if key.startswith("model.layers.78."):
                        skipped_layer78_tensor_count += 1
                    else:
                        skipped_expert_tensor_count += 1
                    continue
                tensor = reader.get_tensor(key)
                if should_quantize(key, tensor):
                    quantized, scale_inv = quantize_block_fp8_tensor(
                        tensor,
                        block_size=weight_block_size,
                        device=quant_device,
                    )
                    shard_tensors[key] = quantized.contiguous()
                    scale_key = f"{key}_scale_inv"
                    shard_tensors[scale_key] = scale_inv.contiguous()
                    weight_map[key] = shard_name
                    weight_map[scale_key] = shard_name
                    quantized_weight_count += 1
                else:
                    shard_tensors[key] = tensor.contiguous()
                    weight_map[key] = shard_name
                    copied_tensor_count += 1
                del tensor
        save_file(shard_tensors, str(export_dir / shard_name), metadata={"format": "pt"})
        del shard_tensors
        if quant_device.type == "cuda":
            torch.cuda.empty_cache()
        print(
            json.dumps(
                {
                    "phase": "shardwise_export_progress",
                    "shard": shard_name,
                    "shard_index": shard_idx,
                    "shard_count": len(shard_names),
                },
                ensure_ascii=True,
            ),
            flush=True,
        )

    index_payload = {
        "metadata": {
            "total_size": sum(path.stat().st_size for path in export_dir.glob("*.safetensors")),
        },
        "weight_map": weight_map,
    }
    (export_dir / "model.safetensors.index.json").write_text(
        json.dumps(index_payload, ensure_ascii=True, indent=2)
    )
    result = {
        "phase": "shardwise_export_done",
        "source_shard_count": len(shard_names),
        "quantized_weight_count": quantized_weight_count,
        "copied_tensor_count": copied_tensor_count,
        "skipped_expert_tensor_count": skipped_expert_tensor_count,
        "skipped_layer78_tensor_count": skipped_layer78_tensor_count,
        "copied_tokenizer_artifacts": copied_tokenizer_artifacts,
        "n_routed_experts": n_routed_experts,
    }
    print(json.dumps(result, ensure_ascii=True), flush=True)
    return result


def main() -> None:
    args = parse_args()
    if args.worker_mode:
        run_moe_fix_worker(args)
        return

    if not args.base_model_path or not args.export_dir:
        raise RuntimeError("controller mode requires --base-model-path and --export-dir")

    requested_export_dir = Path(args.export_dir)
    export_dir = resolve_local_export_dir(
        requested_export_dir=requested_export_dir,
        local_quant_root=args.local_quant_root,
        save_to_pfs=args.save_to_pfs,
    )
    if export_dir.exists() and any(export_dir.iterdir()):
        raise RuntimeError(f"export_dir already exists and is non-empty: {export_dir}")
    export_dir.mkdir(parents=True, exist_ok=True)

    cuda_count = torch.cuda.device_count()
    if cuda_count < 1:
        raise RuntimeError("FP8 quantization requires at least one visible CUDA device")
    max_memory_by_gpu = os.environ.get("GLM5_FP8_MAX_MEMORY_BY_GPU")
    if max_memory_by_gpu:
        max_memory = {i: "130GiB" for i in range(cuda_count)}
        for part in max_memory_by_gpu.split(","):
            gpu_id, value = part.split(":", 1)
            max_memory[int(gpu_id)] = value
    else:
        max_memory_gib = int(os.environ.get("GLM5_FP8_MAX_MEMORY_GIB", "130"))
        max_memory = {i: f"{max_memory_gib}GiB" for i in range(cuda_count)}

    config_source = resolve_config_source(args.base_model_path)
    print(
        json.dumps(
            {"phase": "config_source", "config_source": config_source},
            ensure_ascii=True,
        ),
        flush=True,
    )
    copied_metadata = materialize_merged_model_metadata(args.base_model_path, config_source)
    if copied_metadata:
        print(
            json.dumps(
                {"phase": "metadata_repaired", "copied_files": copied_metadata},
                ensure_ascii=True,
            ),
            flush=True,
        )
    repaired_shards = repair_missing_merged_shards(args.base_model_path, config_source)
    if repaired_shards:
        print(
            json.dumps(
                {
                    "phase": "shards_repaired",
                    "repaired_shards_count": len(repaired_shards),
                    "repaired_shards_sample": repaired_shards[:8],
                },
                ensure_ascii=True,
            ),
            flush=True,
        )
    config = AutoConfig.from_pretrained(
        config_source,
        trust_remote_code=args.trust_remote_code,
    )
    normalize_glm5_config_for_fp8(config)
    modules_to_not_convert = build_fp32_skip_modules(config)
    gpus = parse_gpu_list(args.gpus)

    quant_cfg = FineGrainedFP8Config(modules_to_not_convert=modules_to_not_convert)
    if os.environ.get("GLM5_FP8_SHARDWISE", "1") == "1":
        shardwise_result = export_shardwise_fp8(
            base_model_path=args.base_model_path,
            config_source=config_source,
            config=config,
            export_dir=export_dir,
            modules_to_not_convert=modules_to_not_convert,
            weight_block_size=tuple(quant_cfg.weight_block_size),
        )
        moe_fix_result = rewrite_moe_expert_shards_parallel(
            base_model_path=args.base_model_path,
            config=config,
            export_dir=export_dir,
            weight_block_size=tuple(quant_cfg.weight_block_size),
            gpus=gpus,
            workers_per_gpu=args.workers_per_gpu,
        )
        scrub_result = scrub_stale_moe_keys(export_dir, config)
        mtp_result = add_mtp_layer_from_source(
            base_model_path=args.base_model_path,
            export_dir=export_dir,
            weight_block_size=tuple(quant_cfg.weight_block_size),
        )
        meta = {
            "base_model_path": args.base_model_path,
            "export_dir": str(export_dir),
            "cuda_count": cuda_count,
            "quantization_method": "fp8",
            "is_quantized": True,
            "quantizer": "FineGrainedFP8Config",
            "export_mode": "shardwise",
            "requested_export_dir": str(requested_export_dir),
            "save_to_pfs": bool(args.save_to_pfs),
            "gpus": gpus,
            "workers_per_gpu": args.workers_per_gpu,
            "modules_to_not_convert_count": len(modules_to_not_convert),
            "modules_to_not_convert": modules_to_not_convert,
            "shardwise_export": shardwise_result,
            "moe_fix": moe_fix_result,
            "moe_fix_shard_count": int(moe_fix_result["fixed_shard_count"]),
            "moe_stale_key_scrub": scrub_result,
            "mtp_layer_export": mtp_result,
        }
        (export_dir / "fp8_quant_meta.json").write_text(json.dumps(meta, ensure_ascii=True, indent=2))
        print(json.dumps({"phase": "done", **meta}, ensure_ascii=True), flush=True)
        promote_local_export(export_dir, requested_export_dir, args.save_to_pfs)
        return

    device_map = os.environ.get("GLM5_FP8_DEVICE_MAP", "auto")
    print(
        json.dumps(
            {
                "phase": "load_start",
                "base_model_path": args.base_model_path,
                "export_dir": str(export_dir),
                "cuda_count": cuda_count,
                "max_memory": max_memory,
                "quant_method": "fp8",
                "quantizer": "FineGrainedFP8Config",
                "device_map": device_map,
                "modules_to_not_convert_count": len(modules_to_not_convert),
                "modules_to_not_convert_sample": modules_to_not_convert[:4],
            },
            ensure_ascii=True,
        ),
        flush=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        config=config,
        quantization_config=quant_cfg,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    print(
        json.dumps(
            {
                "phase": "load_done",
                "quantization_method": str(getattr(model, "quantization_method", None)),
                "is_quantized": bool(getattr(model, "is_quantized", False)),
            },
            ensure_ascii=True,
        ),
        flush=True,
    )
    quantization_method = str(getattr(model, "quantization_method", None))
    is_quantized = bool(getattr(model, "is_quantized", False))

    fixed_generation_fields = sanitize_generation_config_for_save(model)
    if fixed_generation_fields:
        print(
            json.dumps(
                {
                    "phase": "generation_config_sanitized",
                    "fields": fixed_generation_fields,
                },
                ensure_ascii=True,
            ),
            flush=True,
        )

    model.save_pretrained(export_dir, safe_serialization=True, max_shard_size="10GB")
    copied_tokenizer_artifacts = copy_tokenizer_artifacts(config_source, export_dir)
    if copied_tokenizer_artifacts:
        print(
            json.dumps(
                {
                    "phase": "tokenizer_artifacts_copied",
                    "files": copied_tokenizer_artifacts,
                },
                ensure_ascii=True,
            ),
            flush=True,
        )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    moe_fix_result = rewrite_moe_expert_shards_parallel(
        base_model_path=args.base_model_path,
        config=config,
        export_dir=export_dir,
        weight_block_size=tuple(quant_cfg.weight_block_size),
        gpus=gpus,
        workers_per_gpu=args.workers_per_gpu,
    )
    scrub_result = scrub_stale_moe_keys(export_dir, config)
    mtp_result = add_mtp_layer_from_source(
        base_model_path=args.base_model_path,
        export_dir=export_dir,
        weight_block_size=tuple(quant_cfg.weight_block_size),
    )

    meta = {
        "base_model_path": args.base_model_path,
        "export_dir": str(export_dir),
        "cuda_count": cuda_count,
        "quantization_method": quantization_method,
        "is_quantized": is_quantized,
        "quantizer": "FineGrainedFP8Config",
        "requested_export_dir": str(requested_export_dir),
        "save_to_pfs": bool(args.save_to_pfs),
        "gpus": gpus,
        "workers_per_gpu": args.workers_per_gpu,
        "modules_to_not_convert_count": len(modules_to_not_convert),
        "modules_to_not_convert": modules_to_not_convert,
        "moe_fix": moe_fix_result,
        "moe_fix_shard_count": int(moe_fix_result["fixed_shard_count"]),
        "moe_stale_key_scrub": scrub_result,
        "mtp_layer_export": mtp_result,
    }
    (export_dir / "fp8_quant_meta.json").write_text(json.dumps(meta, ensure_ascii=True, indent=2))
    print(json.dumps({"phase": "done", **meta}, ensure_ascii=True), flush=True)
    promote_local_export(export_dir, requested_export_dir, args.save_to_pfs)


if __name__ == "__main__":
    main()
GLM5_REF_quantize_glm5_finegrained_fp8_parallel_py
  cat > "${scripts_dir}/quantize_glm5_finegrained_fp8.py" <<'GLM5_REF_quantize_glm5_finegrained_fp8_py'
from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    FineGrainedFP8Config,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model-path", required=True)
    p.add_argument("--export-dir", required=True)
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def resolve_config_source(base_model_path: str) -> str:
    config_path = Path(base_model_path) / "config.json"
    if config_path.is_file():
        return base_model_path

    summary_path = Path(base_model_path) / "merge_summary.json"
    if not summary_path.is_file():
        raise RuntimeError(
            f"missing readable config.json and merge_summary.json under {base_model_path}"
        )
    summary = json.loads(summary_path.read_text())
    source = summary.get("base_model_path")
    if not isinstance(source, str):
        raise RuntimeError(f"merge_summary.json missing base_model_path: {summary_path}")
    if not (Path(source) / "config.json").is_file():
        raise RuntimeError(f"merge_summary base_model_path has no config.json: {source}")
    return source


def materialize_merged_model_metadata(base_model_path: str, config_source: str) -> list[str]:
    copied: list[str] = []
    source_root = Path(config_source)
    target_root = Path(base_model_path)
    for filename in (
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ):
        source_path = source_root / filename
        if not source_path.is_file():
            continue
        target_path = target_root / filename
        if target_path.is_file():
            continue
        if target_path.is_symlink():
            target_path.unlink()
        target_path.write_bytes(source_path.read_bytes())
        copied.append(filename)
    return copied


def repair_missing_merged_shards(base_model_path: str, config_source: str) -> list[str]:
    source_root = Path(config_source)
    target_root = Path(base_model_path)
    index_path = target_root / "model.safetensors.index.json"
    if not index_path.is_file():
        return []
    summary_path = target_root / "merge_summary.json"
    if not summary_path.is_file():
        raise RuntimeError(
            f"missing merge_summary.json for merged checkpoint shard repair: {summary_path}"
        )

    shard_names = sorted(set(json.loads(index_path.read_text())["weight_map"].values()))
    summary = json.loads(summary_path.read_text())
    unchanged_shards = {
        shard_info["shard"]
        for worker in summary.get("worker_results", [])
        for shard_info in worker.get("shards", [])
        if shard_info.get("changed_params") == 0
    }
    repaired: list[str] = []
    for shard_name in shard_names:
        target_path = target_root / shard_name
        if target_path.exists():
            continue
        if shard_name not in unchanged_shards:
            raise RuntimeError(
                f"missing merged shard is not marked unchanged in merge_summary.json: {shard_name}"
            )
        source_path = source_root / shard_name
        if not source_path.exists():
            continue
        if target_path.is_symlink():
            target_path.unlink()
        target_path.symlink_to(source_path)
        repaired.append(shard_name)
    return repaired


def copy_tokenizer_artifacts(config_source: str, export_dir: Path) -> list[str]:
    source_root = Path(config_source)
    copied: list[str] = []
    for filename in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "spiece.model",
        "sentencepiece.bpe.model",
    ):
        source_path = source_root / filename
        if not source_path.is_file():
            continue
        target_path = export_dir / filename
        target_path.write_bytes(source_path.read_bytes())
        copied.append(filename)
    return copied


def build_fp32_skip_modules(config) -> list[str]:
    # FineGrainedFP8 block quantization requires 128x128 tiling. GLM-5's
    # kv_a_proj_with_mqa has width 576, and the indexer weights_proj stays bf16
    # by design in the upstream HF model. vLLM also fuses q_a + kv_a into one
    # fused_qkv_a_proj, so q_a must stay bf16 as well or vLLM rejects the fused
    # layer as partially quantized.
    layer_ids = range(int(config.num_hidden_layers))
    return [
        "lm_head",
        "model.embed_tokens",
        *[
            f"model.layers.{layer_idx}.input_layernorm"
            for layer_idx in layer_ids
        ],
        *[
            f"model.layers.{layer_idx}.post_attention_layernorm"
            for layer_idx in layer_ids
        ],
        *[
            f"model.layers.{layer_idx}.mlp.gate"
            for layer_idx in layer_ids
        ],
        *[
            f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias"
            for layer_idx in layer_ids
        ],
        *[
            f"model.layers.{layer_idx}.self_attn.indexer.k_norm"
            for layer_idx in layer_ids
        ],
        *[
            f"model.layers.{layer_idx}.self_attn.indexer.k_norm.bias"
            for layer_idx in layer_ids
        ],
        *[
            f"model.layers.{layer_idx}.self_attn.indexers_proj"
            for layer_idx in layer_ids
        ],
        *[
            f"model.layers.{layer_idx}.self_attn.indexer.weights_proj"
            for layer_idx in layer_ids
        ],
        *[
            f"model.layers.{layer_idx}.self_attn.kv_a_layernorm"
            for layer_idx in layer_ids
        ],
        *[
            f"model.layers.{layer_idx}.self_attn.q_a_layernorm"
            for layer_idx in layer_ids
        ],
        *[
            f"model.layers.{layer_idx}.self_attn.q_a_proj"
            for layer_idx in layer_ids
        ],
        *[
            f"model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa"
            for layer_idx in layer_ids
        ],
    ]


def normalize_glm5_config_for_fp8(config) -> None:
    # Transformers' FineGrainedFP8 MoE wrapper currently assumes a generic
    # `num_experts` attribute, while GLM-5 remote config exposes
    # `n_routed_experts` instead. Add the alias before module replacement so
    # the wrapper does not crash during model construction.
    if getattr(config, "model_type", None) == "glm_moe_dsa" and getattr(config, "num_experts", None) is None:
        routed = getattr(config, "n_routed_experts", None)
        if routed is not None:
            config.num_experts = int(routed)


def sanitize_generation_config_for_save(model) -> list[str]:
    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        return []

    fixed: list[str] = []
    if bool(getattr(generation_config, "do_sample", False)):
        return fixed

    sample_only_defaults = {
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 1.0,
        "min_p": None,
        "typical_p": 1.0,
        "epsilon_cutoff": 0.0,
        "eta_cutoff": 0.0,
    }
    for field_name, neutral_value in sample_only_defaults.items():
        current = getattr(generation_config, field_name, None)
        if current != neutral_value:
            setattr(generation_config, field_name, neutral_value)
            fixed.append(field_name)
    return fixed


def quantize_block_fp8_tensor(
    tensor: torch.Tensor,
    *,
    block_size: tuple[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = tensor.shape
    block_m, block_n = block_size
    padded_rows = math.ceil(rows / block_m) * block_m
    padded_cols = math.ceil(cols / block_n) * block_n
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    weight_fp32 = tensor.to(device=device, dtype=torch.float32, non_blocking=True)
    if padded_rows != rows or padded_cols != cols:
        padded = torch.zeros(
            (padded_rows, padded_cols),
            dtype=torch.float32,
            device=device,
        )
        padded[:rows, :cols] = weight_fp32
        weight_fp32 = padded
    reshaped = weight_fp32.reshape(padded_rows // block_m, block_m, padded_cols // block_n, block_n)
    max_abs = reshaped.abs().amax(dim=(1, 3))
    safe_max = torch.where(max_abs > 0, max_abs, torch.ones_like(max_abs))
    scales = fp8_max / safe_max
    scales = torch.where(max_abs > 0, scales, torch.ones_like(scales))
    quantized = torch.clamp(
        reshaped * scales.unsqueeze(1).unsqueeze(3),
        min=torch.finfo(torch.float8_e4m3fn).min,
        max=fp8_max,
    ).to(torch.float8_e4m3fn)
    quantized = quantized.reshape(padded_rows, padded_cols)[:rows, :cols].to("cpu")
    scale_inv = scales.reciprocal().to(dtype=torch.float32, device="cpu")
    del weight_fp32, reshaped, max_abs, safe_max, scales
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return quantized, scale_inv


def rewrite_moe_expert_shards(
    *,
    base_model_path: str,
    config,
    export_dir: Path,
    weight_block_size: tuple[int, int],
) -> list[str]:
    source_root = Path(base_model_path)
    source_index_path = source_root / "model.safetensors.index.json"
    if not source_index_path.is_file():
        raise RuntimeError(f"missing source safetensors index: {source_index_path}")
    export_index_path = export_dir / "model.safetensors.index.json"
    if not export_index_path.is_file():
        raise RuntimeError(f"missing export safetensors index: {export_index_path}")

    source_weight_map = json.loads(source_index_path.read_text())["weight_map"]
    index_payload = json.loads(export_index_path.read_text())
    weight_map = index_payload["weight_map"]
    mlp_layer_types = list(getattr(config, "mlp_layer_types", []))
    if not mlp_layer_types:
        raise RuntimeError("config.mlp_layer_types is empty")

    open_handles: dict[str, object] = {}
    fixed_shards: list[str] = []
    quant_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _load_tensor(name: str) -> torch.Tensor:
        shard_name = source_weight_map.get(name)
        if shard_name is None:
            raise KeyError(f"missing source tensor in merged checkpoint: {name}")
        handle = open_handles.get(shard_name)
        if handle is None:
            handle = safe_open(str(source_root / shard_name), framework="pt", device="cpu")
            open_handles[shard_name] = handle
        return handle.get_tensor(name)

    experts_per_chunk = 32
    num_experts = int(getattr(config, "n_routed_experts"))
    for layer_idx, layer_type in enumerate(mlp_layer_types):
        if layer_type != "sparse":
            continue
        for chunk_start in range(0, num_experts, experts_per_chunk):
            chunk_stop = min(chunk_start + experts_per_chunk, num_experts)
            shard_name = f"model-experts-fix-layer-{layer_idx:03d}-chunk-{chunk_start:03d}.safetensors"
            shard_tensors: dict[str, torch.Tensor] = {}
            for expert_idx in range(chunk_start, chunk_stop):
                expert_prefix = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"
                for proj_name in ("gate_proj", "up_proj", "down_proj"):
                    weight_key = f"{expert_prefix}.{proj_name}.weight"
                    quantized, scale_inv = quantize_block_fp8_tensor(
                        _load_tensor(weight_key),
                        block_size=weight_block_size,
                        device=quant_device,
                    )
                    shard_tensors[weight_key] = quantized.contiguous()
                    shard_tensors[f"{expert_prefix}.{proj_name}.weight_scale_inv"] = scale_inv.contiguous()
                    weight_map[weight_key] = shard_name
                    weight_map[f"{expert_prefix}.{proj_name}.weight_scale_inv"] = shard_name
            save_file(shard_tensors, str(export_dir / shard_name), metadata={"format": "pt"})
            fixed_shards.append(shard_name)
            del shard_tensors
            if quant_device.type == "cuda":
                torch.cuda.empty_cache()

    total_size = sum(path.stat().st_size for path in export_dir.glob("*.safetensors"))
    index_payload.setdefault("metadata", {})["total_size"] = total_size
    export_index_path.write_text(json.dumps(index_payload, ensure_ascii=True, indent=2))
    print(
        json.dumps(
            {
                "phase": "moe_fix_done",
                "fixed_shard_count": len(fixed_shards),
                "fixed_sparse_layer_count": sum(1 for x in mlp_layer_types if x == "sparse"),
                "total_safetensors_bytes": total_size,
            },
            ensure_ascii=True,
        ),
        flush=True,
    )
    return fixed_shards


def scrub_stale_moe_keys(export_dir: Path, config) -> dict[str, object]:
    index_path = export_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise RuntimeError(f"missing export safetensors index: {index_path}")

    index_payload = json.loads(index_path.read_text())
    weight_map = index_payload["weight_map"]
    n_routed_experts = int(getattr(config, "n_routed_experts"))
    expert_key_re = re.compile(r"\.mlp\.experts\.(\d+)\.")

    def _is_overrange_expert_key(key: str) -> bool:
        match = expert_key_re.search(key)
        return match is not None and int(match.group(1)) >= n_routed_experts

    overrange_index_keys = [key for key in weight_map if _is_overrange_expert_key(key)]
    for key in overrange_index_keys:
        del weight_map[key]

    fix_keys = {
        key
        for key, shard_name in weight_map.items()
        if shard_name.startswith("model-experts-fix-layer-")
    }
    if not fix_keys:
        raise RuntimeError("no repaired model-experts-fix shards found in export index")

    scrubbed: list[dict[str, object]] = []
    for shard_path in sorted(export_dir.glob("model-*-of-*.safetensors")):
        with safe_open(str(shard_path), framework="pt", device="cpu") as reader:
            file_keys = list(reader.keys())
            remove_keys = [
                key
                for key in file_keys
                if key in fix_keys or _is_overrange_expert_key(key)
            ]
            if not remove_keys:
                continue
            keep_tensors = {
                key: reader.get_tensor(key)
                for key in file_keys
                if key not in remove_keys
            }

        tmp_path = shard_path.with_suffix(shard_path.suffix + ".tmp")
        backup_path = shard_path.with_suffix(shard_path.suffix + ".pre_moe_scrub")
        if tmp_path.exists():
            tmp_path.unlink()
        if backup_path.exists():
            backup_path.unlink()
        save_file(keep_tensors, str(tmp_path), metadata={"format": "pt"})
        shard_path.rename(backup_path)
        tmp_path.rename(shard_path)
        backup_path.unlink()
        scrubbed.append(
            {
                "shard": shard_path.name,
                "removed_key_count": len(remove_keys),
                "removed_overrange_count": sum(
                    1 for key in remove_keys if _is_overrange_expert_key(key)
                ),
                "removed_redirected_count": sum(1 for key in remove_keys if key in fix_keys),
            }
        )

    total_size = sum(path.stat().st_size for path in export_dir.glob("*.safetensors"))
    index_payload.setdefault("metadata", {})["total_size"] = total_size
    index_payload["metadata"]["scrubbed_overrange_expert_key_count"] = len(overrange_index_keys)
    index_payload["metadata"]["scrubbed_n_routed_experts"] = n_routed_experts
    index_path.write_text(json.dumps(index_payload, ensure_ascii=True, indent=2))

    result = {
        "phase": "moe_stale_key_scrub_done",
        "scrubbed_shard_count": len(scrubbed),
        "scrubbed_shards": scrubbed,
        "removed_index_overrange_key_count": len(overrange_index_keys),
        "n_routed_experts": n_routed_experts,
        "total_safetensors_bytes": total_size,
    }
    print(json.dumps(result, ensure_ascii=True), flush=True)
    return result


def add_mtp_layer_from_source(
    *,
    base_model_path: str,
    export_dir: Path,
    weight_block_size: tuple[int, int],
) -> dict[str, object]:
    source_root = Path(base_model_path)
    source_index_path = source_root / "model.safetensors.index.json"
    export_index_path = export_dir / "model.safetensors.index.json"
    if not source_index_path.is_file() or not export_index_path.is_file():
        raise RuntimeError("missing source/export safetensors index for MTP repair")

    source_weight_map = json.loads(source_index_path.read_text())["weight_map"]
    layer78_keys = sorted(key for key in source_weight_map if key.startswith("model.layers.78."))
    if not layer78_keys:
        return {"phase": "mtp_layer_export_skipped", "reason": "source_has_no_layer_78"}

    index_payload = json.loads(export_index_path.read_text())
    weight_map = index_payload["weight_map"]
    existing_layer78 = [key for key in weight_map if key.startswith("model.layers.78.")]
    if existing_layer78:
        return {
            "phase": "mtp_layer_export_skipped",
            "reason": "export_already_has_layer_78",
            "existing_key_count": len(existing_layer78),
        }

    # Match the official GLM-5.1-FP8 MTP layout: norms and gates stay bf16/fp32,
    # while all 2-D projection matrices are block-fp8 with *_weight_scale_inv.
    keep_bf16_or_fp32_suffixes = (
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        ".eh_proj.weight",
        ".enorm.weight",
        ".hnorm.weight",
        ".shared_head.norm.weight",
        ".mlp.gate.weight",
        ".mlp.gate.e_score_correction_bias",
        ".self_attn.indexer.k_norm.weight",
        ".self_attn.indexer.k_norm.bias",
        ".self_attn.indexer.weights_proj.weight",
        ".self_attn.kv_a_layernorm.weight",
        ".self_attn.q_a_layernorm.weight",
    )
    expert_key_re = re.compile(r"\.mlp\.experts\.(\d+)\.")
    open_handles: dict[str, object] = {}
    quant_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    shard_tensors: dict[str, torch.Tensor] = {}
    quantized_count = 0
    copied_count = 0

    def _load_tensor(name: str) -> torch.Tensor:
        shard_name = source_weight_map[name]
        handle = open_handles.get(shard_name)
        if handle is None:
            handle = safe_open(str(source_root / shard_name), framework="pt", device="cpu")
            open_handles[shard_name] = handle
        return handle.get_tensor(name)

    for key in layer78_keys:
        if key.endswith(keep_bf16_or_fp32_suffixes):
            shard_tensors[key] = _load_tensor(key).contiguous()
            copied_count += 1
            continue
        tensor = _load_tensor(key)
        if tensor.ndim != 2:
            shard_tensors[key] = tensor.contiguous()
            copied_count += 1
            continue
        quantized, scale_inv = quantize_block_fp8_tensor(
            tensor,
            block_size=weight_block_size,
            device=quant_device,
        )
        shard_tensors[key] = quantized.contiguous()
        shard_tensors[f"{key}_scale_inv"] = scale_inv.contiguous()
        quantized_count += 1

    for handle in open_handles.values():
        handle.__exit__(None, None, None)

    shard_name = "model-mtp-layer-078.safetensors"
    save_file(shard_tensors, str(export_dir / shard_name), metadata={"format": "pt"})
    for key in shard_tensors:
        weight_map[key] = shard_name

    config_path = export_dir / "config.json"
    if config_path.is_file():
        config_payload = json.loads(config_path.read_text())
        quant_config = config_payload.setdefault("quantization_config", {})
        modules_to_not_convert = quant_config.setdefault("modules_to_not_convert", [])
        for module_name in (
            "model.layers.78.eh_proj",
            "model.layers.78.enorm",
            "model.layers.78.hnorm",
            "model.layers.78.input_layernorm",
            "model.layers.78.mlp.gate",
            "model.layers.78.mlp.gate.e_score_correction_bias",
            "model.layers.78.post_attention_layernorm",
            "model.layers.78.self_attn.indexer.k_norm",
            "model.layers.78.self_attn.indexer.k_norm.bias",
            "model.layers.78.self_attn.indexers_proj",
            "model.layers.78.self_attn.kv_a_layernorm",
            "model.layers.78.self_attn.q_a_layernorm",
            "model.layers.78.shared_head.norm",
        ):
            if module_name not in modules_to_not_convert:
                modules_to_not_convert.append(module_name)
        for module_name in (
            "model.layers.78.self_attn.kv_a_proj_with_mqa",
            "model.layers.78.self_attn.indexer.weights_proj",
        ):
            if module_name in modules_to_not_convert:
                modules_to_not_convert.remove(module_name)
        config_path.write_text(json.dumps(config_payload, ensure_ascii=True, indent=2))

    total_size = sum(path.stat().st_size for path in export_dir.glob("*.safetensors"))
    index_payload.setdefault("metadata", {})["total_size"] = total_size
    index_payload["metadata"]["mtp_layer_78_exported"] = True
    export_index_path.write_text(json.dumps(index_payload, ensure_ascii=True, indent=2))
    result = {
        "phase": "mtp_layer_export_done",
        "source_layer78_key_count": len(layer78_keys),
        "exported_key_count": len(shard_tensors),
        "quantized_weight_count": quantized_count,
        "copied_weight_count": copied_count,
        "shard": shard_name,
        "total_safetensors_bytes": total_size,
    }
    print(json.dumps(result, ensure_ascii=True), flush=True)
    return result


def module_name_from_tensor_key(key: str) -> str:
    for suffix in (
        ".weight_scale_inv",
        ".weight",
        ".bias",
        ".e_score_correction_bias",
    ):
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return key


def export_shardwise_fp8(
    *,
    base_model_path: str,
    config_source: str,
    config,
    export_dir: Path,
    modules_to_not_convert: list[str],
    weight_block_size: tuple[int, int],
) -> dict[str, object]:
    source_root = Path(base_model_path)
    source_index_path = source_root / "model.safetensors.index.json"
    if not source_index_path.is_file():
        raise RuntimeError(f"missing source safetensors index: {source_index_path}")

    config_payload = config.to_dict()
    config_payload["quantization_config"] = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": list(weight_block_size),
        "modules_to_not_convert": modules_to_not_convert,
    }
    (export_dir / "config.json").write_text(json.dumps(config_payload, ensure_ascii=True, indent=2))

    generation_config_path = Path(config_source) / "generation_config.json"
    if generation_config_path.is_file():
        (export_dir / "generation_config.json").write_bytes(generation_config_path.read_bytes())
    copied_tokenizer_artifacts = copy_tokenizer_artifacts(config_source, export_dir)

    source_weight_map = json.loads(source_index_path.read_text())["weight_map"]
    shard_names = sorted(set(source_weight_map.values()))
    n_routed_experts = int(getattr(config, "n_routed_experts"))
    expert_key_re = re.compile(r"\.mlp\.experts\.(\d+)\.")
    skip_modules = set(modules_to_not_convert)
    quant_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weight_map: dict[str, str] = {}
    quantized_weight_count = 0
    copied_tensor_count = 0
    skipped_expert_tensor_count = 0
    skipped_layer78_tensor_count = 0

    def should_skip_main_key(key: str) -> bool:
        if key.startswith("model.layers.78."):
            return True
        match = expert_key_re.search(key)
        return match is not None

    def should_quantize(key: str, tensor: torch.Tensor) -> bool:
        if not key.endswith(".weight") or tensor.ndim != 2:
            return False
        return module_name_from_tensor_key(key) not in skip_modules

    for shard_idx, shard_name in enumerate(shard_names, start=1):
        shard_tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(source_root / shard_name), framework="pt", device="cpu") as reader:
            for key in reader.keys():
                if should_skip_main_key(key):
                    if key.startswith("model.layers.78."):
                        skipped_layer78_tensor_count += 1
                    else:
                        skipped_expert_tensor_count += 1
                    continue
                tensor = reader.get_tensor(key)
                if should_quantize(key, tensor):
                    quantized, scale_inv = quantize_block_fp8_tensor(
                        tensor,
                        block_size=weight_block_size,
                        device=quant_device,
                    )
                    shard_tensors[key] = quantized.contiguous()
                    scale_key = f"{key}_scale_inv"
                    shard_tensors[scale_key] = scale_inv.contiguous()
                    weight_map[key] = shard_name
                    weight_map[scale_key] = shard_name
                    quantized_weight_count += 1
                else:
                    shard_tensors[key] = tensor.contiguous()
                    weight_map[key] = shard_name
                    copied_tensor_count += 1
                del tensor
        save_file(shard_tensors, str(export_dir / shard_name), metadata={"format": "pt"})
        del shard_tensors
        if quant_device.type == "cuda":
            torch.cuda.empty_cache()
        print(
            json.dumps(
                {
                    "phase": "shardwise_export_progress",
                    "shard": shard_name,
                    "shard_index": shard_idx,
                    "shard_count": len(shard_names),
                },
                ensure_ascii=True,
            ),
            flush=True,
        )

    index_payload = {
        "metadata": {
            "total_size": sum(path.stat().st_size for path in export_dir.glob("*.safetensors")),
        },
        "weight_map": weight_map,
    }
    (export_dir / "model.safetensors.index.json").write_text(
        json.dumps(index_payload, ensure_ascii=True, indent=2)
    )
    result = {
        "phase": "shardwise_export_done",
        "source_shard_count": len(shard_names),
        "quantized_weight_count": quantized_weight_count,
        "copied_tensor_count": copied_tensor_count,
        "skipped_expert_tensor_count": skipped_expert_tensor_count,
        "skipped_layer78_tensor_count": skipped_layer78_tensor_count,
        "copied_tokenizer_artifacts": copied_tokenizer_artifacts,
        "n_routed_experts": n_routed_experts,
    }
    print(json.dumps(result, ensure_ascii=True), flush=True)
    return result


def main() -> None:
    args = parse_args()
    export_dir = Path(args.export_dir)
    if export_dir.exists() and any(export_dir.iterdir()):
        raise RuntimeError(f"export_dir already exists and is non-empty: {export_dir}")
    export_dir.mkdir(parents=True, exist_ok=True)

    cuda_count = torch.cuda.device_count()
    if cuda_count < 1:
        raise RuntimeError("FP8 quantization requires at least one visible CUDA device")
    max_memory_by_gpu = os.environ.get("GLM5_FP8_MAX_MEMORY_BY_GPU")
    if max_memory_by_gpu:
        max_memory = {i: "130GiB" for i in range(cuda_count)}
        for part in max_memory_by_gpu.split(","):
            gpu_id, value = part.split(":", 1)
            max_memory[int(gpu_id)] = value
    else:
        max_memory_gib = int(os.environ.get("GLM5_FP8_MAX_MEMORY_GIB", "130"))
        max_memory = {i: f"{max_memory_gib}GiB" for i in range(cuda_count)}

    config_source = resolve_config_source(args.base_model_path)
    print(
        json.dumps(
            {"phase": "config_source", "config_source": config_source},
            ensure_ascii=True,
        ),
        flush=True,
    )
    copied_metadata = materialize_merged_model_metadata(args.base_model_path, config_source)
    if copied_metadata:
        print(
            json.dumps(
                {"phase": "metadata_repaired", "copied_files": copied_metadata},
                ensure_ascii=True,
            ),
            flush=True,
        )
    repaired_shards = repair_missing_merged_shards(args.base_model_path, config_source)
    if repaired_shards:
        print(
            json.dumps(
                {
                    "phase": "shards_repaired",
                    "repaired_shards_count": len(repaired_shards),
                    "repaired_shards_sample": repaired_shards[:8],
                },
                ensure_ascii=True,
            ),
            flush=True,
        )
    config = AutoConfig.from_pretrained(
        config_source,
        trust_remote_code=args.trust_remote_code,
    )
    normalize_glm5_config_for_fp8(config)
    modules_to_not_convert = build_fp32_skip_modules(config)

    quant_cfg = FineGrainedFP8Config(modules_to_not_convert=modules_to_not_convert)
    if os.environ.get("GLM5_FP8_SHARDWISE") == "1":
        shardwise_result = export_shardwise_fp8(
            base_model_path=args.base_model_path,
            config_source=config_source,
            config=config,
            export_dir=export_dir,
            modules_to_not_convert=modules_to_not_convert,
            weight_block_size=tuple(quant_cfg.weight_block_size),
        )
        fixed_shards = rewrite_moe_expert_shards(
            base_model_path=args.base_model_path,
            config=config,
            export_dir=export_dir,
            weight_block_size=tuple(quant_cfg.weight_block_size),
        )
        scrub_result = scrub_stale_moe_keys(export_dir, config)
        mtp_result = add_mtp_layer_from_source(
            base_model_path=args.base_model_path,
            export_dir=export_dir,
            weight_block_size=tuple(quant_cfg.weight_block_size),
        )
        meta = {
            "base_model_path": args.base_model_path,
            "export_dir": str(export_dir),
            "cuda_count": cuda_count,
            "quantization_method": "fp8",
            "is_quantized": True,
            "quantizer": "FineGrainedFP8Config",
            "export_mode": "shardwise",
            "modules_to_not_convert_count": len(modules_to_not_convert),
            "modules_to_not_convert": modules_to_not_convert,
            "shardwise_export": shardwise_result,
            "moe_fix_shard_count": len(fixed_shards),
            "moe_stale_key_scrub": scrub_result,
            "mtp_layer_export": mtp_result,
        }
        (export_dir / "fp8_quant_meta.json").write_text(json.dumps(meta, ensure_ascii=True, indent=2))
        print(json.dumps({"phase": "done", **meta}, ensure_ascii=True), flush=True)
        return

    device_map = os.environ.get("GLM5_FP8_DEVICE_MAP", "auto")
    print(
        json.dumps(
            {
                "phase": "load_start",
                "base_model_path": args.base_model_path,
                "export_dir": str(export_dir),
                "cuda_count": cuda_count,
                "max_memory": max_memory,
                "quant_method": "fp8",
                "quantizer": "FineGrainedFP8Config",
                "device_map": device_map,
                "modules_to_not_convert_count": len(modules_to_not_convert),
                "modules_to_not_convert_sample": modules_to_not_convert[:4],
            },
            ensure_ascii=True,
        ),
        flush=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        config=config,
        quantization_config=quant_cfg,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    print(
        json.dumps(
            {
                "phase": "load_done",
                "quantization_method": str(getattr(model, "quantization_method", None)),
                "is_quantized": bool(getattr(model, "is_quantized", False)),
            },
            ensure_ascii=True,
        ),
        flush=True,
    )
    quantization_method = str(getattr(model, "quantization_method", None))
    is_quantized = bool(getattr(model, "is_quantized", False))

    fixed_generation_fields = sanitize_generation_config_for_save(model)
    if fixed_generation_fields:
        print(
            json.dumps(
                {
                    "phase": "generation_config_sanitized",
                    "fields": fixed_generation_fields,
                },
                ensure_ascii=True,
            ),
            flush=True,
        )

    model.save_pretrained(export_dir, safe_serialization=True, max_shard_size="10GB")
    copied_tokenizer_artifacts = copy_tokenizer_artifacts(config_source, export_dir)
    if copied_tokenizer_artifacts:
        print(
            json.dumps(
                {
                    "phase": "tokenizer_artifacts_copied",
                    "files": copied_tokenizer_artifacts,
                },
                ensure_ascii=True,
            ),
            flush=True,
        )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    fixed_shards = rewrite_moe_expert_shards(
        base_model_path=args.base_model_path,
        config=config,
        export_dir=export_dir,
        weight_block_size=tuple(quant_cfg.weight_block_size),
    )
    scrub_result = scrub_stale_moe_keys(export_dir, config)
    mtp_result = add_mtp_layer_from_source(
        base_model_path=args.base_model_path,
        export_dir=export_dir,
        weight_block_size=tuple(quant_cfg.weight_block_size),
    )

    meta = {
        "base_model_path": args.base_model_path,
        "export_dir": str(export_dir),
        "cuda_count": cuda_count,
        "quantization_method": quantization_method,
        "is_quantized": is_quantized,
        "quantizer": "FineGrainedFP8Config",
        "modules_to_not_convert_count": len(modules_to_not_convert),
        "modules_to_not_convert": modules_to_not_convert,
        "moe_fix_shard_count": len(fixed_shards),
        "moe_stale_key_scrub": scrub_result,
        "mtp_layer_export": mtp_result,
    }
    (export_dir / "fp8_quant_meta.json").write_text(json.dumps(meta, ensure_ascii=True, indent=2))
    print(json.dumps({"phase": "done", **meta}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
GLM5_REF_quantize_glm5_finegrained_fp8_py
  cat > "${scripts_dir}/convert_megatron_lora_to_peft.py" <<'GLM5_REF_convert_megatron_lora_to_peft_py'
from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import torch


ADAPTER_RE = re.compile(r"mp_rank_(\d+)_(\d+)_adapter\.pt$")
LOCAL_EXPERT_RE = re.compile(
    r"^decoder\.layers\.(\d+)\.mlp\.experts\.local_experts\.(\d+)\."
    r"(linear_fc[12])\.adapter\.(linear_in|linear_out)\.weight$"
)
SHARED_RE = re.compile(
    r"^decoder\.layers\.(\d+)\.(self_attention|mlp)(?:\.([^.]+))?"
    r"(?:\.([^.]+))?\.adapter\.(linear_in|linear_out)\.weight$"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-path", type=Path, required=True)
    p.add_argument("--output-path", type=Path, required=True)
    p.add_argument("--base-model-path", type=Path, default=None)
    p.add_argument("--tp-size", type=int, default=None)
    p.add_argument("--expert-parallel-size", type=int, default=None)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def parse_adapter_files(path: Path) -> list[tuple[int, int, Path]]:
    parsed = []
    for item in sorted(path.glob("mp_rank_*_adapter.pt")):
        m = ADAPTER_RE.fullmatch(item.name)
        if m:
            parsed.append((int(m.group(1)), int(m.group(2)), item))
    if not parsed:
        raise RuntimeError(f"no mp_rank adapter shards found under {path}")
    return parsed


def load_state(path: Path) -> dict[str, torch.Tensor]:
    data = torch.load(path, map_location="cpu")
    state = data.get("adapter_state_dict", data)
    if not isinstance(state, dict):
        raise RuntimeError(f"adapter shard has no tensor dict: {path}")
    return state


def infer_tp_size(files: list[tuple[int, int, Path]], explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    tp_ranks = sorted({tp for tp, _, _ in files})
    if tp_ranks != list(range(len(tp_ranks))):
        raise RuntimeError(f"cannot infer contiguous TP ranks from {tp_ranks}")
    return len(tp_ranks)


def select_shared_tp_group(files: list[tuple[int, int, Path]], tp_size: int) -> list[tuple[int, int, Path]]:
    by_global = {global_rank: (tp_rank, global_rank, path) for tp_rank, global_rank, path in files}
    for start in sorted(global_rank for _, global_rank, _ in files):
        group = []
        for tp_rank in range(tp_size):
            item = by_global.get(start + tp_rank)
            if item is None or item[0] != tp_rank:
                break
            group.append(item)
        if len(group) == tp_size:
            return group
    raise RuntimeError(f"cannot find complete shared TP group for tp_size={tp_size}")


def lora_type(raw_key: str) -> str:
    if ".adapter.linear_in.weight" in raw_key:
        return "lora_A"
    if ".adapter.linear_out.weight" in raw_key:
        return "lora_B"
    raise RuntimeError(f"cannot determine LoRA type for {raw_key}")


def shared_split_dim(raw_key: str) -> int | None:
    typ = lora_type(raw_key)
    if typ == "lora_B":
        return 0
    if raw_key == "output_layer.adapter.linear_in.weight":
        return 0
    if ".self_attention.linear_proj.adapter.linear_in.weight" in raw_key:
        return 1
    if ".mlp.linear_fc2.adapter.linear_in.weight" in raw_key:
        return 1
    if ".mlp.shared_experts.linear_fc2.adapter.linear_in.weight" in raw_key:
        return 1
    return 0


def split_fused_gate_up_lora_b(tensor: torch.Tensor, tp_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if tp_size <= 1:
        gate, up = tensor.chunk(2, dim=0)
        return gate.contiguous(), up.contiguous()
    if tensor.shape[0] % tp_size != 0:
        raise RuntimeError(f"cannot split fused gate/up tensor with shape {tuple(tensor.shape)} by TP={tp_size}")
    shard_size = tensor.shape[0] // tp_size
    gate_parts = []
    up_parts = []
    for shard in tensor.split(shard_size, dim=0):
        gate, up = shard.chunk(2, dim=0)
        gate_parts.append(gate)
        up_parts.append(up)
    return torch.cat(gate_parts, dim=0).contiguous(), torch.cat(up_parts, dim=0).contiguous()


def split_fused_local_expert_lora_b(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gate, up = tensor.chunk(2, dim=0)
    return gate.contiguous(), up.contiguous()


def peft_key(base_key: str, typ: str) -> str:
    return f"base_model.model.{base_key}.{typ}.weight"


def materialize(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().contiguous().clone()


def add_fused_pair(
    out: dict[str, torch.Tensor],
    prefix: str,
    typ: str,
    tensor: torch.Tensor,
    *,
    tp_size_for_b: int,
) -> None:
    if typ == "lora_A":
        out[peft_key(f"{prefix}.gate_proj", typ)] = materialize(tensor)
        out[peft_key(f"{prefix}.up_proj", typ)] = materialize(tensor)
        return
    gate, up = split_fused_gate_up_lora_b(tensor, tp_size_for_b)
    out[peft_key(f"{prefix}.gate_proj", typ)] = materialize(gate)
    out[peft_key(f"{prefix}.up_proj", typ)] = materialize(up)


def add_local_expert_fused_pair(
    out: dict[str, torch.Tensor],
    prefix: str,
    typ: str,
    tensor: torch.Tensor,
) -> None:
    if typ == "lora_A":
        out[peft_key(f"{prefix}.gate_proj", typ)] = materialize(tensor)
        out[peft_key(f"{prefix}.up_proj", typ)] = materialize(tensor)
        return
    gate, up = split_fused_local_expert_lora_b(tensor)
    out[peft_key(f"{prefix}.gate_proj", typ)] = materialize(gate)
    out[peft_key(f"{prefix}.up_proj", typ)] = materialize(up)


def convert_shared_key(raw_key: str, tensor: torch.Tensor, tp_size: int) -> dict[str, torch.Tensor]:
    typ = lora_type(raw_key)
    if raw_key.startswith("output_layer.adapter."):
        return {peft_key("lm_head", typ): materialize(tensor)}

    m = SHARED_RE.fullmatch(raw_key)
    if not m:
        raise RuntimeError(f"unrecognized shared key: {raw_key}")
    layer, block, name1, name2, _ = m.groups()
    out: dict[str, torch.Tensor] = {}

    if block == "self_attention":
        mapping = {
            "linear_proj": "self_attn.o_proj",
            "linear_q_down_proj": "self_attn.q_a_proj",
            "linear_q_up_proj": "self_attn.q_b_proj",
            "linear_kv_down_proj": "self_attn.kv_a_proj_with_mqa",
            "linear_kv_up_proj": "self_attn.kv_b_proj",
        }
        target = mapping.get(name1)
        if target is None:
            raise RuntimeError(f"unrecognized self_attention target in {raw_key}")
        return {peft_key(f"model.layers.{layer}.{target}", typ): materialize(tensor)}

    if block != "mlp":
        raise RuntimeError(f"unrecognized block in {raw_key}")

    if name1 == "linear_fc1":
        add_fused_pair(out, f"model.layers.{layer}.mlp", typ, tensor, tp_size_for_b=tp_size)
    elif name1 == "linear_fc2":
        out[peft_key(f"model.layers.{layer}.mlp.down_proj", typ)] = materialize(tensor)
    elif name1 == "shared_experts" and name2 == "linear_fc1":
        add_fused_pair(out, f"model.layers.{layer}.mlp.shared_experts", typ, tensor, tp_size_for_b=tp_size)
    elif name1 == "shared_experts" and name2 == "linear_fc2":
        out[peft_key(f"model.layers.{layer}.mlp.shared_experts.down_proj", typ)] = materialize(tensor)
    else:
        raise RuntimeError(f"unrecognized mlp target in {raw_key}")
    return out


def convert_local_expert_key(
    raw_key: str,
    tensor: torch.Tensor,
    *,
    global_expert_id: int,
) -> dict[str, torch.Tensor]:
    m = LOCAL_EXPERT_RE.fullmatch(raw_key)
    if not m:
        raise RuntimeError(f"unrecognized local expert key: {raw_key}")
    layer, _, linear_name, _ = m.groups()
    typ = lora_type(raw_key)
    prefix = f"model.layers.{layer}.mlp.experts.{global_expert_id}"
    out: dict[str, torch.Tensor] = {}
    if linear_name == "linear_fc1":
        add_local_expert_fused_pair(out, prefix, typ, tensor)
    elif linear_name == "linear_fc2":
        out[peft_key(f"{prefix}.down_proj", typ)] = materialize(tensor)
    else:
        raise RuntimeError(f"unrecognized local expert linear target in {raw_key}")
    return out


def infer_local_experts(state: dict[str, torch.Tensor]) -> int:
    ids = []
    for key in state:
        m = LOCAL_EXPERT_RE.fullmatch(key)
        if m:
            ids.append(int(m.group(2)))
    if not ids:
        raise RuntimeError("no local expert keys found in adapter shard")
    expected = list(range(max(ids) + 1))
    present = sorted(set(ids))
    if present != expected:
        raise RuntimeError(f"local expert ids are not contiguous from zero: {present}")
    return len(present)


def load_base_shapes(base_model_path: Path | None) -> dict[str, tuple[int, ...]]:
    if base_model_path is None:
        return {}
    from safetensors import safe_open

    index = json.loads((base_model_path / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    by_shard: dict[str, list[str]] = defaultdict(list)
    for key, shard in weight_map.items():
        by_shard[shard].append(key)

    shapes = {}
    for shard, keys in by_shard.items():
        with safe_open(str(base_model_path / shard), framework="pt", device="cpu") as f:
            for key in keys:
                try:
                    shapes[key] = tuple(f.get_slice(key).get_shape())
                except AttributeError:
                    shapes[key] = tuple(f.get_tensor(key).shape)
    return shapes


def validate_shapes(out: dict[str, torch.Tensor], base_shapes: dict[str, tuple[int, ...]]) -> None:
    if not base_shapes:
        return
    missing = []
    mismatched = []
    for key, tensor in out.items():
        base_key = key.removeprefix("base_model.model.").replace(".lora_A.weight", ".weight").replace(
            ".lora_B.weight", ".weight"
        )
        base_shape = base_shapes.get(base_key)
        if base_shape is None:
            missing.append(base_key)
            continue
        expected_shape = (tensor.shape[0], tensor.shape[1])
        if key.endswith(".lora_A.weight"):
            expected_shape = (tensor.shape[1],)
            if expected_shape[0] != base_shape[1]:
                mismatched.append((key, tuple(tensor.shape), base_key, base_shape))
        else:
            expected_shape = (tensor.shape[0],)
            if expected_shape[0] != base_shape[0]:
                mismatched.append((key, tuple(tensor.shape), base_key, base_shape))
    if missing or mismatched:
        raise RuntimeError(
            f"shape validation failed: missing={missing[:5]} mismatched={mismatched[:5]} "
            f"missing_count={len(missing)} mismatched_count={len(mismatched)}"
        )


def copy_metadata(input_path: Path, output_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    for name in ("adapter_config.json", "metadata.json"):
        src = input_path / name
        if src.is_file():
            shutil.copy2(src, output_path / name)


def main() -> None:
    args = parse_args()
    if args.output_path.exists() and any(args.output_path.iterdir()):
        if not args.force:
            raise RuntimeError(f"output path already exists and is non-empty: {args.output_path}")
        shutil.rmtree(args.output_path)
    args.output_path.mkdir(parents=True, exist_ok=True)

    files = parse_adapter_files(args.input_path)
    tp_size = infer_tp_size(files, args.tp_size)
    shared_group = select_shared_tp_group(files, tp_size)
    shared_states = [(tp_rank, global_rank, load_state(path)) for tp_rank, global_rank, path in shared_group]
    local_experts_per_rank = infer_local_experts(shared_states[0][2])
    expert_parallel_size = args.expert_parallel_size or (len(files) // tp_size)
    expected_experts = expert_parallel_size * tp_size * local_experts_per_rank

    out: dict[str, torch.Tensor] = {}
    shared_keys = [
        key for key in shared_states[0][2]
        if "local_experts" not in key
    ]
    for key in shared_keys:
        key_sets_ok = all(key in state for _, _, state in shared_states)
        if not key_sets_ok:
            raise RuntimeError(f"shared key missing from one or more TP shards: {key}")
        dim = shared_split_dim(key)
        gathered = torch.cat([state[key] for _, _, state in shared_states], dim=dim)
        for peft_name, tensor in convert_shared_key(key, gathered, tp_size).items():
            if peft_name in out:
                raise RuntimeError(f"duplicate converted key: {peft_name}")
            out[peft_name] = tensor

    for tp_rank, global_rank, path in files:
        state = load_state(path)
        for key, tensor in state.items():
            m = LOCAL_EXPERT_RE.fullmatch(key)
            if not m:
                continue
            local_expert_id = int(m.group(2))
            global_expert_id = global_rank * local_experts_per_rank + local_expert_id
            if global_expert_id >= expected_experts:
                raise RuntimeError(
                    f"global expert id out of range: {global_expert_id} from {path.name}:{key}; "
                    f"expected_experts={expected_experts}"
                )
            for peft_name, converted in convert_local_expert_key(
                key,
                tensor,
                global_expert_id=global_expert_id,
            ).items():
                if peft_name in out:
                    raise RuntimeError(f"duplicate converted expert key: {peft_name}")
                out[peft_name] = converted

    copy_metadata(args.input_path, args.output_path)
    validate_shapes(out, load_base_shapes(args.base_model_path))

    from safetensors.torch import save_file

    adapter_file = args.output_path / "adapter_model.safetensors"
    save_file(out, str(adapter_file))
    summary = {
        "input_path": str(args.input_path),
        "output_path": str(args.output_path),
        "tp_size": tp_size,
        "expert_parallel_size": expert_parallel_size,
        "local_experts_per_rank": local_experts_per_rank,
        "expected_experts": expected_experts,
        "shared_tp_group": [
            {"tp_rank": tp_rank, "global_rank": global_rank}
            for tp_rank, global_rank, _ in shared_group
        ],
        "converted_tensor_count": len(out),
        "adapter_file": str(adapter_file),
    }
    (args.output_path / "megatron_to_peft_summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2))
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
GLM5_REF_convert_megatron_lora_to_peft_py
  cat > "${scripts_dir}/validate_glm51_fp8_checkpoint.py" <<'GLM5_REF_validate_glm51_fp8_checkpoint_py'
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from safetensors import safe_open


LAYER78_MODULES_TO_KEEP_UNQUANTIZED = {
    "model.layers.78.eh_proj",
    "model.layers.78.enorm",
    "model.layers.78.hnorm",
    "model.layers.78.input_layernorm",
    "model.layers.78.mlp.gate",
    "model.layers.78.mlp.gate.e_score_correction_bias",
    "model.layers.78.post_attention_layernorm",
    "model.layers.78.self_attn.indexer.k_norm",
    "model.layers.78.self_attn.indexer.k_norm.bias",
    "model.layers.78.self_attn.indexers_proj",
    "model.layers.78.self_attn.kv_a_layernorm",
    "model.layers.78.self_attn.q_a_layernorm",
    "model.layers.78.shared_head.norm",
}

LAYER78_UNQUANTIZED_WEIGHT_EXCEPTIONS = {
    "model.layers.78.self_attn.indexer.weights_proj",
}

LAYER78_MODULES_THAT_MUST_BE_QUANTIZED = {
    "model.layers.78.self_attn.kv_a_proj_with_mqa",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--require-mtp", action="store_true")
    parser.add_argument("--scan-shards", action="store_true", default=True)
    return parser.parse_args()


def module_name_from_weight_key(key: str) -> str:
    suffixes = (
        ".weight_scale_inv",
        ".weight",
        ".bias",
        ".e_score_correction_bias",
    )
    for suffix in suffixes:
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return key


def main() -> None:
    args = parse_args()
    model_path = args.model_path
    index_path = model_path / "model.safetensors.index.json"
    config_path = model_path / "config.json"
    if not index_path.is_file():
        raise RuntimeError(f"missing index: {index_path}")
    if not config_path.is_file():
        raise RuntimeError(f"missing config: {config_path}")

    index_payload = json.loads(index_path.read_text())
    config_payload = json.loads(config_path.read_text())
    weight_map: dict[str, str] = index_payload["weight_map"]
    n_routed_experts = int(
        config_payload.get("n_routed_experts") or config_payload.get("num_experts")
    )

    expert_re = re.compile(r"\.mlp\.experts\.(\d+)\.")
    overrange_index_keys = [
        key
        for key in weight_map
        if (match := expert_re.search(key)) and int(match.group(1)) >= n_routed_experts
    ]
    if overrange_index_keys:
        raise RuntimeError(f"overrange expert keys remain in index: {overrange_index_keys[:5]}")

    fix_keys = {
        key
        for key, shard_name in weight_map.items()
        if shard_name.startswith("model-experts-fix-layer-")
    }
    if not fix_keys:
        raise RuntimeError("missing repaired model-experts-fix-layer shards in index")

    stale_file_keys: list[dict[str, object]] = []
    if args.scan_shards:
        for shard_path in sorted(model_path.glob("model-*-of-*.safetensors")):
            with safe_open(str(shard_path), framework="pt", device="cpu") as reader:
                keys = list(reader.keys())
            bad_keys = [
                key
                for key in keys
                if key in fix_keys
                or ((match := expert_re.search(key)) and int(match.group(1)) >= n_routed_experts)
            ]
            if bad_keys:
                stale_file_keys.append(
                    {
                        "shard": shard_path.name,
                        "bad_key_count": len(bad_keys),
                        "sample": bad_keys[:5],
                    }
                )
        if stale_file_keys:
            raise RuntimeError(f"stale malformed MoE keys remain in main shards: {stale_file_keys}")

    layer78_keys = sorted(key for key in weight_map if key.startswith("model.layers.78."))
    if args.require_mtp and not layer78_keys:
        raise RuntimeError("MTP required but no model.layers.78.* keys were found")
    if layer78_keys:
        layer78_scale_keys = {key for key in layer78_keys if key.endswith(".weight_scale_inv")}
        for key in layer78_keys:
            if not key.endswith(".weight") or key.endswith(".weight_scale_inv"):
                continue
            module_name = module_name_from_weight_key(key)
            if module_name in LAYER78_MODULES_TO_KEEP_UNQUANTIZED:
                continue
            if module_name in LAYER78_UNQUANTIZED_WEIGHT_EXCEPTIONS:
                continue
            scale_key = f"{key}_scale_inv"
            if scale_key not in layer78_scale_keys:
                raise RuntimeError(f"quantized MTP weight is missing scale_inv: {key}")

        quant_cfg = config_payload.get("quantization_config", {})
        modules_to_not_convert = set(quant_cfg.get("modules_to_not_convert", []))
        missing_skip = sorted(LAYER78_MODULES_TO_KEEP_UNQUANTIZED - modules_to_not_convert)
        wrong_skip = sorted(LAYER78_MODULES_THAT_MUST_BE_QUANTIZED & modules_to_not_convert)
        if missing_skip:
            raise RuntimeError(f"MTP modules_to_not_convert missing entries: {missing_skip}")
        if wrong_skip:
            raise RuntimeError(f"MTP modules_to_not_convert has quantized modules: {wrong_skip}")

    shard_names = set(weight_map.values())
    missing_shards = sorted(name for name in shard_names if not (model_path / name).is_file())
    if missing_shards:
        raise RuntimeError(f"index points to missing shards: {missing_shards[:8]}")

    print(
        json.dumps(
            {
                "ok": True,
                "model_path": str(model_path),
                "index_key_count": len(weight_map),
                "shard_count": len(shard_names),
                "n_routed_experts": n_routed_experts,
                "repaired_moe_key_count": len(fix_keys),
                "mtp_key_count": len(layer78_keys),
                "mtp_required": bool(args.require_mtp),
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
GLM5_REF_validate_glm51_fp8_checkpoint_py
  cat > "${scripts_dir}/serve_glm51_fp8.sh" <<'GLM5_REF_serve_glm51_fp8_sh'
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
GLM5_REF_serve_glm51_fp8_sh
  cat > "${scripts_dir}/serve_glm51_fp8_compile_mtp.sh" <<'GLM5_REF_serve_glm51_fp8_compile_mtp_sh'
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "serve_glm51_fp8_compile_mtp.sh is a compatibility wrapper; use serve_glm51_fp8.sh. Forwarding..." >&2
exec "$SCRIPT_DIR/serve_glm51_fp8.sh" "$@"
GLM5_REF_serve_glm51_fp8_compile_mtp_sh
  cat > "${scripts_dir}/scrub_fp8_export_duplicate_moe_keys.py" <<'GLM5_REF_scrub_fp8_export_duplicate_moe_keys_py'
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--export-dir", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    export_dir = Path(args.export_dir)
    index_path = export_dir / "model.safetensors.index.json"
    config_path = export_dir / "config.json"
    if not index_path.is_file():
        raise RuntimeError(f"missing index: {index_path}")
    if not config_path.is_file():
        raise RuntimeError(f"missing config: {config_path}")
    index_payload = json.loads(index_path.read_text())
    weight_map: dict[str, str] = index_payload["weight_map"]
    config_payload = json.loads(config_path.read_text())
    n_routed_experts = int(
        config_payload.get("n_routed_experts") or config_payload["num_experts"]
    )
    expert_key_re = re.compile(r"\.mlp\.experts\.(\d+)\.")

    def is_overrange_expert_key(key: str) -> bool:
        match = expert_key_re.search(key)
        return match is not None and int(match.group(1)) >= n_routed_experts

    overrange_index_keys = [key for key in weight_map if is_overrange_expert_key(key)]
    for key in overrange_index_keys:
        del weight_map[key]

    fix_keys = {
        key
        for key, shard_name in weight_map.items()
        if shard_name.startswith("model-experts-fix-layer-")
    }
    if not fix_keys:
        raise RuntimeError("no model-experts-fix shards found in weight_map")

    scrubbed = []
    for shard_path in sorted(export_dir.glob("model-*-of-00003.safetensors")):
        with safe_open(str(shard_path), framework="pt", device="cpu") as reader:
            file_keys = list(reader.keys())
            remove_keys = [
                key
                for key in file_keys
                if key in fix_keys or is_overrange_expert_key(key)
            ]
            if not remove_keys:
                continue

            keep_tensors = {
                key: reader.get_tensor(key)
                for key in file_keys
                if key not in remove_keys
            }

        backup_path = shard_path.with_suffix(shard_path.suffix + ".bak")
        tmp_path = shard_path.with_suffix(shard_path.suffix + ".tmp")
        if backup_path.exists():
            raise RuntimeError(f"backup already exists, refusing to overwrite: {backup_path}")
        if tmp_path.exists():
            raise RuntimeError(f"tmp already exists, refusing to overwrite: {tmp_path}")

        save_file(keep_tensors, str(tmp_path), metadata={"format": "pt"})
        shard_path.rename(backup_path)
        tmp_path.rename(shard_path)
        scrubbed.append(
            {
                "shard": shard_path.name,
                "removed_key_count": len(remove_keys),
                "removed_overrange_count": sum(
                    1 for key in remove_keys if is_overrange_expert_key(key)
                ),
                "removed_redirected_count": sum(1 for key in remove_keys if key in fix_keys),
                "backup": backup_path.name,
            }
        )

    total_size = sum(path.stat().st_size for path in export_dir.glob("*.safetensors"))
    index_payload.setdefault("metadata", {})["total_size"] = total_size
    index_payload["metadata"]["scrubbed_overrange_expert_key_count"] = len(overrange_index_keys)
    index_payload["metadata"]["scrubbed_n_routed_experts"] = n_routed_experts
    index_path.write_text(json.dumps(index_payload, ensure_ascii=True, indent=2))

    print(
        json.dumps(
            {
                "scrubbed_shards": scrubbed,
                "scrubbed_count": len(scrubbed),
                "fix_key_count": len(fix_keys),
                "removed_index_overrange_key_count": len(overrange_index_keys),
                "n_routed_experts": n_routed_experts,
                "total_safetensors_bytes": total_size,
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
GLM5_REF_scrub_fp8_export_duplicate_moe_keys_py
  cat > "${scripts_dir}/finegrained_fp8_quantize_glm5_ray.py" <<'GLM5_REF_finegrained_fp8_quantize_glm5_ray_py'
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from pathlib import Path

import ray


@ray.remote
class NodeProcActor:
    def __init__(self) -> None:
        self.proc: subprocess.Popen[str] | None = None
        self.log_path: str | None = None

    def node_ip(self) -> str:
        import ray

        return ray.util.get_node_ip_address()

    def probe_imports(self, python: str, env: dict[str, str]) -> dict[str, object]:
        cmd = [
            python,
            "-c",
            (
                "import importlib.util, json, os;"
                "mods=['accelerate','transformers','triton'];"
                "print(json.dumps({"
                "'python': os.sys.executable,"
                "'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),"
                "'mods': {m: bool(importlib.util.find_spec(m)) for m in mods}"
                "}, ensure_ascii=True))"
            ),
        ]
        out = subprocess.run(
            cmd,
            env={**os.environ, **env},
            text=True,
            capture_output=True,
            check=False,
        )
        return {"returncode": out.returncode, "stdout": out.stdout, "stderr": out.stderr}

    def start(self, cmd: str, env: dict[str, str], cwd: str, log_path: str) -> dict[str, object]:
        if self.proc is not None and self.proc.poll() is None:
            raise RuntimeError("process already running")
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "a", encoding="utf-8")
        merged_env = dict(os.environ)
        merged_env.update(env)
        self.proc = subprocess.Popen(
            ["bash", "-lc", cmd],
            cwd=cwd,
            env=merged_env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.log_path = log_path
        return {"pid": int(self.proc.pid), "log_path": log_path}

    def poll(self) -> dict[str, object]:
        if self.proc is None:
            return {"running": False, "started": False}
        code = self.proc.poll()
        return {
            "running": code is None,
            "started": True,
            "returncode": None if code is None else int(code),
            "pid": int(self.proc.pid),
            "log_path": self.log_path,
        }

    def read_from(self, start: int) -> dict[str, object]:
        if self.log_path is None:
            return {"start": start, "end": start, "text": ""}
        path = Path(self.log_path)
        if not path.exists():
            return {"start": start, "end": start, "text": ""}
        data = path.read_text(encoding="utf-8", errors="replace")
        end = len(data)
        if start >= end:
            return {"start": start, "end": end, "text": ""}
        return {"start": start, "end": end, "text": data[start:end]}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ray-address", default="10.11.26.106:6379")
    p.add_argument("--ray-namespace", default="tinker")
    p.add_argument("--node-ip", required=True)
    p.add_argument("--base-model-path", required=True)
    p.add_argument("--export-dir", required=True)
    p.add_argument("--python", default="/usr/bin/python3.12")
    p.add_argument(
        "--quant-script",
        default=str(Path(__file__).resolve().parent / "quantize_glm5_finegrained_fp8.py"),
    )
    p.add_argument("--workdir", default="/vePFS-Mindverse/share/code/tinker-server-aliyun")
    p.add_argument(
        "--remote-log-dir",
        default="/vePFS-Mindverse/share/tmp/glm5_finegrained_fp8_quant_logs",
    )
    return p.parse_args()


def _make_env() -> dict[str, str]:
    return {
        "HF_HOME": "/vePFS-Mindverse/share/huggingface",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": (
            "/vePFS-Mindverse/share/code/mint-runtime-py31213-rl-v4/site-packages:"
            "/vePFS-Mindverse/share/code/tinker-server-aliyun:"
            "/vePFS-Mindverse/share/huggingface/modules:"
            "/vePFS-Mindverse/share/huggingface/modules/transformers_modules"
        ),
    }


def main() -> None:
    args = parse_args()
    ray.init(address=args.ray_address, namespace=args.ray_namespace, ignore_reinit_error=True)
    actor = NodeProcActor.options(
        num_gpus=8,
        num_cpus=8,
        resources={f"node:{args.node_ip}": 0.001},
    ).remote()
    actual_ip = ray.get(actor.node_ip.remote())
    if actual_ip != args.node_ip:
        raise RuntimeError(f"node pin mismatch: wanted {args.node_ip}, got {actual_ip}")

    env = _make_env()
    probe = ray.get(actor.probe_imports.remote(args.python, env))
    if probe["returncode"] != 0:
        raise RuntimeError(f"import probe failed: {probe}")
    payload = json.loads(str(probe["stdout"]).strip())
    missing = [name for name, ok in payload["mods"].items() if not ok]
    if missing:
        raise RuntimeError(f"missing imports on {args.node_ip}: {missing} payload={payload}")
    print(json.dumps({"phase": "probe_ok", "node_ip": args.node_ip, **payload}, ensure_ascii=True), flush=True)

    timestamp = int(time.time())
    log_path = str(Path(args.remote_log_dir) / f"glm5_finegrained_fp8_quant_node_{timestamp}.log")
    cmd = (
        f"{shlex.quote(args.python)} {shlex.quote(args.quant_script)} "
        f"--base-model-path {shlex.quote(args.base_model_path)} "
        f"--export-dir {shlex.quote(args.export_dir)} "
        "--trust-remote-code"
    )
    start = ray.get(actor.start.remote(cmd, env, args.workdir, log_path))
    print(json.dumps({"phase": "started", "node_ip": args.node_ip, **start}, ensure_ascii=True), flush=True)

    offset = 0
    while True:
        status = ray.get(actor.poll.remote())
        chunk = ray.get(actor.read_from.remote(offset))
        offset = int(chunk["end"])
        text = str(chunk["text"])
        if text:
            print(json.dumps({"phase": "log", "node_ip": args.node_ip, "text": text[-8000:]}, ensure_ascii=True), flush=True)
        print(json.dumps({"phase": "status", "node_ip": args.node_ip, **status}, ensure_ascii=True), flush=True)
        if not status.get("running", False):
            break
        time.sleep(15)

    status = ray.get(actor.poll.remote())
    if status.get("returncode") != 0:
        raise RuntimeError(f"finegrained fp8 quantization failed: {status}")
    print(json.dumps({"phase": "done", "node_ip": args.node_ip, **status}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
GLM5_REF_finegrained_fp8_quantize_glm5_ray_py
  cat > "${scripts_dir}/merge_glm5_lora_into_base_ray.py" <<'GLM5_REF_merge_glm5_lora_into_base_ray_py'
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import ray
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ray-address", default="10.11.26.106:6379")
    parser.add_argument("--ray-namespace", default="tinker")
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--lora-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--node-ips-json", default=None)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--limit-shards", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def map_lora_key_to_base_key(lora_key: str) -> str | None:
    if not lora_key.endswith(".lora_A.weight"):
        return None
    prefix = "base_model.model."
    if not lora_key.startswith(prefix):
        return None
    base_key = lora_key.removeprefix(prefix)
    base_key = base_key.replace(".shared_expert.", ".shared_experts.")
    base_key = base_key.replace(".lora_A.weight", ".weight")
    return base_key


def build_merge_plan(
    *,
    base_model_path: Path,
    lora_path: Path,
) -> tuple[dict[str, tuple[str, str]], dict[str, str], float]:
    from safetensors import safe_open

    config = json.loads((lora_path / "adapter_config.json").read_text())
    rank = int(config["r"])
    alpha = float(config["lora_alpha"])
    scaling = alpha / rank

    index = json.loads((base_model_path / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    adapter_file = lora_path / "adapter_model.safetensors"
    with safe_open(str(adapter_file), framework="pt", device="cpu") as f:
        keys = list(f.keys())

    base_to_pair: dict[str, tuple[str, str]] = {}
    missing: list[dict[str, str]] = []
    for key in keys:
        base_key = map_lora_key_to_base_key(key)
        if base_key is None:
            continue
        b_key = key.replace(".lora_A.weight", ".lora_B.weight")
        if b_key not in keys:
            missing.append({"lora_A": key, "reason": "missing_lora_B"})
            continue
        if base_key not in weight_map:
            missing.append({"lora_A": key, "reason": f"missing_base:{base_key}"})
            continue
        base_to_pair[base_key] = (key, b_key)

    if missing:
        raise RuntimeError(
            f"merge plan has {len(missing)} unmapped LoRA entries; first={missing[:5]}"
        )
    if not base_to_pair:
        raise RuntimeError("merge plan found no mergeable LoRA pairs")
    return base_to_pair, weight_map, scaling


def shard_worker(
    *,
    shard_names: list[str],
    base_model_path: str,
    lora_adapter_file: str,
    output_path: str,
    base_to_pair: dict[str, tuple[str, str]],
    scaling: float,
) -> dict[str, object]:
    from safetensors import safe_open
    from safetensors.torch import save_file

    base_dir = Path(base_model_path)
    out_dir = Path(output_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    modified_shards = 0
    modified_params = 0
    shard_summaries = []

    with safe_open(lora_adapter_file, framework="pt", device="cpu") as lora_file:
        for shard_name in shard_names:
            shard_path = base_dir / shard_name
            out_path = out_dir / shard_name
            if out_path.exists():
                out_path.unlink()

            changed = 0
            updated: dict[str, torch.Tensor] = {}
            with safe_open(str(shard_path), framework="pt", device=device) as shard_file:
                for key in shard_file.keys():
                    weight = shard_file.get_tensor(key)
                    pair = base_to_pair.get(key)
                    if pair is not None:
                        a_key, b_key = pair
                        lora_a = lora_file.get_tensor(a_key).to(device=device, dtype=torch.float32)
                        lora_b = lora_file.get_tensor(b_key).to(device=device, dtype=torch.float32)
                        merged = weight.to(torch.float32).add_(torch.matmul(lora_b, lora_a), alpha=scaling)
                        weight = merged.to(weight.dtype)
                        del lora_a, lora_b, merged
                        changed += 1
                    updated[key] = weight.cpu()
                    del weight
            save_file(updated, str(out_path))
            del updated
            if device == "cuda":
                torch.cuda.empty_cache()
            if changed:
                modified_shards += 1
                modified_params += changed
            shard_summaries.append(
                {
                    "shard": shard_name,
                    "changed_params": changed,
                }
            )

    return {
        "worker_device": device,
        "assigned_shards": shard_names,
        "modified_shards": modified_shards,
        "modified_params": modified_params,
        "shards": shard_summaries,
    }


def main() -> None:
    args = parse_args()
    ray.init(address=args.ray_address, namespace=args.ray_namespace, ignore_reinit_error=True)

    base_to_pair, weight_map, scaling = build_merge_plan(
        base_model_path=args.base_model_path,
        lora_path=args.lora_path,
    )
    affected_by_shard: dict[str, list[str]] = defaultdict(list)
    for base_key in sorted(base_to_pair):
        affected_by_shard[weight_map[base_key]].append(base_key)

    shard_names = sorted(
        p.name for p in args.base_model_path.glob("model-*.safetensors")
    )
    if not shard_names:
        raise RuntimeError(f"no model shards under {args.base_model_path}")
    if args.limit_shards is not None:
        if args.limit_shards <= 0:
            raise RuntimeError("--limit-shards must be positive")
        shard_names = shard_names[: args.limit_shards]

    out_dir = args.output_path
    if out_dir.exists():
        if not args.force:
            raise RuntimeError(f"output path already exists: {out_dir}")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-create metadata/config symlinks or copies. Shards are handled later.
    for src in args.base_model_path.iterdir():
        if src.name.startswith("model-") and src.name.endswith(".safetensors"):
            continue
        dst = out_dir / src.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            dst.symlink_to(src)
        except OSError:
            if src.is_file():
                dst.write_bytes(src.read_bytes())

    if args.node_ips_json:
        node_ips = json.loads(args.node_ips_json)
        if not isinstance(node_ips, list) or not all(isinstance(x, str) for x in node_ips):
            raise TypeError("--node-ips-json must decode to list[str]")
    else:
        node_ips = []

    worker_count = max(1, min(args.num_workers, len(shard_names)))
    shard_buckets = [[] for _ in range(worker_count)]
    for idx, shard_name in enumerate(shard_names):
        shard_buckets[idx % worker_count].append(shard_name)

    remote_fn = ray.remote(num_gpus=1)(shard_worker)
    refs = []
    for idx, bucket in enumerate(shard_buckets):
        if not bucket:
            continue
        opts = {}
        if node_ips:
            node_ip = node_ips[idx % len(node_ips)]
            opts["resources"] = {f"node:{node_ip}": 0.001}
        refs.append(
            remote_fn.options(**opts).remote(
                shard_names=bucket,
                base_model_path=str(args.base_model_path),
                lora_adapter_file=str(args.lora_path / "adapter_model.safetensors"),
                output_path=str(out_dir),
                base_to_pair=base_to_pair,
                scaling=scaling,
            )
        )

    worker_results = ray.get(refs)

    changed_shards = {item["shard"] for result in worker_results for item in result["shards"] if item["changed_params"] > 0}
    untouched = sorted(set(shard_names) - changed_shards)
    for shard_name in untouched:
        src = args.base_model_path / shard_name
        dst = out_dir / shard_name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)

    summary = {
        "base_model_path": str(args.base_model_path),
        "lora_path": str(args.lora_path),
        "output_path": str(out_dir),
        "scaling": scaling,
        "mapped_param_count": len(base_to_pair),
        "affected_shard_count": len(affected_by_shard),
        "total_shard_count": len(shard_names),
        "changed_shards": len(changed_shards),
        "unchanged_shards": len(untouched),
        "worker_results": worker_results,
    }
    (out_dir / "merge_summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2))
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
GLM5_REF_merge_glm5_lora_into_base_ray_py
  cat > "${scripts_dir}/preflight_glm51_deploy.sh" <<'GLM5_REF_preflight_glm51_deploy_sh'
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
GLM5_REF_preflight_glm51_deploy_sh
  cat > "${scripts_dir}/test_glm51_endpoint_matrix.py" <<'GLM5_REF_test_glm51_endpoint_matrix_py'
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


CHAT_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "city name"}},
            "required": ["city"],
        },
    },
}

RESPONSES_TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": "Get weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "city name"}},
        "required": ["city"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible base URL, e.g. http://host:7893/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, default=Path("/tmp/glm51_endpoint_matrix.json"))
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def post(base_url: str, path: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            status = resp.status
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        status = exc.code
    except Exception as exc:
        return {
            "status": None,
            "elapsed_s": round(time.monotonic() - start, 3),
            "ok": False,
            "error": repr(exc),
        }

    try:
        data = json.loads(raw)
    except Exception:
        data = None
    return {
        "status": status,
        "elapsed_s": round(time.monotonic() - start, 3),
        "ok": 200 <= status < 300,
        "raw": raw[:4000],
        "json": data,
    }


def summarize_chat(data: dict[str, object]) -> dict[str, object]:
    try:
        choice = data["choices"][0]  # type: ignore[index]
        msg = choice["message"]
    except Exception:
        return {"parsed": False}
    return {
        "parsed": True,
        "finish_reason": choice.get("finish_reason"),
        "content_preview": (msg.get("content") or "")[:200],
        "reasoning_preview": (msg.get("reasoning_content") or "")[:200],
        "tool_call_count": len(msg.get("tool_calls") or []),
        "tool_calls": msg.get("tool_calls"),
    }


def summarize_completion(data: dict[str, object]) -> dict[str, object]:
    try:
        choice = data["choices"][0]  # type: ignore[index]
    except Exception:
        return {"parsed": False}
    return {
        "parsed": True,
        "finish_reason": choice.get("finish_reason"),
        "text_preview": (choice.get("text") or "")[:240],
    }


def summarize_responses(data: dict[str, object]) -> dict[str, object]:
    output = data.get("output") if isinstance(data, dict) else None
    calls = []
    text_parts = []
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") in {"function_call", "tool_call"}:
                calls.append(item)
            for content_item in item.get("content") or []:
                if isinstance(content_item, dict) and "text" in content_item:
                    text_parts.append(content_item["text"])
    return {
        "parsed": isinstance(data, dict),
        "status": data.get("status") if isinstance(data, dict) else None,
        "output_types": [item.get("type") for item in output if isinstance(item, dict)]
        if isinstance(output, list)
        else None,
        "text_preview": "".join(text_parts)[:240],
        "tool_call_count": len(calls),
        "tool_calls": calls,
    }


def main() -> None:
    args = parse_args()
    prompt = "Do not show reasoning. Reply with exactly this word: hello"
    tool_prompt = "Use the get_weather tool to look up weather for Beijing. Do not answer directly."

    tests = [
        ("/v1/completions no tools", "/completions", {"model": args.model, "prompt": prompt, "temperature": 0, "max_tokens": 32}, summarize_completion),
        ("/v1/chat/completions no tools", "/chat/completions", {"model": args.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 512}, summarize_chat),
        ("/v1/responses no tools", "/responses", {"model": args.model, "input": prompt, "temperature": 0, "max_output_tokens": 512}, summarize_responses),
        ("/v1/chat/completions tools auto", "/chat/completions", {"model": args.model, "messages": [{"role": "user", "content": tool_prompt}], "tools": [CHAT_TOOL], "tool_choice": "auto", "temperature": 0, "max_tokens": 128}, summarize_chat),
        ("/v1/responses tools auto", "/responses", {"model": args.model, "input": tool_prompt, "tools": [RESPONSES_TOOL], "tool_choice": "auto", "temperature": 0, "max_output_tokens": 128}, summarize_responses),
        ("/v1/chat/completions tools required", "/chat/completions", {"model": args.model, "messages": [{"role": "user", "content": tool_prompt}], "tools": [CHAT_TOOL], "tool_choice": "required", "temperature": 0, "max_tokens": 128}, summarize_chat),
        ("/v1/responses tools required", "/responses", {"model": args.model, "input": tool_prompt, "tools": [RESPONSES_TOOL], "tool_choice": "required", "temperature": 0, "max_output_tokens": 128}, summarize_responses),
        ("/v1/chat/completions forced function", "/chat/completions", {"model": args.model, "messages": [{"role": "user", "content": tool_prompt}], "tools": [CHAT_TOOL], "tool_choice": {"type": "function", "function": {"name": "get_weather"}}, "temperature": 0, "max_tokens": 128}, summarize_chat),
        ("/v1/responses forced function", "/responses", {"model": args.model, "input": tool_prompt, "tools": [RESPONSES_TOOL], "tool_choice": {"type": "function", "name": "get_weather"}, "temperature": 0, "max_output_tokens": 128}, summarize_responses),
    ]

    results = []
    for name, path, payload, summarizer in tests:
        result = post(args.base_url, path, payload, args.timeout)
        summary: dict[str, object] = {}
        if result.get("json") is not None:
            summary = summarizer(result["json"])  # type: ignore[arg-type]
        row = {
            "name": name,
            "status": result.get("status"),
            "ok": result.get("ok"),
            "elapsed_s": result.get("elapsed_s"),
            "summary": summary,
            "error": result.get("error"),
        }
        print(json.dumps(row, ensure_ascii=False), flush=True)
        results.append({"name": name, "path": path, "payload": payload, "result": result, "summary": summary})

    args.out.write_text(
        json.dumps(
            {
                "base_url": args.base_url,
                "model": args.model,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"RESULT_JSON={args.out}")


if __name__ == "__main__":
    main()
GLM5_REF_test_glm51_endpoint_matrix_py
  chmod +x "${scripts_dir}"/*.sh "${scripts_dir}"/*.py
}

write_active_model_env() {
  mkdir -p "$STATE_DIR"
  {
    printf 'MODEL_DIR=%q\n' "$MODEL_DIR"
    printf 'MODEL_DIR_HOST=%q\n' "$MODEL_DIR_HOST"
  } > "$ACTIVE_MODEL_ENV"
}

load_active_model_env() {
  if [ -s "$ACTIVE_MODEL_ENV" ]; then
    # shellcheck disable=SC1090
    source "$ACTIVE_MODEL_ENV"
    MODEL_DIR_HOST="$(host_path_for_container_path "$MODEL_DIR")"
    export MODEL_DIR MODEL_DIR_HOST
    log "loaded active model from $ACTIVE_MODEL_ENV: MODEL_DIR=$MODEL_DIR"
  fi
}

tinker_merge_quant_enabled() {
  case "${RUN_TINKER_MERGE_QUANT:-auto}" in
    1|true|yes|on) return 0 ;;
    0|false|no|off) return 1 ;;
    auto)
      case "${TINKER_URL:-}" in
        tinker://*) return 0 ;;
        *) return 1 ;;
      esac
      ;;
    *) fail "RUN_TINKER_MERGE_QUANT must be auto, 1, or 0; got: $RUN_TINKER_MERGE_QUANT" ;;
  esac
}

########################################
# 2. 基础检查
########################################

host_tmp_writable() {
  local test_file="/tmp/.glm51-rw-test.$$"
  if touch "$test_file" >/dev/null 2>&1; then
    rm -f "$test_file" >/dev/null 2>&1 || true
    return 0
  fi
  return 1
}

print_docker_systemd_diagnostics() {
  log "docker/containerd systemd diagnostics"
  sudo systemctl status docker.service --no-pager -l || true
  sudo systemctl status containerd.service --no-pager -l || true
  sudo journalctl -u docker.service -n 80 --no-pager || true
  sudo journalctl -u containerd.service -n 80 --no-pager || true
}

revert_docker_tmpdir_dropins() {
  sudo rm -f /etc/systemd/system/docker.service.d/10-glm51-tmpdir.conf || true
  sudo rm -f /etc/systemd/system/containerd.service.d/10-glm51-tmpdir.conf || true
  if command -v systemctl >/dev/null 2>&1; then
    sudo systemctl daemon-reload || true
  fi
}

configure_host_tmpdir() {
  log "configure host tmpdir fallback: $HOST_TMPDIR"
  sudo mkdir -p "$HOST_TMPDIR"
  sudo chmod 1777 "$HOST_TMPDIR"

  revert_docker_tmpdir_dropins

  if ! host_tmp_writable; then
    log "/tmp not writable; bind mounting $HOST_TMPDIR to /tmp"
    sudo mount --bind "$HOST_TMPDIR" /tmp
    sudo chmod 1777 /tmp
  fi

  host_tmp_writable || fail "host /tmp is not writable even after binding HOST_TMPDIR=$HOST_TMPDIR"

  if command -v systemctl >/dev/null 2>&1; then
    if ! sudo docker info >/dev/null 2>&1; then
      log "docker info failed; attempting docker/containerd recovery restart"
      sudo systemctl restart containerd || true
      sudo systemctl restart docker || true
      if ! sudo docker info >/dev/null 2>&1; then
        print_docker_systemd_diagnostics
        fail "docker unavailable after recovery restart"
      fi
    fi
  fi
}

check_prereqs() {
  log "check prerequisites"
  require_cmd sudo
  require_cmd docker
  require_cmd curl
  require_cmd tmux

  : "${BASE_IMAGE:?BASE_IMAGE empty}"
  : "${SGLANG_IMAGE:?SGLANG_IMAGE empty}"
  : "${SGLANG_CONTAINER:?SGLANG_CONTAINER empty}"
  : "${HOST_WORKDIR:?HOST_WORKDIR empty}"
  : "${CONTAINER_WORKDIR:?CONTAINER_WORKDIR empty}"
  : "${HOST_TMPDIR:?HOST_TMPDIR empty}"
  : "${MODEL_ID:?MODEL_ID empty}"
  : "${MODEL_REVISION:?MODEL_REVISION empty}"
  : "${MODEL_DIR:?MODEL_DIR empty}"
  : "${SERVED_MODEL_NAME:?SERVED_MODEL_NAME empty}"
  : "${RUN_TINKER_MERGE_QUANT:?RUN_TINKER_MERGE_QUANT empty}"
  case "$RUN_TINKER_MERGE_QUANT" in
    auto|1|0|true|false|yes|no|on|off) ;;
    *) fail "RUN_TINKER_MERGE_QUANT must be auto, 1, or 0; got: $RUN_TINKER_MERGE_QUANT" ;;
  esac
  if tinker_merge_quant_enabled; then
    : "${GPU_LEASE_BASE_URL:?GPU_LEASE_BASE_URL empty}"
    : "${GPU_LEASE_API_KEY:?GPU_LEASE_API_KEY empty}"
    : "${TINKER_URL:?TINKER_URL empty}"
    : "${LORA_DIR:?LORA_DIR empty}"
    : "${MERGED_MODEL_DIR:?MERGED_MODEL_DIR empty}"
    : "${QUANT_MODEL_DIR:?QUANT_MODEL_DIR empty}"
    : "${QUANT_SCHEME:?QUANT_SCHEME empty}"
    : "${GLM5_REFERENCE_SCRIPTS_DIR:?GLM5_REFERENCE_SCRIPTS_DIR empty}"
    : "${GLM5_MERGE_QUANT_GPUS:?GLM5_MERGE_QUANT_GPUS empty}"
    : "${GLM5_MERGE_QUANT_WORKERS_PER_GPU:?GLM5_MERGE_QUANT_WORKERS_PER_GPU empty}"
    : "${GLM5_LOCAL_QUANT_ROOT:?GLM5_LOCAL_QUANT_ROOT empty}"
    case "$TINKER_URL" in
      tinker://*) ;;
      *"<这里填 "*|"" ) fail "Tinker merge/quant is enabled but TINKER_URL is still a placeholder; set TINKER_URL=tinker://.../weights/..." ;;
      *) fail "Tinker merge/quant is enabled and requires TINKER_URL=tinker://.../weights/...; got: $TINKER_URL" ;;
    esac
  fi
  : "${SGLANG_PORT:?SGLANG_PORT empty}"
  : "${SGLANG_SERVE_ARGS:?SGLANG_SERVE_ARGS empty}"
  : "${GENERATED_SCRIPT_DIR:?GENERATED_SCRIPT_DIR empty}"
  : "${SGLANG_EXTERNAL_MODEL_PACKAGE:?SGLANG_EXTERNAL_MODEL_PACKAGE empty}"
  : "${AITER_QUICK_REDUCE_QUANTIZATION:?AITER_QUICK_REDUCE_QUANTIZATION empty}"
  : "${SGLANG_AITER_FP8_PREFILL_ATTN:?SGLANG_AITER_FP8_PREFILL_ATTN empty}"

  : "${ATOM_REPO_URL:?ATOM_REPO_URL empty}"
  : "${ATOM_REF:?ATOM_REF empty}"
  : "${ATOM_REPO_ROOT:?ATOM_REPO_ROOT empty}"
  ATOM_REPO_HOST="${ATOM_REPO_HOST:-$ATOM_REPO_ROOT}"
  : "${ATOM_REPO_HOST:?ATOM_REPO_HOST empty}"
  : "${ATOM_REPO_CONTAINER:?ATOM_REPO_CONTAINER empty}"
  case ":${ATOM_PLUGIN_PYTHONPATH:-}:" in
    *":${ATOM_REPO_CONTAINER}:"*) ;;
    *) ATOM_PLUGIN_PYTHONPATH="${ATOM_PLUGIN_PYTHONPATH:+${ATOM_PLUGIN_PYTHONPATH}:}${ATOM_REPO_CONTAINER}" ;;
  esac
  [ -n "${ATOM_PLUGIN_PYTHONPATH:-}" ] || fail "ATOM_PLUGIN_PYTHONPATH is empty. For PR355 OOT plugin, set container PYTHONPATH to include SGLang and ATOM paths."
  export ATOM_REPO_HOST ATOM_PLUGIN_PYTHONPATH

  sudo mkdir -p "$GENERATED_SCRIPT_DIR"
  sudo chown "$(id -u):$(id -g)" "$GENERATED_SCRIPT_DIR" || true
  chmod 0755 "$GENERATED_SCRIPT_DIR" || true
  install_glm5_reference_scripts
  sudo mkdir -p "$HOST_WORKDIR" "$STATE_DIR" "$RUNTIME_DIR" "$MODEL_DIR_HOST" "$LORA_DIR_HOST" "$MERGED_MODEL_DIR_HOST" "$QUANT_MODEL_DIR_HOST" "$TMUX_TMPDIR"
  # Do not recursively chown HOST_WORKDIR: it contains hundreds of GB of model/cache data and can block startup.
  sudo chown "$(id -u):$(id -g)" "$HOST_WORKDIR" "$STATE_DIR" "$RUNTIME_DIR" "$MODEL_DIR_HOST" "$LORA_DIR_HOST" "$MERGED_MODEL_DIR_HOST" "$QUANT_MODEL_DIR_HOST" "$TMUX_TMPDIR" || true
  chmod 700 "$TMUX_TMPDIR" || true

  assert_dir_writable "$HOST_WORKDIR" "HOST_WORKDIR"
  configure_host_tmpdir
  ensure_space_or_cleanup "$HOST_WORKDIR" "HOST_WORKDIR" "$MIN_HOST_FREE_GB" 0

  log "env summary"
  printf 'BASE_IMAGE(tool)=%s
SGLANG_IMAGE=%s
SGLANG_CONTAINER=%s
HOST_WORKDIR=%s
CONTAINER_WORKDIR=%s
HOST_TMPDIR=%s
MODEL_ID=%s
MODEL_REVISION=%s
MODEL_DIR=%s
MODEL_DIR_HOST=%s
SERVED_MODEL_NAME=%s
RUN_TINKER_MERGE_QUANT=%s
GPU_LEASE_BASE_URL=%s
GPU_LEASE_API_KEY_SET=%s
TINKER_URL_SET=%s
LORA_DIR=%s
MERGED_MODEL_DIR=%s
QUANT_MODEL_DIR=%s
QUANT_SCHEME=%s
GLM5_REFERENCE_SCRIPTS_DIR=%s
GLM5_MERGE_QUANT_GPUS=%s
GLM5_MERGE_QUANT_WORKERS_PER_GPU=%s
GLM5_LOCAL_QUANT_ROOT=%s
SGLANG_PORT=%s
INSTALL_SYSTEMD_AUTOSTART=%s
AUTOSTART_SERVICE_NAME=%s
AUTOSTART_CHECK_INTERVAL_SECONDS=%s
AUTO_FREE_SPACE=%s
AUTO_DELETE_MODEL_IF_LOW_SPACE=%s
MIN_HOST_FREE_GB=%s
MIN_MODEL_DOWNLOAD_FREE_GB=%s
MIN_DOCKER_BUILD_FREE_GB=%s
HF_TOKEN_SET=%s
ATOM_REPO_URL=%s
ATOM_REF=%s
ATOM_REPO_ROOT=%s
ATOM_REPO_HOST=%s
ATOM_REPO_CONTAINER=%s
ATOM_PLUGIN_PYTHONPATH=%s
SGLANG_EXTERNAL_MODEL_PACKAGE=%s
AITER_QUICK_REDUCE_QUANTIZATION=%s
SGLANG_AITER_FP8_PREFILL_ATTN=%s
PATCH_SGLANG_QUARK_FUSED_QKV_A_PROJ=%s
' \
    "$BASE_IMAGE" "$SGLANG_IMAGE" "$SGLANG_CONTAINER" "$HOST_WORKDIR" "$CONTAINER_WORKDIR" "$HOST_TMPDIR" "$MODEL_ID" "$MODEL_REVISION" "$MODEL_DIR" "$MODEL_DIR_HOST" "$SERVED_MODEL_NAME" "$RUN_TINKER_MERGE_QUANT" "$GPU_LEASE_BASE_URL" "$([ -n "${GPU_LEASE_API_KEY:-}" ] && printf yes || printf no)" "$([ -n "${TINKER_URL:-}" ] && printf yes || printf no)" "$LORA_DIR" "$MERGED_MODEL_DIR" "$QUANT_MODEL_DIR" "$QUANT_SCHEME" "$GLM5_REFERENCE_SCRIPTS_DIR" "$GLM5_MERGE_QUANT_GPUS" "$GLM5_MERGE_QUANT_WORKERS_PER_GPU" "$GLM5_LOCAL_QUANT_ROOT" "$SGLANG_PORT" "$INSTALL_SYSTEMD_AUTOSTART" "$AUTOSTART_SERVICE_NAME" "$AUTOSTART_CHECK_INTERVAL_SECONDS" "$AUTO_FREE_SPACE" "$AUTO_DELETE_MODEL_IF_LOW_SPACE" "$MIN_HOST_FREE_GB" "$MIN_MODEL_DOWNLOAD_FREE_GB" "$MIN_DOCKER_BUILD_FREE_GB" "$([ -n "${HF_TOKEN:-}" ] && printf yes || printf no)" "$ATOM_REPO_URL" "$ATOM_REF" "$ATOM_REPO_ROOT" "$ATOM_REPO_HOST" "$ATOM_REPO_CONTAINER" "$ATOM_PLUGIN_PYTHONPATH" "$SGLANG_EXTERNAL_MODEL_PACKAGE" "$AITER_QUICK_REDUCE_QUANTIZATION" "$SGLANG_AITER_FP8_PREFILL_ATTN" "$PATCH_SGLANG_QUARK_FUSED_QKV_A_PROJ"
  printf 'SGLANG_SERVE_ARGS=
%s
' "$SGLANG_SERVE_ARGS"
  print_storage_diagnostics
  print_platform_diagnostics
}

########################################
# 3. Docker 缓存检查 / 轻量修复
########################################

docker_smoke_test_image() {
  local image="$1"
  sudo docker run --rm --entrypoint bash "$image" -lc 'echo ok' >"$RUNTIME_DIR/docker_smoke.out" 2>"$RUNTIME_DIR/docker_smoke.err"
}

clear_containerd_ingest_light() {
  # 清理半拉取临时内容，不删除完整镜像目录。
  if [ -d /data/containerd/root/io.containerd.content.v1.content/ingest ]; then
    sudo rm -rf /data/containerd/root/io.containerd.content.v1.content/ingest/* || true
  fi
  if [ -d /var/lib/containerd/io.containerd.content.v1.content/ingest ]; then
    sudo rm -rf /var/lib/containerd/io.containerd.content.v1.content/ingest/* || true
  fi
}

fix_docker_cache_light() {
  log "attempt light docker/containerd cache repair"
  sudo docker container prune -f || true
  sudo docker builder prune -f || true

  sudo systemctl stop docker || true
  sudo systemctl stop containerd || true
  clear_containerd_ingest_light
  sudo systemctl start containerd || true
  sudo systemctl start docker || true
}

fix_docker_snapshot_conflict_light() {
  log "[docker-cache] detected snapshot conflict during pull; attempting general light repair"
  sudo docker container prune -f || true
  sudo docker builder prune -f || true
  sudo docker image prune -f || true

  sudo systemctl stop docker || true
  sudo systemctl stop containerd || true
  clear_containerd_ingest_light

  sudo systemctl start containerd || true
  if command -v ctr >/dev/null 2>&1; then
    sudo ctr -n moby snapshots ls || true
    sudo ctr -n moby snapshots prune || true
    sudo ctr -n moby content prune references || true
  fi
  sudo systemctl start docker || true
}

pull_failed_with_snapshot_conflict() {
  local err_file="$1"
  grep -Eiq 'AlreadyExists: target snapshot|target snapshot .* already exists|snapshot.*already exists' "$err_file" 2>/dev/null
}

extract_containerd_target_snapshot_id() {
  local err_file="$1"
  grep -Eo 'target snapshot "sha256:[0-9a-f]{64}"' "$err_file" 2>/dev/null | \
    sed -nE 's/^target snapshot "(sha256:[0-9a-f]{64})"$/\1/p' | \
    head -n 1
}

fix_containerd_target_snapshot_light() {
  local snapshot_id="$1"
  if ! printf '%s' "$snapshot_id" | grep -Eq '^sha256:[0-9a-f]{64}$'; then
    log "[docker-cache] invalid target snapshot id; skip targeted repair: ${snapshot_id:-empty}"
    return 1
  fi
  if ! command -v ctr >/dev/null 2>&1; then
    log "[docker-cache] ctr missing; cannot run targeted snapshot repair"
    return 1
  fi

  log "[docker-cache] target snapshot diagnostics: $snapshot_id"
  sudo ctr -n moby snapshots info "$snapshot_id" || true
  sudo ctr -n moby snapshots ls | grep -F "$snapshot_id" || true
  sudo ctr -n moby content ls | grep -F "${snapshot_id#sha256:}" || true

  log "[docker-cache] attempting targeted snapshot rm: $snapshot_id"
  if sudo ctr -n moby snapshots rm "$snapshot_id"; then
    log "[docker-cache] targeted snapshot rm completed: $snapshot_id"
    return 0
  fi

  log "[docker-cache] targeted snapshot rm failed: $snapshot_id"
  log "[docker-cache] snapshot may still be referenced by a lease/container/child snapshot; manual containerd inspection required"
  return 1
}

cleanup_existing_containers_if_enabled() {
  if [ "$AUTO_KILL_EXISTING_CONTAINERS" != "1" ]; then
    log "skip killing existing containers because AUTO_KILL_EXISTING_CONTAINERS=$AUTO_KILL_EXISTING_CONTAINERS"
    return
  fi

  log "kill/remove existing Docker containers to free GPU/ports/overlay space"
  local containers
  containers="$(sudo docker ps -aq || true)"
  if [ -z "$containers" ]; then
    log "no existing Docker containers"
    return
  fi

  sudo docker ps -a --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}' || true
  sudo docker rm -f $containers || true
  sudo docker container prune -f || true
  log "existing containers removed"
}

ensure_base_image() {
  log "ensure base image works: $BASE_IMAGE"

  if docker_smoke_test_image "$BASE_IMAGE"; then
    log "base image smoke test OK"
    return
  fi

  log "base image smoke test failed"
  sed -n '1,120p' "$RUNTIME_DIR/docker_smoke.err" || true

  if [ "$AUTO_FIX_DOCKER_CACHE" = "1" ]; then
    fix_docker_cache_light
  fi

  log "pull base image"
  if ! sudo docker pull "$BASE_IMAGE"; then
    log "docker pull failed; trying image rm then pull"
    sudo docker image rm -f "$BASE_IMAGE" || true
    sudo docker pull "$BASE_IMAGE"
  fi

  docker_smoke_test_image "$BASE_IMAGE" || {
    sed -n '1,160p' "$RUNTIME_DIR/docker_smoke.err" || true
    fail "base image still cannot start. Docker/containerd store may need manual reset."
  }

  log "base image ready"
}

########################################
# 4. 下载模型，可重复执行
########################################

validate_model_dir() {
  sudo docker run --rm -i \
    -e "HF_TOKEN=${HF_TOKEN:-}" \
    -e "HF_HOME=${CONTAINER_WORKDIR}/hf-cache" \
    -e HF_XET_HIGH_PERFORMANCE=1 \
    -e "HF_DOWNLOAD_MAX_WORKERS=${HF_DOWNLOAD_MAX_WORKERS}" \
    -e "MODEL_ID=${MODEL_ID}" \
    -e "MODEL_REVISION=${MODEL_REVISION}" \
    -e "MODEL_DIR=${MODEL_DIR}" \
    -v "${HOST_WORKDIR}:${CONTAINER_WORKDIR}" \
    --entrypoint python3 \
    "$BASE_IMAGE" \
    - <<'PY'
import hashlib
import json
import os
import pathlib
import sys
import time

model_id = os.environ["MODEL_ID"]
root = pathlib.Path(os.environ["MODEL_DIR"])
token = os.environ.get("HF_TOKEN") or None
errors = []
started_at = time.monotonic()
VALIDATOR_VERSION = "hf-manifest-size-sha256-v2"
print(f"VALIDATOR_VERSION {VALIDATOR_VERSION}")

def lfs_dict(sibling):
    lfs = getattr(sibling, "lfs", None)
    if isinstance(lfs, dict):
        return lfs
    if lfs is None:
        return {}
    out = {}
    for key in ("size", "sha256", "oid"):
        if hasattr(lfs, key):
            out[key] = getattr(lfs, key)
    return out

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024 * 8), b""):
            h.update(chunk)
    return h.hexdigest()

def local_structural_check():
    local_errors = []
    if not root.is_dir():
        local_errors.append(f"missing model directory: {root}")
        return local_errors

    required_any = [
        ("config", ["config.json"]),
        ("tokenizer", ["tokenizer.json", "tokenizer.model"]),
        ("tokenizer_config", ["tokenizer_config.json"]),
    ]
    for label, names in required_any:
        if not any((root / name).is_file() and (root / name).stat().st_size > 0 for name in names):
            local_errors.append(f"missing {label}: one of {names}")

    leftovers = []
    for pattern in ("*.incomplete", "*.lock", "*.tmp"):
        leftovers.extend(root.rglob(pattern))
    if leftovers:
        preview = ", ".join(str(p.relative_to(root)) for p in leftovers[:20])
        local_errors.append(f"found incomplete/lock/tmp files: {preview}")

    index_files = sorted(root.glob("*.safetensors.index.json")) + sorted(root.glob("*.bin.index.json"))
    if index_files:
        expected = set()
        for index_file in index_files:
            try:
                data = json.loads(index_file.read_text())
            except Exception as exc:
                local_errors.append(f"cannot parse {index_file.name}: {exc}")
                continue
            expected.update(data.get("weight_map", {}).values())
        for rel in sorted(expected):
            p = root / rel
            if not p.is_file() or p.stat().st_size == 0:
                local_errors.append(f"missing or empty weight shard listed by index: {rel}")
    else:
        weights = list(root.glob("*.safetensors")) + list(root.glob("*.bin"))
        if not any(p.is_file() and p.stat().st_size > 0 for p in weights):
            local_errors.append("missing non-empty model weights: *.safetensors or *.bin")

    shard_re = re.compile(r"^model-(\d+)-of-(\d+)\.(safetensors|bin)$")
    shard_groups = {}
    for path in root.glob("model-*-of-*.*"):
        match = shard_re.match(path.name)
        if not match:
            continue
        idx = int(match.group(1))
        total = int(match.group(2))
        ext = match.group(3)
        shard_groups.setdefault((total, ext), set()).add(idx)

    for (total, ext), present in sorted(shard_groups.items()):
        missing = [i for i in range(1, total + 1) if i not in present]
        if missing:
            preview = ", ".join(f"model-{i:05d}-of-{total:05d}.{ext}" for i in missing[:20])
            suffix = " ..." if len(missing) > 20 else ""
            local_errors.append(f"missing {len(missing)} of {total} numbered {ext} shards: {preview}{suffix}")
        for i in sorted(present):
            p = root / f"model-{i:05d}-of-{total:05d}.{ext}"
            if not p.is_file() or p.stat().st_size == 0:
                local_errors.append(f"empty numbered shard: {p.name}")

    if not index_files:
        for ext in ("safetensors", "bin"):
            numbered = sorted(root.glob(f"model-*-of-*.{ext}"))
            if not numbered:
                continue
            total_values = []
            for path in numbered:
                match = shard_re.match(path.name)
                if match:
                    total_values.append(int(match.group(2)))
            if total_values:
                expected_total = max(total_values)
                if len(numbered) != expected_total:
                    local_errors.append(
                        f"local hard gate: found {len(numbered)} numbered {ext} shards, expected {expected_total}"
                    )
            break
    return local_errors

def hub_manifest_check():
    hub_errors = []
    if not root.is_dir():
        hub_errors.append(f"missing model directory: {root}")
        return hub_errors

    from huggingface_hub import HfApi

    api = HfApi()
    revision = os.environ["MODEL_REVISION"]
    print(f"HF_MANIFEST_REVISION {revision}")

    try:
        info = api.model_info(model_id, revision=revision, token=token, files_metadata=True)
    except TypeError:
        info = api.model_info(model_id, revision=revision, token=token)

    missing = []
    mismatched = []
    sha_mismatched = []
    remote_files = []
    size_checked = 0
    sha_checked = 0
    weight_sha_checked = 0
    for sibling in info.siblings:
        rel = getattr(sibling, "rfilename", None)
        if not rel or rel == ".gitattributes":
            continue
        expected_size = getattr(sibling, "size", None)
        lfs = lfs_dict(sibling)
        if expected_size is None:
            expected_size = lfs.get("size")
        expected_sha256 = lfs.get("sha256") or lfs.get("oid")
        remote_files.append(rel)
        p = root / rel
        if not p.is_file():
            missing.append(rel)
        elif isinstance(expected_size, int) and p.stat().st_size != expected_size:
            size_checked += 1
            mismatched.append(f"{rel}: local={p.stat().st_size}, hub={expected_size}")
        elif p.stat().st_size == 0:
            mismatched.append(f"{rel}: empty")
        else:
            if isinstance(expected_size, int):
                size_checked += 1
        if p.is_file() and expected_sha256:
            actual_sha256 = sha256_file(p)
            sha_checked += 1
            if rel.endswith((".safetensors", ".bin")):
                weight_sha_checked += 1
            if actual_sha256 != expected_sha256:
                sha_mismatched.append(f"{rel}: local_sha256={actual_sha256}, hub_sha256={expected_sha256}")

    if not remote_files:
        hub_errors.append("huggingface manifest is empty; refusing to treat model as valid")

    if missing:
        hub_errors.append(f"huggingface manifest says {len(missing)} repo files are missing")
        hub_errors.extend(f"missing from Hub manifest: {name}" for name in missing[:80])
    if mismatched:
        hub_errors.append(f"huggingface manifest says {len(mismatched)} repo files have wrong size")
        hub_errors.extend(f"size mismatch: {name}" for name in mismatched[:80])
    if sha_mismatched:
        hub_errors.append(f"huggingface manifest says {len(sha_mismatched)} LFS files have wrong sha256")
        hub_errors.extend(f"sha256 mismatch: {name}" for name in sha_mismatched[:20])

    shard_re = re.compile(r"^(.+-)?(\d+)-of-(\d+)\.(safetensors|bin)$")
    remote_shards = [name for name in remote_files if shard_re.match(pathlib.PurePosixPath(name).name)]
    if remote_shards:
        print(f"HF_REMOTE_SHARDS {len(remote_shards)}")
        local_shards = {
            str(path.relative_to(root)).replace(os.sep, "/")
            for path in root.rglob("*")
            if path.is_file() and shard_re.match(path.name)
        }
        print(f"HF_LOCAL_SHARDS {len(local_shards)}")
        missing_shards = sorted(set(remote_shards) - local_shards)
        if missing_shards:
            hub_errors.append(f"missing {len(missing_shards)} weight shards from Hugging Face manifest")
            hub_errors.extend(f"missing shard: {name}" for name in missing_shards[:80])
        if weight_sha_checked == 0:
            hub_errors.append("strict sha256 validation did not run for any weight shard; refusing fast pass")

    print(f"HF_REMOTE_FILES {len(remote_files)}")
    print(f"HF_SIZE_CHECKED {size_checked}")
    print(f"HF_SHA256_CHECKED {sha_checked}")
    print(f"HF_WEIGHT_SHA256_CHECKED {weight_sha_checked}")

    return hub_errors

try:
    errors.extend(hub_manifest_check())
except Exception as exc:
    errors.append(f"huggingface manifest validation unavailable: {type(exc).__name__}: {exc}")

errors.extend(local_structural_check())

if errors:
    print("MODEL_DIR_INVALID")
    for error in errors:
        print(f"- {error}")
    sys.exit(1)

elapsed = time.monotonic() - started_at
print(f"VALIDATION_ELAPSED_SECONDS {elapsed:.1f}")
print("MODEL_DIR_OK_STRICT_HF_MANIFEST_SIZE_SHA256")
PY
}

local_shard_gate() {
  [ -d "$MODEL_DIR_HOST" ] || {
    echo "MODEL_DIR_INVALID"
    echo "- missing model directory: $MODEL_DIR_HOST"
    return 1
  }

  local first total ext count missing preview name
  first="$(find "$MODEL_DIR_HOST" -maxdepth 1 -type f \( -name 'model-*-of-*.safetensors' -o -name 'model-*-of-*.bin' \) -printf '%f\n' 2>/dev/null | sort | head -n 1 || true)"
  if [ -z "$first" ]; then
    echo "MODEL_DIR_INVALID"
    echo "- missing numbered model shards: $MODEL_DIR_HOST"
    return 1
  fi

  case "$first" in
    *.safetensors) ext="safetensors" ;;
    *.bin) ext="bin" ;;
    *)
      echo "MODEL_DIR_INVALID"
      echo "- missing numbered model shards: $MODEL_DIR_HOST"
      return 1
      ;;
  esac
  total="$(printf '%s\n' "$first" | sed -nE "s/^model-[0-9]+-of-([0-9]+)\\.${ext}$/\\1/p")"
  if [ -z "$total" ]; then
    echo "MODEL_DIR_INVALID"
    echo "- missing numbered model shards: $MODEL_DIR_HOST"
    return 1
  fi
  total="$((10#$total))"

  count="$(find "$MODEL_DIR_HOST" -maxdepth 1 -type f -name "model-*-of-*.$ext" | wc -l | tr -d ' ')"
  if [ "$count" -ne "$total" ]; then
    preview=""
    for i in $(seq 1 "$total"); do
      name="$(printf "model-%05d-of-%05d.%s" "$i" "$total" "$ext")"
      [ -f "${MODEL_DIR_HOST}/${name}" ] && continue
      preview="${preview}${preview:+, }${name}"
      [ "$(printf '%s' "$preview" | awk -F, '{print NF}')" -ge 20 ] && break
    done
    echo "MODEL_DIR_INVALID"
    echo "- local hard gate: found $count numbered $ext shards, expected $total"
    echo "- missing shard preview: ${preview:-unknown}"
    return 1
  fi

  echo "LOCAL_SHARD_GATE_OK found $count/$total numbered $ext shards"
}

model_ready() {
  log "quick check model directory: $MODEL_DIR_HOST"
  : > "$RUNTIME_DIR/model_validate.out"
  : > "$RUNTIME_DIR/model_validate.err"

  [ -d "$MODEL_DIR_HOST" ] || {
    echo "MODEL_DIR_INVALID" >>"$RUNTIME_DIR/model_validate.out"
    echo "- missing model directory: $MODEL_DIR_HOST" >>"$RUNTIME_DIR/model_validate.out"
    cat "$RUNTIME_DIR/model_validate.out" 2>/dev/null || true
    return 1
  }

  [ -s "$MODEL_DIR_HOST/config.json" ] || {
    echo "MODEL_DIR_INVALID" >>"$RUNTIME_DIR/model_validate.out"
    echo "- missing or empty config.json" >>"$RUNTIME_DIR/model_validate.out"
    cat "$RUNTIME_DIR/model_validate.out" 2>/dev/null || true
    return 1
  }

  [ -s "$MODEL_DIR_HOST/tokenizer_config.json" ] || {
    echo "MODEL_DIR_INVALID" >>"$RUNTIME_DIR/model_validate.out"
    echo "- missing or empty tokenizer_config.json" >>"$RUNTIME_DIR/model_validate.out"
    cat "$RUNTIME_DIR/model_validate.out" 2>/dev/null || true
    return 1
  }

  if ! [ -s "$MODEL_DIR_HOST/tokenizer.json" ] && ! [ -s "$MODEL_DIR_HOST/tokenizer.model" ]; then
    echo "MODEL_DIR_INVALID" >>"$RUNTIME_DIR/model_validate.out"
    echo "- missing tokenizer.json or tokenizer.model" >>"$RUNTIME_DIR/model_validate.out"
    cat "$RUNTIME_DIR/model_validate.out" 2>/dev/null || true
    return 1
  fi

  local first_weight
  first_weight="$(find "$MODEL_DIR_HOST" -maxdepth 1 -type f \( -name '*.safetensors' -o -name '*.bin' \) -size +0c -print -quit 2>/dev/null || true)"
  if [ -z "$first_weight" ]; then
    echo "MODEL_DIR_INVALID" >>"$RUNTIME_DIR/model_validate.out"
    echo "- missing non-empty model weights: *.safetensors or *.bin" >>"$RUNTIME_DIR/model_validate.out"
    cat "$RUNTIME_DIR/model_validate.out" 2>/dev/null || true
    return 1
  fi

  if find "$MODEL_DIR_HOST" -maxdepth 1 -type f \( -name '*.safetensors.index.json' -o -name '*.bin.index.json' \) -size +0c -print -quit 2>/dev/null | grep -q .; then
    echo "LOCAL_INDEX_GATE_OK found weight index file" >>"$RUNTIME_DIR/model_validate.out"
  else
    if local_shard_gate >>"$RUNTIME_DIR/model_validate.out" 2>>"$RUNTIME_DIR/model_validate.err"; then
      :
    elif find "$MODEL_DIR_HOST" -maxdepth 1 -type f \( -name '*.safetensors' -o -name '*.bin' \) ! -name 'model-*-of-*' -size +0c -print -quit 2>/dev/null | grep -q .; then
      echo "SINGLE_WEIGHT_GATE_OK found non-sharded weight file" >>"$RUNTIME_DIR/model_validate.out"
    else
      cat "$RUNTIME_DIR/model_validate.out" 2>/dev/null || true
      cat "$RUNTIME_DIR/model_validate.err" 2>/dev/null || true
      return 1
    fi
  fi

  echo "MODEL_DIR_QUICK_OK revision=$MODEL_REVISION" >>"$RUNTIME_DIR/model_validate.out"
  cat "$RUNTIME_DIR/model_validate.out" 2>/dev/null || true
}

cleanup_stale_downloads_if_enabled() {
  [ "$AUTO_KILL_STALE_DOWNLOADS" = "1" ] || return 0

  log "cleanup stale HF downloads for model/workdir"
  sudo docker rm -f glm51-model-download >/dev/null 2>&1 || true

  pgrep -af 'hf download|huggingface_hub|hf_transfer|xet|python3' | \
    grep -F -e "$MODEL_ID" -e "$MODEL_DIR" -e "$MODEL_DIR_HOST" -e "$HOST_WORKDIR" | \
    awk '{print $1}' | \
    sort -u | \
    xargs -r kill -TERM 2>/dev/null || true
  sleep 3
  pgrep -af 'hf download|huggingface_hub|hf_transfer|xet|python3' | \
    grep -F -e "$MODEL_ID" -e "$MODEL_DIR" -e "$MODEL_DIR_HOST" -e "$HOST_WORKDIR" | \
    awk '{print $1}' | \
    sort -u | \
    xargs -r kill -KILL 2>/dev/null || true

  if [ -d "$MODEL_DIR_HOST" ]; then
    find "$MODEL_DIR_HOST" -maxdepth 2 \
      \( -name '*.lock' -o -name '*.incomplete' -o -name '*.tmp' -o -name '.cache' \) \
      -print -exec rm -rf {} + 2>/dev/null || true
  fi
}

run_snapshot_download() {
  log "snapshot_download: $MODEL_ID -> $MODEL_DIR (max_workers=$HF_DOWNLOAD_MAX_WORKERS)"
  sudo mkdir -p "$MODEL_DIR_HOST"
  sudo chown -R "$(id -u):$(id -g)" "$MODEL_DIR_HOST" || true

  sudo docker run --rm -i \
    --name glm51-model-download \
    -e "HF_TOKEN=${HF_TOKEN:-}" \
    -e "HF_HOME=${CONTAINER_WORKDIR}/hf-cache" \
    -e HF_XET_HIGH_PERFORMANCE=1 \
    -e "HF_DOWNLOAD_MAX_WORKERS=${HF_DOWNLOAD_MAX_WORKERS}" \
    -e "MODEL_ID=${MODEL_ID}" \
    -e "MODEL_REVISION=${MODEL_REVISION}" \
    -e "MODEL_DIR=${MODEL_DIR}" \
    -v "${HOST_WORKDIR}:${CONTAINER_WORKDIR}" \
    --entrypoint python3 \
    "$BASE_IMAGE" \
    - <<'PY'
import inspect
import os

from huggingface_hub import snapshot_download

token = os.environ.get("HF_TOKEN") or None
revision = os.environ["MODEL_REVISION"]
print(f"SNAPSHOT_DOWNLOAD_REVISION {revision}")

kwargs = {
    "repo_id": os.environ["MODEL_ID"],
    "revision": revision,
    "local_dir": os.environ["MODEL_DIR"],
    "token": token,
    "max_workers": int(os.environ.get("HF_DOWNLOAD_MAX_WORKERS", "8")),
}
sig = inspect.signature(snapshot_download)
if "resume_download" in sig.parameters:
    kwargs["resume_download"] = True
if "local_dir_use_symlinks" in sig.parameters:
    kwargs["local_dir_use_symlinks"] = False

path = snapshot_download(**kwargs)
print(f"SNAPSHOT_DOWNLOAD_OK {path}")
PY
}

download_model() {
  ensure_space_or_cleanup "$HOST_WORKDIR" "model download workspace" "$MIN_MODEL_DOWNLOAD_FREE_GB" 1
  cleanup_stale_downloads_if_enabled

  log "download/refresh model: $MODEL_ID@$MODEL_REVISION -> $MODEL_DIR"
  run_snapshot_download

  if ! model_ready; then
    if [ "$AUTO_DELETE_MODEL_IF_LOW_SPACE" = "1" ]; then
      log "model still invalid after refresh; deleting and retrying once: $MODEL_DIR_HOST"
      rm -rf "$MODEL_DIR_HOST"
      run_snapshot_download
    fi
  fi

  if ! model_ready; then
    fail "model download finished but model directory is incomplete or invalid: $MODEL_DIR_HOST"
  fi
  log "model ready"
}

scrub_sglang_attention_scale_inv_keys() {
  fail "checkpoint attention scale scrubbing is disabled; use the SGLang quark loader runtime patch instead"
}

run_glm5_reference_merge_quant() {
  log "run glm5-fp8-deploy reference merge+quant scripts without editing their implementation"
  install_glm5_reference_scripts
  check_glm5_reference_amd_runtime
  download_model

  local tinker_body tinker_weight_path tinker_weight_name effective_fp8
  local local_quant_root_host
  local scripts_container="/opt/glm5-fp8-deploy/scripts"
  tinker_body="${TINKER_URL#tinker://}"
  tinker_weight_path="${tinker_body#*/weights/}"
  tinker_weight_name="${tinker_weight_path##*/}"
  [ -n "$tinker_weight_name" ] || fail "cannot parse Tinker weight name from TINKER_URL=$TINKER_URL"

  sudo rm -rf "$MERGED_MODEL_DIR_HOST" "$QUANT_MODEL_DIR_HOST" "$LORA_DIR_HOST"
  local_quant_root_host="$(host_path_for_container_path "$GLM5_LOCAL_QUANT_ROOT")"
  sudo mkdir -p "$MERGED_MODEL_DIR_HOST" "$QUANT_MODEL_DIR_HOST" "$LORA_DIR_HOST" "$local_quant_root_host"
  sudo chown "$(id -u):$(id -g)" "$MERGED_MODEL_DIR_HOST" "$QUANT_MODEL_DIR_HOST" "$LORA_DIR_HOST" "$local_quant_root_host" || true

  sudo docker rm -f glm51-tinker-merge-quant >/dev/null 2>&1 || true
  sudo docker run --rm -i \
    --name glm51-tinker-merge-quant \
    --network host \
    --ipc host \
    --device /dev/kfd \
    --device /dev/dri \
    --group-add video \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --security-opt label=disable \
    --shm-size 16G \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "${HOST_WORKDIR}:${CONTAINER_WORKDIR}" \
    -v "${GLM5_REFERENCE_SCRIPTS_DIR}:${scripts_container}:ro" \
    -e "GPU_LEASE_BASE_URL=${GPU_LEASE_BASE_URL}" \
    -e "GPU_LEASE_API_KEY=${GPU_LEASE_API_KEY}" \
    -e "GLM5_BASE=${MODEL_DIR}" \
    -e "GLM5_ADAPTER_ROOT=$(dirname "$LORA_DIR")" \
    -e "GLM5_LOCAL_QUANT_ROOT=${GLM5_LOCAL_QUANT_ROOT}" \
    -e "GLM5_FP8_SHARDWISE=${GLM5_FP8_SHARDWISE:-1}" \
    -e "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}" \
    -e "HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}" \
    -e "HF_HOME=${CONTAINER_WORKDIR}/hf-cache" \
    -e "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-}" \
    -e "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-}" \
    -e "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" \
    --entrypoint bash \
    "$BASE_IMAGE" \
    -lc 'set -Eeuo pipefail
      [ -n "${HIP_VISIBLE_DEVICES:-}" ] || unset HIP_VISIBLE_DEVICES
      [ -n "${ROCR_VISIBLE_DEVICES:-}" ] || unset ROCR_VISIBLE_DEVICES
      [ -n "${CUDA_VISIBLE_DEVICES:-}" ] || unset CUDA_VISIBLE_DEVICES
      if [ -x /opt/venv/bin/python3 ]; then
        [ -e /usr/bin/python3.12.system ] || mv /usr/bin/python3.12 /usr/bin/python3.12.system
        cat > /usr/bin/python3.12 <<'"'"'PYSH'"'"'
#!/usr/bin/env bash
exec -a /opt/venv/bin/python3 /usr/bin/python3.12.system "$@"
PYSH
        chmod 0755 /usr/bin/python3.12
      else
        py="$(command -v python3 || true)"
        [ -n "$py" ] || { echo "python3 not found in container" >&2; exit 1; }
        ln -sfn "$py" /usr/bin/python3.12
      fi
      export PATH="/opt/venv/bin:$PATH"
      echo "reference_python=$(/usr/bin/python3.12 -c '"'"'import sys; print(sys.executable)'"'"')"
      /usr/bin/python3.12 - <<'"'"'PY'"'"'
import importlib.util
import sys
missing = [name for name in ("torch", "safetensors", "transformers") if importlib.util.find_spec(name) is None]
if missing:
    print("missing_python_modules=" + ",".join(missing), file=sys.stderr)
    raise SystemExit(1)
import torch
print("torch_version", torch.__version__)
print("torch_version_hip", getattr(torch.version, "hip", None))
print("torch_cuda_is_available", torch.cuda.is_available())
print("torch_cuda_device_count", torch.cuda.device_count())
if getattr(torch.version, "hip", None) is None:
    raise SystemExit("expected ROCm PyTorch build with torch.version.hip set")
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("ROCm PyTorch did not expose GPUs through torch.cuda")
PY
      exec "$@"' \
    glm5-reference \
    "${scripts_container}/merge_quant_serve_glm51_fp8.sh" \
      --tinker-url "$TINKER_URL" \
      --base "$MODEL_DIR" \
      --lora "$LORA_DIR" \
      --merged "$MERGED_MODEL_DIR" \
      --fp8 "$QUANT_MODEL_DIR" \
      --served-model-name "$SERVED_MODEL_NAME" \
      --gpus "$GLM5_MERGE_QUANT_GPUS" \
      --workers-per-gpu "$GLM5_MERGE_QUANT_WORKERS_PER_GPU" \
      --adapter-root "$(dirname "$LORA_DIR")" \
      --skip-serve \
      --force-download-adapter \
      --force-merge \
      --force-quant

  effective_fp8="$QUANT_MODEL_DIR_HOST"
  if [ ! -f "$effective_fp8/fp8_quant_meta.json" ]; then
    effective_fp8="${local_quant_root_host}/$(basename "$QUANT_MODEL_DIR_HOST")"
  fi
  [ -f "$effective_fp8/fp8_quant_meta.json" ] || fail "reference quant did not produce fp8_quant_meta.json under $QUANT_MODEL_DIR_HOST or $effective_fp8"

  if [ "$effective_fp8" != "$QUANT_MODEL_DIR_HOST" ]; then
    sudo rm -rf "$QUANT_MODEL_DIR_HOST"
    sudo mkdir -p "$(dirname "$QUANT_MODEL_DIR_HOST")"
    sudo mv "$effective_fp8" "$QUANT_MODEL_DIR_HOST"
    sudo chown -R "$(id -u):$(id -g)" "$QUANT_MODEL_DIR_HOST" || true
  fi

  log "skip checkpoint scrub; SGLang quark attention A-proj loader issue is handled by runtime patch at serve time"
}

check_glm5_reference_amd_runtime() {
  log "check glm5-fp8-deploy reference runtime inside AMD/ROCm Docker image: $BASE_IMAGE"
  sudo docker run --rm -i \
    --name glm51-tinker-merge-quant-check \
    --network host \
    --ipc host \
    --device /dev/kfd \
    --device /dev/dri \
    --group-add video \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --security-opt label=disable \
    --shm-size 4G \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-}" \
    -e "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-}" \
    -e "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" \
    --entrypoint bash \
    "$BASE_IMAGE" \
    -lc 'set -Eeuo pipefail
      [ -n "${HIP_VISIBLE_DEVICES:-}" ] || unset HIP_VISIBLE_DEVICES
      [ -n "${ROCR_VISIBLE_DEVICES:-}" ] || unset ROCR_VISIBLE_DEVICES
      [ -n "${CUDA_VISIBLE_DEVICES:-}" ] || unset CUDA_VISIBLE_DEVICES
      if [ -x /opt/venv/bin/python3 ]; then
        [ -e /usr/bin/python3.12.system ] || mv /usr/bin/python3.12 /usr/bin/python3.12.system
        cat > /usr/bin/python3.12 <<'"'"'PYSH'"'"'
#!/usr/bin/env bash
exec -a /opt/venv/bin/python3 /usr/bin/python3.12.system "$@"
PYSH
        chmod 0755 /usr/bin/python3.12
      else
        py="$(command -v python3 || true)"
        [ -n "$py" ] || { echo "python3 not found in container" >&2; exit 1; }
        ln -sfn "$py" /usr/bin/python3.12
      fi
      export PATH="/opt/venv/bin:$PATH"
      echo "reference_python=$(/usr/bin/python3.12 -c '"'"'import sys; print(sys.executable)'"'"')"
      /usr/bin/python3.12 - <<'"'"'PY'"'"'
import importlib.util
import sys

required = ["torch", "safetensors", "transformers"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print("missing_python_modules=" + ",".join(missing), file=sys.stderr)
    raise SystemExit(1)

import torch

hip = getattr(torch.version, "hip", None)
cuda = getattr(torch.version, "cuda", None)
print(f"torch_version={torch.__version__}")
print(f"torch_version_hip={hip}")
print(f"torch_version_cuda={cuda}")
print(f"torch_cuda_is_available={torch.cuda.is_available()}")
print(f"torch_cuda_device_count={torch.cuda.device_count()}")

if not hip:
    print("expected a ROCm PyTorch build with torch.version.hip set; this looks like a CUDA/NVIDIA or CPU build", file=sys.stderr)
    raise SystemExit(1)
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    print("ROCm PyTorch did not expose any GPUs through torch.cuda", file=sys.stderr)
    raise SystemExit(1)
PY
    ' || fail "glm5-fp8-deploy reference runtime is not AMD/ROCm compatible inside $BASE_IMAGE; refusing LoRA merge/quant"
}

prepare_tinker_merge_quant_if_enabled() {
  if ! tinker_merge_quant_enabled; then
    log "skip Tinker merge/quant because RUN_TINKER_MERGE_QUANT=$RUN_TINKER_MERGE_QUANT"
    return
  fi

  run_glm5_reference_merge_quant

  MODEL_DIR="$QUANT_MODEL_DIR"
  MODEL_DIR_HOST="$QUANT_MODEL_DIR_HOST"
  export MODEL_DIR MODEL_DIR_HOST
  model_ready || fail "quantized model directory is incomplete or invalid: $MODEL_DIR_HOST"
  write_active_model_env
  log "serving quantized merged model from MODEL_DIR=$MODEL_DIR"
}

prepare_model_for_serve() {
  if tinker_merge_quant_enabled; then
    prepare_tinker_merge_quant_if_enabled
  else
    log "prepare base model for serve; no Tinker LoRA merge/quant enabled"
    download_model
    write_active_model_env
  fi
}

########################################
# 5. 拉取/校验 SGLang serve 镜像，可重复执行
########################################

sync_atom_repo_if_needed() {
  local resolved_ref head_ref plugin_init parent_dir trimmed_atom_ref

  require_cmd git
  : "${ATOM_REPO_URL:?ATOM_REPO_URL empty}"
  : "${ATOM_REF:?ATOM_REF empty}"
  : "${ATOM_REPO_HOST:?ATOM_REPO_HOST empty}"

  trimmed_atom_ref="$(printf '%s' "$ATOM_REF" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  [ -n "$trimmed_atom_ref" ] || fail "ATOM_REF is empty after trimming whitespace"
  ATOM_REF="$trimmed_atom_ref"
  export ATOM_REF

  atom_git() { sudo git -c safe.directory="$ATOM_REPO_HOST" -C "$ATOM_REPO_HOST" "$@"; }

  parent_dir="$(dirname "$ATOM_REPO_HOST")"
  sudo mkdir -p "$parent_dir" || fail "cannot create ATOM repo parent directory: $parent_dir"

  if [ -e "$ATOM_REPO_HOST" ] && [ ! -d "${ATOM_REPO_HOST}/.git" ]; then
    fail "ATOM_REPO_HOST exists but is not a git repo; refusing to overwrite: $ATOM_REPO_HOST"
  fi

  if [ ! -d "${ATOM_REPO_HOST}/.git" ]; then
    log "[atom-repo] clone ATOM repo with sudo git: url=${ATOM_REPO_URL} path=${ATOM_REPO_HOST}"
    sudo git clone "$ATOM_REPO_URL" "$ATOM_REPO_HOST" || fail "git clone failed for ATOM repo: ${ATOM_REPO_URL} -> ${ATOM_REPO_HOST}"
  fi

  atom_git remote set-url origin "$ATOM_REPO_URL" || fail "cannot set ATOM origin URL: $ATOM_REPO_URL"
  if ! atom_git fetch --tags origin; then
    log "[atom-repo] fetch --tags failed; retry fetching pinned ref directly: ${ATOM_REF}"
    atom_git fetch origin "$ATOM_REF" || log "[atom-repo] warning: direct fetch failed for ATOM_REF=${ATOM_REF}; will verify local refs before failing"
  fi

  if resolved_ref="$(atom_git rev-parse --verify "${ATOM_REF}^{commit}" 2>/dev/null)"; then
    :
  else
    log "[atom-repo] ATOM_REF not resolvable after fetch --tags; retry fetching pinned ref directly: ${ATOM_REF}"
    atom_git fetch origin "$ATOM_REF" || log "[atom-repo] warning: direct fetch failed for ATOM_REF=${ATOM_REF}; retrying rev-parse before final failure"
    if resolved_ref="$(atom_git rev-parse --verify "${ATOM_REF}^{commit}" 2>/dev/null)"; then
      :
    else
      fail "cannot resolve ATOM_REF to a commit after fetch attempts: ATOM_REF=${ATOM_REF} ATOM_REPO_URL=${ATOM_REPO_URL} ATOM_REPO_HOST=${ATOM_REPO_HOST}. Check that the ref exists and that network fetch succeeded."
    fi
  fi
  [ -n "$resolved_ref" ] || fail "cannot resolve ATOM_REF to a commit after fetch attempts: ATOM_REF=${ATOM_REF} ATOM_REPO_URL=${ATOM_REPO_URL} ATOM_REPO_HOST=${ATOM_REPO_HOST}. Check that the ref exists and that network fetch succeeded."
  log "[atom-repo] resolved ATOM ref: ref=${ATOM_REF} resolved=${resolved_ref}"

  atom_git checkout --detach "$resolved_ref" || fail "cannot checkout ATOM ref: $ATOM_REF (resolved=${resolved_ref})"
  head_ref="$(atom_git rev-parse HEAD)"
  [ "$head_ref" = "$resolved_ref" ] || fail "ATOM checkout mismatch: ref=$ATOM_REF resolved=$resolved_ref head=$head_ref"

  plugin_init="${ATOM_REPO_HOST}/atom/plugin/sglang/models/__init__.py"
  [ -f "$plugin_init" ] || sudo test -f "$plugin_init" || fail "ATOM PR355 SGLang plugin entry missing: $plugin_init"

  log "[atom-repo] synced ATOM repo: path=${ATOM_REPO_HOST} ref=${ATOM_REF} resolved=${resolved_ref} head=${head_ref}"
}

sglang_image_ready() {
  local atom_mount_args=()
  if [ -n "${ATOM_REPO_HOST:-}" ]; then
    [ -d "$ATOM_REPO_HOST" ] || fail "ATOM_REPO_HOST is set but not a directory: $ATOM_REPO_HOST"
    [ -f "${ATOM_REPO_HOST}/atom/plugin/sglang/models/__init__.py" ] || fail "ATOM SGLang plugin entry missing: ${ATOM_REPO_HOST}/atom/plugin/sglang/models/__init__.py"
    atom_mount_args=(-v "${ATOM_REPO_HOST}:${ATOM_REPO_CONTAINER}:ro")
  fi
  sudo docker image inspect "$SGLANG_IMAGE" >/dev/null 2>&1 &&   sudo docker run --rm -i --entrypoint python3     --ipc host     --device /dev/kfd     --device /dev/dri     --group-add video     --cap-add SYS_PTRACE     --security-opt seccomp=unconfined     --security-opt label=disable     "${atom_mount_args[@]}"     -e "PYTHONPATH=${ATOM_PLUGIN_PYTHONPATH}"     -e "SGLANG_EXTERNAL_MODEL_PACKAGE=${SGLANG_EXTERNAL_MODEL_PACKAGE}"     "$SGLANG_IMAGE" - <<'PY' >"$RUNTIME_DIR/image_check.out" 2>"$RUNTIME_DIR/image_check.err"
import importlib
import os
import sglang
import torch
print('torch', torch.__version__, 'hip', torch.version.hip)
print('cuda_available', torch.cuda.is_available())
print('device_count', torch.cuda.device_count())
print('sglang', getattr(sglang, '__version__', 'unknown'), getattr(sglang, '__file__', ''))
print('PYTHONPATH', os.environ.get('PYTHONPATH', ''))
plugin = os.environ['SGLANG_EXTERNAL_MODEL_PACKAGE']
module = importlib.import_module(plugin)
print('external_model_package', plugin, getattr(module, '__file__', ''))
assert torch.version.hip is not None
assert torch.cuda.device_count() > 0
PY
}

ensure_sglang_image() {
  sync_atom_repo_if_needed
  if sglang_image_ready; then
    log "SGLang ROCm image already OK: $SGLANG_IMAGE"
    cat "$RUNTIME_DIR/image_check.out" || true
    return
  fi

  log "pull/validate SGLang ROCm image: $SGLANG_IMAGE"
  sed -n '1,160p' "$RUNTIME_DIR/image_check.err" 2>/dev/null || true
  ensure_space_or_cleanup "$HOST_WORKDIR" "SGLang image workspace" "$MIN_DOCKER_BUILD_FREE_GB" 0
  local pull_err
  pull_err="${RUNTIME_DIR}/sglang_image_pull.err"
  : > "$pull_err"
  if ! sudo docker pull "$SGLANG_IMAGE" 2> >(tee "$pull_err" >&2); then
    log "docker pull failed; trying image rm then pull"
    sudo docker image rm -f "$SGLANG_IMAGE" || true
    : > "$pull_err"
    if ! sudo docker pull "$SGLANG_IMAGE" 2> >(tee "$pull_err" >&2); then
      if [ "${AUTO_FIX_DOCKER_CACHE:-1}" = "1" ] && pull_failed_with_snapshot_conflict "$pull_err"; then
        local target_snapshot_id=""
        fix_docker_snapshot_conflict_light
        : > "$pull_err"
        if ! sudo docker pull "$SGLANG_IMAGE" 2> >(tee "$pull_err" >&2); then
          target_snapshot_id="$(extract_containerd_target_snapshot_id "$pull_err")"
          if [ -n "$target_snapshot_id" ] && fix_containerd_target_snapshot_light "$target_snapshot_id"; then
            : > "$pull_err"
            sudo docker pull "$SGLANG_IMAGE" 2> >(tee "$pull_err" >&2) || {
              sed -n '1,160p' "$pull_err" || true
              fail "SGLang image pull still failed after targeted snapshot repair; manually inspect /data/docker and containerd snapshotter. The script will not delete Docker/containerd roots automatically."
            }
          else
            sed -n '1,160p' "$pull_err" || true
            fail "SGLang image pull still failed after light repair; target snapshot could not be removed or was not found in stderr. Inspect leases/containers/child snapshots manually."
          fi
        fi
      else
        sed -n '1,160p' "$pull_err" || true
        fail "SGLang image pull failed; if stderr shows snapshot already exists, inspect /data/docker and containerd snapshotter. The script will not reset Docker data automatically."
      fi
    fi
  fi

  sglang_image_ready || {
    sed -n '1,200p' "$RUNTIME_DIR/image_check.err" || true
    fail "SGLang ROCm image validation or ATOM PR355 import check failed. Check ATOM_REPO_URL/ATOM_REF/ATOM_REPO_HOST sync logs, or use a custom image with ATOM PR355 built in."
  }
  cat "$RUNTIME_DIR/image_check.out" || true
  log "SGLang ROCm image ready"
}

# Keep the old subcommand name for muscle memory; it now pulls/smoke-tests the prebuilt SGLang image.
build_image() {
  ensure_sglang_image
}

########################################
# 6. 下载模型 + 拉取/校验 SGLang 镜像：顺序执行，避免 HF 下载和 Docker pull 互相抢资源
########################################

export_common_tmux_env() {
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" BASE_IMAGE "$BASE_IMAGE"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" GLM51_SCRIPT_VERSION "$GLM51_SCRIPT_VERSION"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" SGLANG_IMAGE "$SGLANG_IMAGE"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" SGLANG_CONTAINER "$SGLANG_CONTAINER"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" LMEVAL_IMAGE "$LMEVAL_IMAGE"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" LMEVAL_CONTAINER "$LMEVAL_CONTAINER"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" CONTROL_DIR "$CONTROL_DIR"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" GENERATED_SCRIPT_DIR "$GENERATED_SCRIPT_DIR"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" GLM51_SECRETS_FILE "$GLM51_SECRETS_FILE"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" HF_TOKEN_FILE "$HF_TOKEN_FILE"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" HOST_WORKDIR "$HOST_WORKDIR"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" CONTAINER_WORKDIR "$CONTAINER_WORKDIR"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" HOST_TMPDIR "$HOST_TMPDIR"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" MODEL_ID "$MODEL_ID"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" MODEL_REVISION "$MODEL_REVISION"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" MODEL_DIR "$MODEL_DIR"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" SERVED_MODEL_NAME "$SERVED_MODEL_NAME"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" RUN_TINKER_MERGE_QUANT "$RUN_TINKER_MERGE_QUANT"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" GPU_LEASE_BASE_URL "$GPU_LEASE_BASE_URL"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" LORA_DIR "$LORA_DIR"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" MERGED_MODEL_DIR "$MERGED_MODEL_DIR"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" QUANT_MODEL_DIR "$QUANT_MODEL_DIR"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" QUANT_SCHEME "$QUANT_SCHEME"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" GLM5_REFERENCE_SCRIPTS_DIR "$GLM5_REFERENCE_SCRIPTS_DIR"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" GLM5_MERGE_QUANT_GPUS "$GLM5_MERGE_QUANT_GPUS"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" GLM5_MERGE_QUANT_WORKERS_PER_GPU "$GLM5_MERGE_QUANT_WORKERS_PER_GPU"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" GLM5_LOCAL_QUANT_ROOT "$GLM5_LOCAL_QUANT_ROOT"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" SGLANG_PORT "$SGLANG_PORT"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" ATOM_REPO_URL "$ATOM_REPO_URL"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" ATOM_REF "$ATOM_REF"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" ATOM_REPO_ROOT "$ATOM_REPO_ROOT"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" ATOM_REPO_HOST "$ATOM_REPO_HOST"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" ATOM_REPO_CONTAINER "$ATOM_REPO_CONTAINER"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" ATOM_PLUGIN_PYTHONPATH "$ATOM_PLUGIN_PYTHONPATH"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" SGLANG_EXTERNAL_MODEL_PACKAGE "$SGLANG_EXTERNAL_MODEL_PACKAGE"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" AITER_QUICK_REDUCE_QUANTIZATION "$AITER_QUICK_REDUCE_QUANTIZATION"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" SGLANG_AITER_FP8_PREFILL_ATTN "$SGLANG_AITER_FP8_PREFILL_ATTN"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" PATCH_SGLANG_QUARK_FUSED_QKV_A_PROJ "$PATCH_SGLANG_QUARK_FUSED_QKV_A_PROJ"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" INSTALL_SYSTEMD_AUTOSTART "$INSTALL_SYSTEMD_AUTOSTART"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" AUTOSTART_SERVICE_NAME "$AUTOSTART_SERVICE_NAME"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" AUTOSTART_CHECK_INTERVAL_SECONDS "$AUTOSTART_CHECK_INTERVAL_SECONDS"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" ALLOW_REDOWNLOAD_ON_LOCAL_NVME_LOSS "$ALLOW_REDOWNLOAD_ON_LOCAL_NVME_LOSS"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" SGLANG_SERVE_ARGS "$SGLANG_SERVE_ARGS"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" LMEVAL_WORKDIR "$LMEVAL_WORKDIR"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" LMEVAL_WORKDIR_HOST "$LMEVAL_WORKDIR_HOST"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" STATE_DIR "$STATE_DIR"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" TMUX_TMPDIR "$TMUX_TMPDIR"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" TMUX_SOCKET "$TMUX_SOCKET"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" TMUX_SESSION "$TMUX_SESSION"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" PREP_WINDOW "$PREP_WINDOW"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" SERVE_WINDOW "$SERVE_WINDOW"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" LMEVAL_WINDOW "$LMEVAL_WINDOW"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" PERSIST_LOG_ROOT "${PERSIST_LOG_ROOT:-}"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" PERSIST_LOG_DIR "${PERSIST_LOG_DIR:-}"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" EXPERIMENT_NAME "${EXPERIMENT_NAME:-}"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" LOG_TIMEZONE "${LOG_TIMEZONE:-}"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" RUN_ID "${RUN_ID:-}"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" RUN_DIR "${RUN_DIR:-}"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" FIRST_RUN_CHECK_LOG_PERSIST "${FIRST_RUN_CHECK_LOG_PERSIST:-}"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" RUN_FIRST_RUN_CHECK_LOG_PERSIST "${RUN_FIRST_RUN_CHECK_LOG_PERSIST:-}"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" FIRST_RUN_LOG_COPY_TIMEOUT_SECONDS "${FIRST_RUN_LOG_COPY_TIMEOUT_SECONDS:-10}"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" AUTO_FIX_DOCKER_CACHE "$AUTO_FIX_DOCKER_CACHE"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" AUTO_FREE_SPACE "$AUTO_FREE_SPACE"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" AUTO_DELETE_MODEL_IF_LOW_SPACE "$AUTO_DELETE_MODEL_IF_LOW_SPACE"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" MIN_HOST_FREE_GB "$MIN_HOST_FREE_GB"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" MIN_MODEL_DOWNLOAD_FREE_GB "$MIN_MODEL_DOWNLOAD_FREE_GB"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" MIN_DOCKER_BUILD_FREE_GB "$MIN_DOCKER_BUILD_FREE_GB"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" AUTO_KILL_STALE_DOWNLOADS "$AUTO_KILL_STALE_DOWNLOADS"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" AUTO_KILL_EXISTING_CONTAINERS "$AUTO_KILL_EXISTING_CONTAINERS"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" AUTO_KILL_EXISTING_TMUX_SESSION "$AUTO_KILL_EXISTING_TMUX_SESSION"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" RUN_LMEVAL "$RUN_LMEVAL"
}

write_parallel_task_scripts() {
  cat > "${GENERATED_SCRIPT_DIR}/glm51_download_only.sh" <<'TASK'
#!/usr/bin/env bash
set -Eeuo pipefail
cd "$HOST_WORKDIR"
rm -f "${STATE_DIR}/download.done" "${STATE_DIR}/download.failed"
if "${GENERATED_SCRIPT_DIR}/glm51_resume.sh" download-only; then
  date '+%Y-%m-%d %H:%M:%S' > "${STATE_DIR}/download.done"
  echo "[DONE] model download"
else
  rc=$?
  echo "$rc" > "${STATE_DIR}/download.failed"
  echo "[FAILED] model download rc=$rc" >&2
  exit "$rc"
fi
TASK

  cat > "${GENERATED_SCRIPT_DIR}/glm51_build_only.sh" <<'TASK'
#!/usr/bin/env bash
set -Eeuo pipefail
cd "$HOST_WORKDIR"
rm -f "${STATE_DIR}/build.done" "${STATE_DIR}/build.failed"
if "${GENERATED_SCRIPT_DIR}/glm51_resume.sh" build-only; then
  date '+%Y-%m-%d %H:%M:%S' > "${STATE_DIR}/build.done"
  echo "[DONE] SGLang image validation"
else
  rc=$?
  echo "$rc" > "${STATE_DIR}/build.failed"
  echo "[FAILED] SGLang image validation rc=$rc" >&2
  exit "$rc"
fi
TASK

  chmod +x "${GENERATED_SCRIPT_DIR}/glm51_download_only.sh" "${GENERATED_SCRIPT_DIR}/glm51_build_only.sh"
}

run_download_and_build_sequential() {
  log "start model prep then SGLang image validation sequentially in tmux window: $PREP_WINDOW"
  mkdir -p "$STATE_DIR" "$TMUX_TMPDIR"
  chmod 700 "$TMUX_TMPDIR" || true

  rm -f "${STATE_DIR}/prep.done" "${STATE_DIR}/prep.failed" "$ACTIVE_MODEL_ENV"

  if ! tmux -S "$TMUX_SOCKET" has-session -t "$TMUX_SESSION" 2>/dev/null; then
    tmux -S "$TMUX_SOCKET" new-session -d -s "$TMUX_SESSION" -c "$HOST_WORKDIR"
  fi
  export_common_tmux_env

  local target="${TMUX_SESSION}:${PREP_WINDOW}"
  local prep_cmd
  printf -v prep_cmd 'if %q prepare-model && %q build-only; then date "+%%Y-%%m-%%d %%H:%%M:%%S" > %q; else rc=$?; echo "$rc" > %q; echo "[FAILED] prep model+image rc=$rc" >&2; fi; exec bash' \
    "${GENERATED_SCRIPT_DIR}/glm51_resume.sh" \
    "${GENERATED_SCRIPT_DIR}/glm51_resume.sh" \
    "${STATE_DIR}/prep.done" \
    "${STATE_DIR}/prep.failed"
  tmux -S "$TMUX_SOCKET" kill-window -t "$target" >/dev/null 2>&1 || true
  tmux -S "$TMUX_SOCKET" new-window -d -t "$TMUX_SESSION" -n "$PREP_WINDOW" -c "$HOST_WORKDIR" "$prep_cmd"

  log "waiting for sequential download/image; inspect tmux window: $PREP_WINDOW"
  local wait_count=0
  while true; do
    if [ -f "${STATE_DIR}/prep.done" ]; then
      break
    fi
    if [ -f "${STATE_DIR}/prep.failed" ]; then
      tmux -S "$TMUX_SOCKET" capture-pane -pt "$target" -S -160 || true
      fail "sequential download/image failed; check tmux window: $PREP_WINDOW"
    fi
    if ! tmux -S "$TMUX_SOCKET" list-windows -t "$TMUX_SESSION" -F '#W' 2>/dev/null | grep -qx "$PREP_WINDOW"; then
      tmux -S "$TMUX_SOCKET" capture-pane -pt "$target" -S -160 || true
      fail "sequential download/image window exited before completion; check tmux window: $PREP_WINDOW"
    fi
    wait_count=$((wait_count + 1))
    if [ "$wait_count" -eq 1 ] || [ $((wait_count % 6)) -eq 0 ]; then
      log "still downloading/preparing image; attach: sudo tmux -S $TMUX_SOCKET attach -t $TMUX_SESSION ; window: $PREP_WINDOW ; recent log: sudo tmux -S $TMUX_SOCKET capture-pane -pt $target -S -120"
    fi
    sleep 10
  done

  log "model download and SGLang image validation completed sequentially"
}

########################################
# 7. 启动 SGLang serve，可重复执行
########################################

server_ready() {
  curl -fsS "http://127.0.0.1:${SGLANG_PORT}/health" >"$RUNTIME_DIR/health.json" 2>"$RUNTIME_DIR/health.err" ||   curl -fsS "http://127.0.0.1:${SGLANG_PORT}/get_model_info" >"$RUNTIME_DIR/model_info.json" 2>"$RUNTIME_DIR/model_info.err" ||   curl -fsS "http://127.0.0.1:${SGLANG_PORT}/v1/models" >"$RUNTIME_DIR/models.json" 2>"$RUNTIME_DIR/models.err"
}

write_serve_script() {
  cat > "${GENERATED_SCRIPT_DIR}/glm51_serve.sh" <<'SERVE'
#!/usr/bin/env bash
set -Eeuo pipefail

: "${SGLANG_CONTAINER:?SGLANG_CONTAINER empty}"
: "${HOST_WORKDIR:?HOST_WORKDIR empty}"
: "${CONTAINER_WORKDIR:?CONTAINER_WORKDIR empty}"
: "${SGLANG_IMAGE:?SGLANG_IMAGE empty}"
: "${MODEL_DIR:?MODEL_DIR empty}"
: "${SGLANG_PORT:?SGLANG_PORT empty}"
: "${SERVED_MODEL_NAME:?SERVED_MODEL_NAME empty}"
: "${SGLANG_SERVE_ARGS:?SGLANG_SERVE_ARGS empty}"

persist_root_writable() {
  local dir="$1" test_file
  [ -n "$dir" ] || return 1
  [ -d "$dir" ] || return 1
  test_file="${dir}/.glm51-sglang-log-rw.$$"
  touch "$test_file" 2>/dev/null || return 1
  rm -f "$test_file" 2>/dev/null || true
}

persist_dir_writable() {
  local dir="$1" test_file
  [ -n "$dir" ] || return 1
  mkdir -p "$dir" 2>/dev/null || return 1
  test_file="${dir}/.glm51-sglang-log-rw.$$"
  touch "$test_file" 2>/dev/null || return 1
  rm -f "$test_file" 2>/dev/null || true
}

choose_persist_log_dir() {
  local candidate
  if [ -n "${PERSIST_LOG_DIR:-}" ]; then
    printf '%s
' "$PERSIST_LOG_DIR"
    return 0
  fi
  if [ -n "${PERSIST_LOG_ROOT:-}" ] && persist_root_writable "$PERSIST_LOG_ROOT"; then
    printf '%s/glm51-logs
' "$PERSIST_LOG_ROOT"
    return 0
  fi
  for candidate in "${CONTROL_PLANE_DIR:-}" /data /data2 /mnt /opt/glm51; do
    if persist_root_writable "$candidate"; then
      printf '%s/glm51-logs
' "$candidate"
      return 0
    fi
  done
  printf '%s
' "/opt/glm51/logs"
}

safe_experiment_name() {
  local raw safe
  raw="${1:-experiment}"
  safe="${raw//[^A-Za-z0-9_.-]/-}"
  [ -n "$safe" ] || safe="experiment"
  printf '%s
' "$safe"
}

setup_sglang_logging() {
  local safe_name tz_label ts
  export EXPERIMENT_NAME="${EXPERIMENT_NAME:-glm51-fp8-atom-pr355-oot}"
  export LOG_TIMEZONE="${LOG_TIMEZONE:-Asia/Shanghai}"
  PERSIST_LOG_DIR="$(choose_persist_log_dir)"
  if ! persist_dir_writable "$PERSIST_LOG_DIR"; then
    PERSIST_LOG_DIR="/opt/glm51/logs"
    mkdir -p "$PERSIST_LOG_DIR" 2>/dev/null || true
  fi
  PERSIST_LOG_ROOT="${PERSIST_LOG_ROOT:-$(dirname "$PERSIST_LOG_DIR")}"
  safe_name="$(safe_experiment_name "$EXPERIMENT_NAME")"
  tz_label="$(printf '%s' "$LOG_TIMEZONE" | tr -cd 'A-Za-z0-9')"
  [ -n "$tz_label" ] || tz_label="UTC"
  if [ -z "${RUN_ID:-}" ]; then
    if ts="$(TZ="$LOG_TIMEZONE" date '+%Y%m%d-%H%M%S' 2>/dev/null)"; then
      RUN_ID="${ts}-${tz_label}-${safe_name}"
    else
      ts="$(date -u '+%Y%m%d-%H%M%S')"
      RUN_ID="${ts}-UTC-${safe_name}"
    fi
  fi
  RUN_DIR="${PERSIST_LOG_DIR}/runs/${RUN_ID}"
  mkdir -p "$RUN_DIR" 2>/dev/null || true
  ln -sfn "runs/${RUN_ID}" "${PERSIST_LOG_DIR}/latest" 2>/dev/null || true
  SGLANG_LOG="${PERSIST_LOG_DIR}/sglang-serve.log"
  RUN_SGLANG_LOG="${RUN_DIR}/sglang-serve.log"
  export PERSIST_LOG_ROOT PERSIST_LOG_DIR RUN_ID RUN_DIR SGLANG_LOG RUN_SGLANG_LOG
  exec > >(tee -a "$SGLANG_LOG" "$RUN_SGLANG_LOG") 2>&1
  echo "[$(TZ="$LOG_TIMEZONE" date '+%Y-%m-%d %H:%M:%S %Z' 2>/dev/null || date '+%Y-%m-%d %H:%M:%S')] [sglang-serve] experiment_name=${EXPERIMENT_NAME} run_id=${RUN_ID} image=${SGLANG_IMAGE} model_dir=${MODEL_DIR} port=${SGLANG_PORT} persist_log_dir=${PERSIST_LOG_DIR} run_dir=${RUN_DIR}"
}

setup_sglang_logging
cd "$HOST_WORKDIR"
sudo docker rm -f "$SGLANG_CONTAINER" >/dev/null 2>&1 || true

# SGLANG_SERVE_ARGS is a shell-style argv block from the top of glm51_resume.sh.
# It intentionally supports "$MODEL_DIR" / "$SGLANG_PORT" expansion.
eval "set -- ${SGLANG_SERVE_ARGS}"

ATOM_DOCKER_MOUNT_ARGS=()
if [ -n "${ATOM_REPO_HOST:-}" ]; then
  [ -d "$ATOM_REPO_HOST" ] || { echo "ERROR: ATOM_REPO_HOST is not a directory: $ATOM_REPO_HOST" >&2; exit 1; }
  [ -f "${ATOM_REPO_HOST}/atom/plugin/sglang/models/__init__.py" ] || { echo "ERROR: ATOM SGLang plugin entry missing under ATOM_REPO_HOST: ${ATOM_REPO_HOST}/atom/plugin/sglang/models/__init__.py" >&2; exit 1; }
  ATOM_DOCKER_MOUNT_ARGS=(-v "${ATOM_REPO_HOST}:${ATOM_REPO_CONTAINER}:ro")
  case ":${ATOM_PLUGIN_PYTHONPATH:-}:" in
    *":${ATOM_REPO_CONTAINER}:"*) ;;
    *) ATOM_PLUGIN_PYTHONPATH="${ATOM_PLUGIN_PYTHONPATH:+${ATOM_PLUGIN_PYTHONPATH}:}${ATOM_REPO_CONTAINER}" ;;
  esac
else
  echo "[sglang-serve] warning: ATOM_REPO_HOST is empty; expecting ATOM PR355 to be built into SGLANG_IMAGE=${SGLANG_IMAGE}" >&2
fi
[ -n "${ATOM_PLUGIN_PYTHONPATH:-}" ] || { echo "ERROR: ATOM_PLUGIN_PYTHONPATH is empty for PR355 OOT ATOM plugin" >&2; exit 1; }
export ATOM_PLUGIN_PYTHONPATH

SGLANG_PATCH_MOUNT_ARGS=()
prepare_sglang_quark_loader_patch() {
  [ "${PATCH_SGLANG_QUARK_FUSED_QKV_A_PROJ:-1}" = "1" ] || return 0

  local patch_root_host patch_pkg_host patch_loader_host patch_root_container patch_loader_container loader_container_path
  patch_root_host="${GENERATED_SCRIPT_DIR:-${HOST_WORKDIR}}/sglang-runtime-patches/quark-loader-no-fused-a-proj"
  patch_pkg_host="${patch_root_host}/sglang/srt/model_loader"
  patch_loader_host="${patch_pkg_host}/loader.py"
  patch_root_container="/glm51-sglang-patch"
  patch_loader_container="${patch_root_container}/sglang/srt/model_loader/loader.py"
  loader_container_path="/sgl-workspace/sglang/python/sglang/srt/model_loader/loader.py"

  sudo install -d -m 0755 -o "$(id -u)" -g "$(id -g)" "$patch_pkg_host"

  echo "[sglang-serve] creating runtime patch for quark fused_qkv_a_proj_with_mqa loader mapping"
  sudo docker run --rm -i \
    -v "${patch_root_host}:${patch_root_container}" \
    --entrypoint python3 \
    "$SGLANG_IMAGE" \
    - "$patch_loader_container" <<'PY'
from pathlib import Path
import re
import sys

dst = Path(sys.argv[1])
src = Path("/sgl-workspace/sglang/python/sglang/srt/model_loader/loader.py")
if not src.is_file():
    raise SystemExit(f"SGLang loader source not found: {src}")
text = src.read_text()
old = '''                "fused_qkv_a_proj_with_mqa": ["q_a_proj", "kv_a_proj_with_mqa"],
'''
new = '''                # GLM models already fuse q_a_proj/kv_a_proj_with_mqa in
                # their model-specific load_weights() path. The generic quark
                # packed mapping also rewrites *_weight_scale_inv keys to a
                # nonexistent fused scale parameter, so leave this pair to the
                # model loader.
'''
if old not in text:
    raise SystemExit("expected quark fused_qkv_a_proj_with_mqa mapping was not found")
text = text.replace(old, new, 1)
if '"gate_up_proj": ["gate_proj", "up_proj"],' not in text:
    raise SystemExit("gate_up_proj quark mapping missing after patch")
if old in text:
    raise SystemExit("fused_qkv_a_proj_with_mqa mapping still present in quark update")
dst.write_text(text)
print(f"patched_loader={dst}")
PY

  SGLANG_PATCH_MOUNT_ARGS=(-v "${patch_loader_host}:${loader_container_path}:ro")
}

prepare_sglang_quark_loader_patch

echo "[sglang-serve] checking ATOM external package import before starting serve"
if ! sudo docker run --rm -i   --network host   --ipc host   --device /dev/kfd   --device /dev/dri   --group-add video   --cap-add SYS_PTRACE   --security-opt seccomp=unconfined   --security-opt label=disable   -v "${HOST_WORKDIR}:${CONTAINER_WORKDIR}"   "${ATOM_DOCKER_MOUNT_ARGS[@]}"   "${SGLANG_PATCH_MOUNT_ARGS[@]}"   -e "PYTHONPATH=${ATOM_PLUGIN_PYTHONPATH}"   -e "SGLANG_EXTERNAL_MODEL_PACKAGE=${SGLANG_EXTERNAL_MODEL_PACKAGE}"   --entrypoint python3   "$SGLANG_IMAGE" - <<'PY'
import importlib
import os
import sglang
import sglang.srt.model_loader.loader as loader
plugin = os.environ['SGLANG_EXTERNAL_MODEL_PACKAGE']
module = importlib.import_module(plugin)
loader_path = getattr(loader, '__file__', '')
loader_text = open(loader_path, encoding='utf-8').read()
bad_mapping = '                "fused_qkv_a_proj_with_mqa": ["q_a_proj", "kv_a_proj_with_mqa"],\n'
if bad_mapping in loader_text:
    raise SystemExit('SGLang generic quark fused_qkv_a_proj_with_mqa mapping is still active')
print('sglang', getattr(sglang, '__version__', 'unknown'), getattr(sglang, '__file__', ''))
print('sglang_loader', loader_path)
print('generic_quark_fused_qkv_a_proj_with_mqa_mapping', 'disabled')
print('PYTHONPATH', os.environ.get('PYTHONPATH', ''))
print('external_model_package', plugin, getattr(module, '__file__', ''))
PY
then
  echo "ERROR: cannot import ${SGLANG_EXTERNAL_MODEL_PACKAGE}. Check ATOM_REPO_URL/ATOM_REF/ATOM_REPO_HOST sync logs, or use a custom SGLang image with ATOM PR355." >&2
  exit 1
fi

exec sudo docker run --rm   --name "$SGLANG_CONTAINER"   --network host   --ipc host   --device /dev/kfd   --device /dev/dri   --group-add video   --cap-add SYS_PTRACE   --security-opt seccomp=unconfined   --security-opt label=disable   --shm-size 16G   --ulimit memlock=-1   --ulimit stack=67108864   -v "${HOST_WORKDIR}:${CONTAINER_WORKDIR}"   "${ATOM_DOCKER_MOUNT_ARGS[@]}"   "${SGLANG_PATCH_MOUNT_ARGS[@]}"   -e "HF_TOKEN=${HF_TOKEN:-}"   -e "HF_HOME=${CONTAINER_WORKDIR}/hf-cache"   -e "SGLANG_EXTERNAL_MODEL_PACKAGE=${SGLANG_EXTERNAL_MODEL_PACKAGE}"   -e "PYTHONPATH=${ATOM_PLUGIN_PYTHONPATH}"   -e "AITER_QUICK_REDUCE_QUANTIZATION=${AITER_QUICK_REDUCE_QUANTIZATION}"   -e "SGLANG_AITER_FP8_PREFILL_ATTN=${SGLANG_AITER_FP8_PREFILL_ATTN}"   --entrypoint bash   "$SGLANG_IMAGE"   -lc 'exec "$@"'   sglang-entry   "$@"
SERVE
  chmod +x "${GENERATED_SCRIPT_DIR}/glm51_serve.sh"
}

start_server() {
  load_active_model_env
  if server_ready; then
    log "server already responding on port $SGLANG_PORT"
    cat "$RUNTIME_DIR/health.json" "$RUNTIME_DIR/model_info.json" "$RUNTIME_DIR/models.json" 2>/dev/null | head || true
    return
  fi

  sync_atom_repo_if_needed

  log "start SGLang foreground in tmux window: $SERVE_WINDOW"
  mkdir -p "$TMUX_TMPDIR"
  chmod 700 "$TMUX_TMPDIR" || true
  write_serve_script

  if ! tmux -S "$TMUX_SOCKET" has-session -t "$TMUX_SESSION" 2>/dev/null; then
    tmux -S "$TMUX_SOCKET" new-session -d -s "$TMUX_SESSION" -c "$HOST_WORKDIR"
  fi

  # New windows inherit tmux's session environment, so update it before launching SGLang.
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" SGLANG_IMAGE "$SGLANG_IMAGE"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" GLM51_SCRIPT_VERSION "$GLM51_SCRIPT_VERSION"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" SGLANG_CONTAINER "$SGLANG_CONTAINER"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" CONTROL_DIR "$CONTROL_DIR"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" GENERATED_SCRIPT_DIR "$GENERATED_SCRIPT_DIR"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" HOST_WORKDIR "$HOST_WORKDIR"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" CONTAINER_WORKDIR "$CONTAINER_WORKDIR"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" MODEL_REVISION "$MODEL_REVISION"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" MODEL_DIR "$MODEL_DIR"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" SGLANG_PORT "$SGLANG_PORT"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" SERVED_MODEL_NAME "$SERVED_MODEL_NAME"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" RUN_TINKER_MERGE_QUANT "$RUN_TINKER_MERGE_QUANT"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" LORA_DIR "$LORA_DIR"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" MERGED_MODEL_DIR "$MERGED_MODEL_DIR"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" QUANT_MODEL_DIR "$QUANT_MODEL_DIR"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" QUANT_SCHEME "$QUANT_SCHEME"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" GLM5_REFERENCE_SCRIPTS_DIR "$GLM5_REFERENCE_SCRIPTS_DIR"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" GLM5_MERGE_QUANT_GPUS "$GLM5_MERGE_QUANT_GPUS"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" GLM5_MERGE_QUANT_WORKERS_PER_GPU "$GLM5_MERGE_QUANT_WORKERS_PER_GPU"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" GLM5_LOCAL_QUANT_ROOT "$GLM5_LOCAL_QUANT_ROOT"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" ATOM_REPO_URL "$ATOM_REPO_URL"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" ATOM_REF "$ATOM_REF"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" ATOM_REPO_ROOT "$ATOM_REPO_ROOT"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" ATOM_REPO_HOST "$ATOM_REPO_HOST"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" ATOM_REPO_CONTAINER "$ATOM_REPO_CONTAINER"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" ATOM_PLUGIN_PYTHONPATH "$ATOM_PLUGIN_PYTHONPATH"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" SGLANG_EXTERNAL_MODEL_PACKAGE "$SGLANG_EXTERNAL_MODEL_PACKAGE"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" PATCH_SGLANG_QUARK_FUSED_QKV_A_PROJ "$PATCH_SGLANG_QUARK_FUSED_QKV_A_PROJ"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" SGLANG_SERVE_ARGS "$SGLANG_SERVE_ARGS"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" PERSIST_LOG_ROOT "${PERSIST_LOG_ROOT:-}"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" PERSIST_LOG_DIR "${PERSIST_LOG_DIR:-}"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" RUN_ID "${RUN_ID:-}"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" EXPERIMENT_NAME "${EXPERIMENT_NAME:-}"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" LOG_TIMEZONE "${LOG_TIMEZONE:-}"

  local target="${TMUX_SESSION}:${SERVE_WINDOW}"
  sudo docker rm -f "$SGLANG_CONTAINER" >/dev/null 2>&1 || true
  tmux -S "$TMUX_SOCKET" kill-window -t "$target" >/dev/null 2>&1 || true
  tmux -S "$TMUX_SOCKET" new-window -d -t "$TMUX_SESSION" -n "$SERVE_WINDOW" -c "$HOST_WORKDIR" "${GENERATED_SCRIPT_DIR}/glm51_serve.sh; exec bash"

  log "waiting for server; SGLang is running foreground in tmux window: $SERVE_WINDOW"
  for i in $(seq 1 120); do
    if server_ready; then
      log "server ready"
      cat "$RUNTIME_DIR/health.json" "$RUNTIME_DIR/model_info.json" "$RUNTIME_DIR/models.json" 2>/dev/null | head || true
      return
    fi
    if [ "$i" -eq 1 ] || [ $((i % 6)) -eq 0 ]; then
      log "server not ready yet; wait attempt $i/120; inspect: sudo tmux -S $TMUX_SOCKET capture-pane -pt $target -S -160 ; attach: sudo tmux -S $TMUX_SOCKET attach -t $TMUX_SESSION"
    fi
    if [ "$i" -gt 6 ] && ! sudo docker ps --format '{{.Names}}' | grep -qx "$SGLANG_CONTAINER"; then
      tmux -S "$TMUX_SOCKET" capture-pane -pt "$target" -S -120 || true
      fail "server container is not running; check tmux window: $SERVE_WINDOW"
    fi
    sleep 10
  done

  tmux -S "$TMUX_SOCKET" capture-pane -pt "$target" -S -120 || true
  fail "server did not become ready in time; check tmux window: $SERVE_WINDOW"
}

########################################
# 8. 可选 lm_eval
########################################

build_lmeval_image() {
  sudo docker build -t "$LMEVAL_IMAGE" - <<'EOF'
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip \
    && python -m pip install "lm-eval[api]" "transformers>=4.56" sentencepiece protobuf tiktoken

WORKDIR /workspace
ENTRYPOINT ["bash", "-lc"]
EOF
}

write_lmeval_script() {
  cat > "${GENERATED_SCRIPT_DIR}/glm51_lmeval.sh" <<'LMEVAL'
#!/usr/bin/env bash
set -Eeuo pipefail

: "${HOST_WORKDIR:?HOST_WORKDIR empty}"
: "${CONTAINER_WORKDIR:?CONTAINER_WORKDIR empty}"
: "${LMEVAL_WORKDIR:?LMEVAL_WORKDIR empty}"
: "${LMEVAL_WORKDIR_HOST:?LMEVAL_WORKDIR_HOST empty}"
: "${LMEVAL_IMAGE:?LMEVAL_IMAGE empty}"
: "${LMEVAL_CONTAINER:?LMEVAL_CONTAINER empty}"
: "${SERVED_MODEL_NAME:?SERVED_MODEL_NAME empty}"
: "${SGLANG_PORT:?SGLANG_PORT empty}"

cd "$HOST_WORKDIR"
sudo mkdir -p "$LMEVAL_WORKDIR_HOST"
sudo chown -R "$(id -u):$(id -g)" "$LMEVAL_WORKDIR_HOST" || true
sudo docker rm -f "$LMEVAL_CONTAINER" >/dev/null 2>&1 || true

exec sudo docker run --rm \
  --name "$LMEVAL_CONTAINER" \
  --network host \
  -e "MODEL=${SERVED_MODEL_NAME}" \
  -e "SGLANG_PORT=${SGLANG_PORT}" \
  -e "HF_TOKEN=${HF_TOKEN:-}" \
  -e "HF_HOME=${CONTAINER_WORKDIR}/hf-cache" \
  -v "${HOST_WORKDIR}:${CONTAINER_WORKDIR}" \
  -w "$LMEVAL_WORKDIR" \
  "$LMEVAL_IMAGE" \
  'SAFE_MODEL="${MODEL//\//_}"; \
   lm_eval \
     --model local-completions \
     --model_args model="$MODEL",base_url="http://127.0.0.1:${SGLANG_PORT}/v1/completions",num_concurrent=256,max_retries=10,max_gen_toks=2048,max_length=1048576,timeout=60000,trust_remote_code=True \
     --batch_size auto \
     --tasks gsm8k \
     --num_fewshot 20 \
     --output_path "./results_${SAFE_MODEL}_gsm8k_numshot20_cc256" \
     --log_samples \
     | tee "lmeval_${SAFE_MODEL}_gsm8k_numshot20_cc256.log"'
LMEVAL
  chmod +x "${GENERATED_SCRIPT_DIR}/glm51_lmeval.sh"
}

run_lmeval_if_enabled() {
  if [ "$RUN_LMEVAL" != "1" ]; then
    log "skip lm_eval because RUN_LMEVAL=$RUN_LMEVAL"
    return
  fi

  log "build lm_eval image and start lm_eval in tmux window: $LMEVAL_WINDOW"
  build_lmeval_image

  sudo mkdir -p "$LMEVAL_WORKDIR_HOST"
  sudo chown -R "$(id -u):$(id -g)" "$LMEVAL_WORKDIR_HOST" || true
  write_lmeval_script

  if ! tmux -S "$TMUX_SOCKET" has-session -t "$TMUX_SESSION" 2>/dev/null; then
    tmux -S "$TMUX_SOCKET" new-session -d -s "$TMUX_SESSION" -c "$HOST_WORKDIR"
  fi

  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" HOST_WORKDIR "$HOST_WORKDIR"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" CONTAINER_WORKDIR "$CONTAINER_WORKDIR"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" LMEVAL_WORKDIR "$LMEVAL_WORKDIR"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" LMEVAL_WORKDIR_HOST "$LMEVAL_WORKDIR_HOST"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" LMEVAL_IMAGE "$LMEVAL_IMAGE"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" LMEVAL_CONTAINER "$LMEVAL_CONTAINER"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" SERVED_MODEL_NAME "$SERVED_MODEL_NAME"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" SGLANG_PORT "$SGLANG_PORT"

  local target="${TMUX_SESSION}:${LMEVAL_WINDOW}"
  sudo docker rm -f "$LMEVAL_CONTAINER" >/dev/null 2>&1 || true
  tmux -S "$TMUX_SOCKET" kill-window -t "$target" >/dev/null 2>&1 || true
  tmux -S "$TMUX_SOCKET" new-window -d -t "$TMUX_SESSION" -n "$LMEVAL_WINDOW" -c "$HOST_WORKDIR" "${GENERATED_SCRIPT_DIR}/glm51_lmeval.sh; exec bash"
  log "lm_eval started in tmux window: $LMEVAL_WINDOW"
}

########################################
# main
########################################

main() {
  check_prereqs
  cleanup_existing_containers_if_enabled
  ensure_base_image
  run_download_and_build_sequential
  load_active_model_env
  start_server
  run_lmeval_if_enabled

  log "DONE"
  echo "OpenAI endpoint: http://$(hostname -I | awk '{print $1}'):${SGLANG_PORT}/v1"
  echo "Local endpoint:  http://127.0.0.1:${SGLANG_PORT}/v1"
  print_tmux_observe_guide
}

case "${1:-all}" in
  all|"")
    main
    ;;
  download-only)
    check_prereqs
    ensure_base_image
    if tinker_merge_quant_enabled; then
      fail "download-only would skip Tinker URL conversion/merge/quant; use prepare-model, or set RUN_TINKER_MERGE_QUANT=0 for base-model-only download"
    fi
    download_model
    ;;
  prepare-model)
    check_prereqs
    ensure_base_image
    prepare_model_for_serve
    ;;
  build-only)
    check_prereqs
    ensure_base_image
    load_active_model_env
    build_image
    ;;
  serve-only)
    load_active_model_env
    start_server
    ;;
  *)
    fail "unknown mode: $1"
    ;;
esac
BASH

chmod +x "${GENERATED_SCRIPT_DIR}/glm51_resume.sh"
echo "[preflight] generated scripts written in ${GENERATED_SCRIPT_DIR}"

install_systemd_autostart_if_enabled() {
  echo "[preflight] writing control-plane bootstrap/autostart/env"
  if [ "${INSTALL_SYSTEMD_AUTOSTART:-1}" != "1" ]; then
    echo "[autostart] skip systemd install because INSTALL_SYSTEMD_AUTOSTART=$INSTALL_SYSTEMD_AUTOSTART"
    return 0
  fi

  local service_name script_path service_path env_path autostart_tmpdir autostart_tmp
  service_name="${AUTOSTART_SERVICE_NAME:-glm51-autostart}"
  script_path="${CONTROL_PLANE_DIR:-${CONTROL_DIR:-/opt/glm51}}/glm51-autostart.sh"
  env_path="${GLM51_ENV_FILE:-${CONTROL_PLANE_DIR:-${CONTROL_DIR:-/opt/glm51}}/glm51.env}"
  service_path="/etc/systemd/system/${service_name}.service"
  autostart_tmpdir="${CONTROL_PLANE_DIR:-${CONTROL_DIR:-/opt/glm51}}/tmp"
  autostart_tmp=""

  print_systemd_fallback() {
    echo "[autostart] WARNING: systemd autostart not installed/updated."
    if [ -x "$script_path" ]; then
      echo "[autostart] Manual start command:"
      printf '  sudo env GLM51_ENV_FILE=%q CONTROL_PLANE_DIR=%q %q\n' "$env_path" "${CONTROL_PLANE_DIR:-${CONTROL_DIR:-/opt/glm51}}" "$script_path"
    else
      echo "[autostart] Manual start command unavailable because autostart script was not written: $script_path"
    fi
    echo "[autostart] RISK: reboot will not auto-start this service until systemd unit is fixed."
  }

  sudo mkdir -p "${CONTROL_PLANE_DIR:-${CONTROL_DIR:-/opt/glm51}}" "$GENERATED_SCRIPT_DIR"
  if ! sudo install -d -m 0700 "$autostart_tmpdir"; then
    echo "[autostart] cannot create control-plane temp dir: $autostart_tmpdir"
    print_systemd_fallback
    return 0
  fi
  sudo chown "$(id -u):$(id -g)" "$autostart_tmpdir" 2>/dev/null || true
  export TMPDIR="$autostart_tmpdir"
  if ! touch "$TMPDIR/.glm51-tmpdir-rw.$$" 2>/dev/null; then
    echo "[autostart] control-plane temp dir is not writable: $TMPDIR"
    print_systemd_fallback
    return 0
  fi
  rm -f "$TMPDIR/.glm51-tmpdir-rw.$$" 2>/dev/null || true
  if ! autostart_tmp="$(mktemp "${autostart_tmpdir}/${service_name}.XXXXXXXXXX")"; then
    echo "[autostart] cannot create temp autostart script under $autostart_tmpdir"
    print_systemd_fallback
    return 0
  fi
  cat > "$autostart_tmp" <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail

if [ -z "${CONTROL_PLANE_DIR:-}" ]; then
  CONTROL_PLANE_DIR="${CONTROL_DIR:-}"
fi
if [ -z "${CONTROL_PLANE_DIR:-}" ]; then
  CONTROL_PLANE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
fi
export CONTROL_PLANE_DIR
export CONTROL_DIR="${CONTROL_DIR:-$CONTROL_PLANE_DIR}"

export MOUNT_POINT="${MOUNT_POINT:-/local_nvme}"
export MD_DEV="${MD_DEV:-/dev/md0}"
export HOST_WORKDIR="${HOST_WORKDIR:-${MOUNT_POINT}/amd_profiling}"
export CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-${MOUNT_POINT}/amd_profiling}"
export TMUX_TMPDIR="${TMUX_TMPDIR:-${CONTROL_PLANE_DIR}/tmux}"
export TMUX_SOCKET="${TMUX_SOCKET:-${TMUX_TMPDIR}/glm51.sock}"
export TMUX_SESSION="${TMUX_SESSION:-glm51}"
export PREP_WINDOW="${PREP_WINDOW:-prep-download-build}"
export SERVE_WINDOW="${SERVE_WINDOW:-sglang-serve}"
export LMEVAL_WINDOW="${LMEVAL_WINDOW:-lm-eval}"
export SGLANG_PORT="${SGLANG_PORT:-7777}"
export GLM51_SCRIPT_VERSION="${GLM51_SCRIPT_VERSION:-markdown-20260602-sglang-quark-loader-patch-v3.13}"
export AUTOSTART_CHECK_INTERVAL_SECONDS="${AUTOSTART_CHECK_INTERVAL_SECONDS:-60}"
export AUTOSTART_RESUME_WINDOW="autostart-resume"
export AUTOSTART_OBSERVE_WINDOW="autostart-observe"
export AUTOSTART_CREATE_OBSERVE_WINDOW="${AUTOSTART_CREATE_OBSERVE_WINDOW:-0}"
export HF_DOWNLOAD_MAX_WORKERS="${HF_DOWNLOAD_MAX_WORKERS:-8}"
export GLM51_ENV_FILE="${GLM51_ENV_FILE:-${CONTROL_PLANE_DIR}/glm51.env}"
export BOOTSTRAP_PATH="${BOOTSTRAP_PATH:-${CONTROL_PLANE_DIR}/bootstrap.sh}"
export BOOTSTRAP_TMUX_TMPDIR="${BOOTSTRAP_TMUX_TMPDIR:-${CONTROL_PLANE_DIR}/tmux}"
export BOOTSTRAP_TMUX_SOCKET="${BOOTSTRAP_TMUX_SOCKET:-${BOOTSTRAP_TMUX_TMPDIR}/glm51-bootstrap.sock}"
export BOOTSTRAP_TMUX_SESSION="glm51-bootstrap"
export BOOTSTRAP_WINDOW="bootstrap-redownload"
export BOOTSTRAP_FAILED="${BOOTSTRAP_FAILED:-${CONTROL_PLANE_DIR}/bootstrap.failed}"
export PERSIST_LOG_ROOT="${PERSIST_LOG_ROOT:-}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-glm51-fp8-atom-pr355-oot}"
export LOG_TIMEZONE="${LOG_TIMEZONE:-Asia/Shanghai}"
export RUN_ID="${RUN_ID:-}"

if [ -r "$GLM51_ENV_FILE" ]; then
  set -a
  . "$GLM51_ENV_FILE"
  set +a
fi
GLM51_SECRETS_FILE="${GLM51_SECRETS_FILE:-${CONTROL_PLANE_DIR}/secrets/glm51-secrets.env}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-${CONTROL_PLANE_DIR}/secrets/hf_token.env}"
for secret_file in "$GLM51_SECRETS_FILE" "$HF_TOKEN_FILE"; do
  if [ -r "$secret_file" ]; then
    set -a
    . "$secret_file"
    set +a
  fi
done
export GLM51_SECRETS_FILE HF_TOKEN_FILE
export ALLOW_REDOWNLOAD_ON_LOCAL_NVME_LOSS="${ALLOW_REDOWNLOAD_ON_LOCAL_NVME_LOSS:-1}"
export CONTROL_PLANE_DIR="${CONTROL_PLANE_DIR:-${CONTROL_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)}}"
export CONTROL_DIR="${CONTROL_DIR:-$CONTROL_PLANE_DIR}"
export GLM51_OPT_DIR="${GLM51_OPT_DIR:-$CONTROL_PLANE_DIR}"
control_root_writable() {
  local dir="$1" test_file
  [ -n "$dir" ] || return 1
  mkdir -p "$dir" 2>/dev/null || return 1
  test_file="${dir}/.glm51-control-rw.$$"
  touch "$test_file" 2>/dev/null || return 1
  rm -f "$test_file" 2>/dev/null || true
}
choose_control_dir() {
  local candidate
  if [ -n "${CONTROL_DIR:-}" ]; then
    if control_root_writable "$CONTROL_DIR"; then
      printf '%s
' "$CONTROL_DIR"
      return 0
    fi
    printf '[bootstrap] warning: CONTROL_DIR is not writable, falling back: %s
' "$CONTROL_DIR" >&2
  fi
  for candidate in /data/glm51-control /data2/glm51-control /opt/glm51; do
    if control_root_writable "$candidate"; then
      printf '%s
' "$candidate"
      return 0
    fi
  done
  printf '%s
' "/opt/glm51"
}
export CONTROL_DIR="$(choose_control_dir)"
export CONTROL_PLANE_DIR="$CONTROL_DIR"
export GENERATED_SCRIPT_DIR="${GENERATED_SCRIPT_DIR:-${CONTROL_DIR}/generated}"
export GLM51_SECRETS_FILE="${GLM51_SECRETS_FILE:-${CONTROL_DIR}/secrets/glm51-secrets.env}"
export HF_TOKEN_FILE="${HF_TOKEN_FILE:-${CONTROL_DIR}/secrets/hf_token.env}"
for secret_file in "$GLM51_SECRETS_FILE" "$HF_TOKEN_FILE"; do
  if [ -r "$secret_file" ]; then
    set -a
    . "$secret_file"
    set +a
  fi
done
export HF_TOKEN="${HF_TOKEN:-}"
export PERSIST_LOG_ROOT="${PERSIST_LOG_ROOT:-}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-glm51-fp8-atom-pr355-oot}"
export LOG_TIMEZONE="${LOG_TIMEZONE:-Asia/Shanghai}"
export RUN_ID="${RUN_ID:-}"
export FIRST_RUN_LOG_COPY_TIMEOUT_SECONDS="${FIRST_RUN_LOG_COPY_TIMEOUT_SECONDS:-10}"
export COPY_FIRST_RUN_LOG_TO_LOCAL_NVME="${COPY_FIRST_RUN_LOG_TO_LOCAL_NVME:-0}"

persist_root_writable() {
  local dir="$1" test_file
  [ -n "$dir" ] || return 1
  [ -d "$dir" ] || return 1
  test_file="${dir}/.glm51-persist-log-rw.$$"
  touch "$test_file" 2>/dev/null || return 1
  rm -f "$test_file" 2>/dev/null || true
}

choose_persist_log_root() {
  local candidate
  if [ -n "${PERSIST_LOG_ROOT:-}" ]; then
    if persist_root_writable "$PERSIST_LOG_ROOT"; then
      printf '%s
' "$PERSIST_LOG_ROOT"
      return 0
    fi
    printf '[autostart] warning: PERSIST_LOG_ROOT is not writable, falling back: %s
' "$PERSIST_LOG_ROOT" >&2
  fi
  for candidate in "${CONTROL_PLANE_DIR:-}" /data /data2 /mnt /opt/glm51; do
    if persist_root_writable "$candidate"; then
      printf '%s
' "$candidate"
      return 0
    fi
  done
  printf '%s
' "/opt/glm51"
}


safe_experiment_name() {
  local raw safe
  raw="${1:-experiment}"
  safe="${raw//[^A-Za-z0-9_.-]/-}"
  [ -n "$safe" ] || safe="experiment"
  printf '%s
' "$safe"
}

ensure_run_logging_metadata() {
  local safe_name tz_label ts
  safe_name="$(safe_experiment_name "${EXPERIMENT_NAME:-glm51-fp8-atom-pr355-oot}")"
  tz_label="$(printf '%s' "${LOG_TIMEZONE:-Asia/Shanghai}" | tr -cd 'A-Za-z0-9')"
  [ -n "$tz_label" ] || tz_label="UTC"
  if [ -z "${RUN_ID:-}" ]; then
    if ts="$(TZ="${LOG_TIMEZONE:-Asia/Shanghai}" date '+%Y%m%d-%H%M%S' 2>/dev/null)"; then
      RUN_ID="${ts}-${tz_label}-${safe_name}"
    else
      ts="$(date -u '+%Y%m%d-%H%M%S')"
      RUN_ID="${ts}-UTC-${safe_name}"
    fi
  fi
  RUNS_DIR="${PERSIST_LOG_DIR}/runs"
  RUN_DIR="${RUNS_DIR}/${RUN_ID}"
  mkdir -p "$RUN_DIR" 2>/dev/null || true
  ln -sfn "runs/${RUN_ID}" "${PERSIST_LOG_DIR}/latest" 2>/dev/null || true
  export EXPERIMENT_NAME LOG_TIMEZONE RUN_ID RUNS_DIR RUN_DIR
}

setup_persistent_logging() {
  local selected fallback
  fallback="${CONTROL_PLANE_DIR:-/opt/glm51}/logs"
  selected="$(choose_persist_log_root)"
  PERSIST_LOG_ROOT="$selected"
  PERSIST_LOG_DIR="${PERSIST_LOG_DIR:-${PERSIST_LOG_ROOT}/glm51-logs}"
  if ! mkdir -p "$PERSIST_LOG_DIR" 2>/dev/null || ! persist_root_writable "$PERSIST_LOG_DIR"; then
    PERSIST_LOG_ROOT="${CONTROL_PLANE_DIR:-/opt/glm51}"
    PERSIST_LOG_DIR="$fallback"
    mkdir -p "$PERSIST_LOG_DIR" 2>/dev/null || true
    printf '[autostart] warning: persistent log dir unavailable, using OS disk fallback: %s
' "$PERSIST_LOG_DIR" >&2
  fi
  export PERSIST_LOG_ROOT PERSIST_LOG_DIR
  ensure_run_logging_metadata
  AUTOSTART_LOG="${PERSIST_LOG_DIR}/autostart.log"
  RUN_AUTOSTART_LOG="${RUN_DIR}/autostart.log"
  BOOTSTRAP_LOG="${PERSIST_LOG_DIR}/bootstrap.log"
  RUN_BOOTSTRAP_LOG="${RUN_DIR}/bootstrap.log"
  export AUTOSTART_LOG RUN_AUTOSTART_LOG BOOTSTRAP_LOG RUN_BOOTSTRAP_LOG
  exec > >(tee -a "$AUTOSTART_LOG" "$RUN_AUTOSTART_LOG") 2>&1
  echo "[$(TZ="${LOG_TIMEZONE}" date '+%Y-%m-%d %H:%M:%S %Z' 2>/dev/null || date '+%Y-%m-%d %H:%M:%S')] [autostart] experiment_name=${EXPERIMENT_NAME} run_id=${RUN_ID} log_timezone=${LOG_TIMEZONE} persistent_log_root=${PERSIST_LOG_ROOT} persistent_log_dir=${PERSIST_LOG_DIR} run_dir=${RUN_DIR}"
}

setup_persistent_logging

echo "[$(TZ="${LOG_TIMEZONE}" date '+%Y-%m-%d %H:%M:%S %Z' 2>/dev/null || date '+%Y-%m-%d %H:%M:%S')] [autostart] GLM51_SCRIPT_VERSION=${GLM51_SCRIPT_VERSION}"

log() {
  printf '[%s] [autostart] %s
' "$(TZ="${LOG_TIMEZONE}" date '+%Y-%m-%d %H:%M:%S %Z' 2>/dev/null || date '+%Y-%m-%d %H:%M:%S')" "$*"
}

path_writable() {
  local dir="$1"
  [ -d "$dir" ] || return 1
  touch "$dir/.glm51-autostart-rw" 2>/dev/null || return 1
  rm -f "$dir/.glm51-autostart-rw" || true
}

mount_is_ready() {
  findmnt "$MOUNT_POINT" >/dev/null 2>&1 && path_writable "$MOUNT_POINT"
}

mount_md_if_present() {
  [ -b "$MD_DEV" ] || return 1
  mkdir -p "$MOUNT_POINT" 2>/dev/null || return 1
  mount "$MD_DEV" "$MOUNT_POINT" 2>/dev/null || true
  mount_is_ready
}

md_member_candidates() {
  local dev name type
  while read -r name type; do
    dev="/dev/${name}"
    [ "$type" = "disk" ] || continue
    [[ "$name" == nvme*n1 ]] || continue
    lsblk -nr -o TYPE "$dev" 2>/dev/null | grep -q '^part$' && continue
    mdadm --examine "$dev" >/dev/null 2>&1 || continue
    printf '%s
' "$dev"
  done < <(lsblk -dn -o NAME,TYPE 2>/dev/null)
}

assemble_existing_md() {
  local members=()
  mount_md_if_present && return 0
  mapfile -t members < <(md_member_candidates)
  [ "${#members[@]}" -gt 0 ] || return 1
  log "assembling $MD_DEV from md superblock members: ${members[*]}"
  mdadm --assemble "$MD_DEV" "${members[@]}" --run >/dev/null 2>&1 || true
  mount_md_if_present
}

ensure_local_nvme_mounted() {
  if mount_is_ready; then
    log "$MOUNT_POINT is mounted/writable"
    return 0
  fi
  if assemble_existing_md; then
    log "$MOUNT_POINT mounted/writable after targeted md assemble"
    return 0
  fi
  log "$MOUNT_POINT is not mounted/writable and no md superblock array could be assembled"
  return 1
}
endpoint_ready() {
  curl -fsS "http://127.0.0.1:${SGLANG_PORT}/health" >/dev/null 2>&1 || curl -fsS "http://127.0.0.1:${SGLANG_PORT}/get_model_info" >/dev/null 2>&1 || curl -fsS "http://127.0.0.1:${SGLANG_PORT}/v1/models" >/dev/null 2>&1
}

window_exists() {
  tmux -S "$TMUX_SOCKET" list-windows -t "$TMUX_SESSION" -F '#W' 2>/dev/null | grep -Fxq "$1"
}

window_has_live_pane() {
  local window="$1"
  tmux -S "$TMUX_SOCKET" list-panes -t "$TMUX_SESSION:$window" -F '#{pane_dead}' 2>/dev/null | grep -Fxq '0'
}

resume_activity_exists() {
  (window_exists "$AUTOSTART_RESUME_WINDOW" && window_has_live_pane "$AUTOSTART_RESUME_WINDOW") || \
    (window_exists "$PREP_WINDOW" && window_has_live_pane "$PREP_WINDOW") || \
    (window_exists "$SERVE_WINDOW" && window_has_live_pane "$SERVE_WINDOW")
}

bootstrap_activity_exists() {
  tmux -S "$BOOTSTRAP_TMUX_SOCKET" list-panes -t "$BOOTSTRAP_TMUX_SESSION:$BOOTSTRAP_WINDOW" -F '#{pane_dead}' 2>/dev/null | grep -Fxq '0'
}

observe_cmd() {
  printf 'while true; do clear 2>/dev/null || true; date -Is; echo %q; curl -fsS --max-time 5 %q || true; echo; echo %q; docker ps --format %q 2>&1 || true; echo; echo %q; tmux -S %q list-windows -t %q 2>&1 || true; echo; echo %q; sleep 30; done' \
    '[autostart-observe] endpoint:' \
    "http://127.0.0.1:${SGLANG_PORT}/health" \
    '[autostart-observe] docker:' \
    'table {{.Names}}\t{{.Status}}\t{{.Image}}' \
    '[autostart-observe] tmux:' \
    "$TMUX_SOCKET" \
    "$TMUX_SESSION" \
    "attach: sudo tmux -S $TMUX_SOCKET attach -t $TMUX_SESSION"
}

ensure_tmux_session_exists() {
  mkdir -p "$TMUX_TMPDIR" "$HOST_WORKDIR"
  chmod 700 "$TMUX_TMPDIR" || true

  if tmux -S "$TMUX_SOCKET" has-session -t "$TMUX_SESSION" 2>/dev/null; then
    return 0
  fi

  log "creating tmux session $TMUX_SESSION"
  tmux -S "$TMUX_SOCKET" new-session -d -s "$TMUX_SESSION" -n "status" -c "$HOST_WORKDIR" 'echo "[glm51] tmux session ready"; exec bash'
}

ensure_observe_window_exists() {
  if [ "${AUTOSTART_CREATE_OBSERVE_WINDOW:-0}" != "1" ]; then
    return 0
  fi
  ensure_tmux_session_exists
  if window_exists "$AUTOSTART_OBSERVE_WINDOW" && window_has_live_pane "$AUTOSTART_OBSERVE_WINDOW"; then
    return 0
  fi
  log "creating tmux window $AUTOSTART_OBSERVE_WINDOW"
  tmux -S "$TMUX_SOCKET" kill-window -t "$TMUX_SESSION:$AUTOSTART_OBSERVE_WINDOW" >/dev/null 2>&1 || true
  tmux -S "$TMUX_SOCKET" new-window -d -t "$TMUX_SESSION" -n "$AUTOSTART_OBSERVE_WINDOW" -c "$HOST_WORKDIR" "$(observe_cmd)"
}

start_bootstrap_in_tmux() {
  if [ "$ALLOW_REDOWNLOAD_ON_LOCAL_NVME_LOSS" != "1" ]; then
    log "ERROR: resume missing or local NVMe unavailable; ALLOW_REDOWNLOAD_ON_LOCAL_NVME_LOSS=$ALLOW_REDOWNLOAD_ON_LOCAL_NVME_LOSS, not rebuilding"
    return 1
  fi
  if [ ! -x "$BOOTSTRAP_PATH" ]; then
    log "ERROR: missing executable $BOOTSTRAP_PATH; rerun the markdown paste block once to install it"
    return 1
  fi
  if bootstrap_activity_exists; then
    log "bootstrap tmux activity already exists; not starting duplicate rebuild"
    return 0
  fi

  mkdir -p "$BOOTSTRAP_TMUX_TMPDIR"
  chmod 700 "$BOOTSTRAP_TMUX_TMPDIR" || true
  local cmd
  printf -v cmd 'echo "[autostart] running %s; log=%s; run_id=%s"; rm -f %q; if env RUN_ID=%q EXPERIMENT_NAME=%q LOG_TIMEZONE=%q PERSIST_LOG_ROOT=%q PERSIST_LOG_DIR=%q GLM51_SECRETS_FILE=%q HF_TOKEN_FILE=%q RUN_TINKER_MERGE_QUANT=%q GPU_LEASE_BASE_URL=%q LORA_DIR=%q MERGED_MODEL_DIR=%q QUANT_MODEL_DIR=%q QUANT_SCHEME=%q GLM5_REFERENCE_SCRIPTS_DIR=%q GLM5_MERGE_QUANT_GPUS=%q GLM5_MERGE_QUANT_WORKERS_PER_GPU=%q GLM5_LOCAL_QUANT_ROOT=%q AUTOSTART_CREATE_OBSERVE_WINDOW=%q %q; then date "+%%Y-%%m-%%d %%H:%%M:%%S" > %q; echo "[autostart] bootstrap completed"; else rc=$?; { date "+%%Y-%%m-%%d %%H:%%M:%%S"; echo "rc=$rc"; } > %q; echo "[autostart] bootstrap failed rc=$rc; see %s"; fi; exec bash' \
    "$BOOTSTRAP_PATH" "$BOOTSTRAP_LOG" "${RUN_ID:-}" "$BOOTSTRAP_FAILED" "${RUN_ID:-}" "${EXPERIMENT_NAME:-}" "${LOG_TIMEZONE:-}" "${PERSIST_LOG_ROOT:-}" "${PERSIST_LOG_DIR:-}" "${GLM51_SECRETS_FILE:-}" "${HF_TOKEN_FILE:-}" "${RUN_TINKER_MERGE_QUANT:-}" "${GPU_LEASE_BASE_URL:-}" "${LORA_DIR:-}" "${MERGED_MODEL_DIR:-}" "${QUANT_MODEL_DIR:-}" "${QUANT_SCHEME:-}" "${GLM5_REFERENCE_SCRIPTS_DIR:-}" "${GLM5_MERGE_QUANT_GPUS:-}" "${GLM5_MERGE_QUANT_WORKERS_PER_GPU:-}" "${GLM5_LOCAL_QUANT_ROOT:-}" "${AUTOSTART_CREATE_OBSERVE_WINDOW:-0}" "$BOOTSTRAP_PATH" "${CONTROL_PLANE_DIR}/bootstrap.last_success" "$BOOTSTRAP_FAILED" "$BOOTSTRAP_LOG"

  log "starting bootstrap/redownload in tmux: sudo tmux -S $BOOTSTRAP_TMUX_SOCKET attach -t $BOOTSTRAP_TMUX_SESSION"
  if tmux -S "$BOOTSTRAP_TMUX_SOCKET" has-session -t "$BOOTSTRAP_TMUX_SESSION" 2>/dev/null; then
    tmux -S "$BOOTSTRAP_TMUX_SOCKET" kill-window -t "$BOOTSTRAP_TMUX_SESSION:$BOOTSTRAP_WINDOW" >/dev/null 2>&1 || true
    tmux -S "$BOOTSTRAP_TMUX_SOCKET" new-window -d -t "$BOOTSTRAP_TMUX_SESSION" -n "$BOOTSTRAP_WINDOW" -c "$GLM51_OPT_DIR" "$cmd"
  else
    tmux -S "$BOOTSTRAP_TMUX_SOCKET" new-session -d -s "$BOOTSTRAP_TMUX_SESSION" -n "$BOOTSTRAP_WINDOW" -c "$GLM51_OPT_DIR" "$cmd"
  fi
}

start_resume_in_tmux() {
  if [ ! -x "$GENERATED_SCRIPT_DIR/glm51_resume.sh" ]; then
    log "missing generated resume $GENERATED_SCRIPT_DIR/glm51_resume.sh; invoking bootstrap path if allowed"
    start_bootstrap_in_tmux
    return $?
  fi

  ensure_tmux_session_exists

  if resume_activity_exists; then
    log "tmux activity already exists; not starting duplicate resume"
    return 0
  fi

  tmux -S "$TMUX_SOCKET" kill-window -t "$TMUX_SESSION:$AUTOSTART_RESUME_WINDOW" >/dev/null 2>&1 || true
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" PERSIST_LOG_ROOT "${PERSIST_LOG_ROOT:-}"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" PERSIST_LOG_DIR "${PERSIST_LOG_DIR:-}"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" RUN_ID "${RUN_ID:-}"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" EXPERIMENT_NAME "${EXPERIMENT_NAME:-}"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" LOG_TIMEZONE "${LOG_TIMEZONE:-}"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" GLM51_SECRETS_FILE "${GLM51_SECRETS_FILE:-}"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" HF_TOKEN_FILE "${HF_TOKEN_FILE:-}"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" RUN_TINKER_MERGE_QUANT "${RUN_TINKER_MERGE_QUANT:-}"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" GPU_LEASE_BASE_URL "${GPU_LEASE_BASE_URL:-}"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" LORA_DIR "${LORA_DIR:-}"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" MERGED_MODEL_DIR "${MERGED_MODEL_DIR:-}"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" QUANT_MODEL_DIR "${QUANT_MODEL_DIR:-}"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" QUANT_SCHEME "${QUANT_SCHEME:-}"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" GLM5_REFERENCE_SCRIPTS_DIR "${GLM5_REFERENCE_SCRIPTS_DIR:-}"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" GLM5_MERGE_QUANT_GPUS "${GLM5_MERGE_QUANT_GPUS:-}"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" GLM5_MERGE_QUANT_WORKERS_PER_GPU "${GLM5_MERGE_QUANT_WORKERS_PER_GPU:-}"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" GLM5_LOCAL_QUANT_ROOT "${GLM5_LOCAL_QUANT_ROOT:-}"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" AITER_QUICK_REDUCE_QUANTIZATION "${AITER_QUICK_REDUCE_QUANTIZATION:-}"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" SGLANG_AITER_FP8_PREFILL_ATTN "${SGLANG_AITER_FP8_PREFILL_ATTN:-}"
  tmux -S "$TMUX_SOCKET" set-environment -t "$TMUX_SESSION" PATCH_SGLANG_QUARK_FUSED_QKV_A_PROJ "${PATCH_SGLANG_QUARK_FUSED_QKV_A_PROJ:-}"
  log "creating tmux window $AUTOSTART_RESUME_WINDOW for resume"
  local resume_cmd
  printf -v resume_cmd 'echo "[autostart] running generated glm51_resume.sh: %s"; %q; exec bash' \
    "${GENERATED_SCRIPT_DIR}/glm51_resume.sh" \
    "${GENERATED_SCRIPT_DIR}/glm51_resume.sh"
  tmux -S "$TMUX_SOCKET" new-window -d -t "$TMUX_SESSION" -n "$AUTOSTART_RESUME_WINDOW" -c "$HOST_WORKDIR" "$resume_cmd"
}

main_loop() {
  while true; do
    if ensure_local_nvme_mounted; then
      ensure_observe_window_exists || true
      if endpoint_ready; then
        log "endpoint ready on port $SGLANG_PORT"
      else
        log "endpoint not ready; ensuring resume is running in tmux"
        start_resume_in_tmux || true
      fi
    else
      log "local NVMe unavailable; invoking bootstrap/redownload path if allowed"
      start_bootstrap_in_tmux || true
    fi
    sleep "$AUTOSTART_CHECK_INTERVAL_SECONDS"
  done
}

main_loop

EOF
  if ! sudo install -m 0755 "$autostart_tmp" "$script_path"; then
    echo "[autostart] cannot write control-plane autostart script: $script_path"
    rm -f "$autostart_tmp" 2>/dev/null || true
    print_systemd_fallback
    return 0
  fi
  rm -f "$autostart_tmp"
  if ! sudo test -x "$script_path"; then
    echo "[autostart] control-plane autostart script is not executable: $script_path"
    rm -f "$script_path" 2>/dev/null || true
    print_systemd_fallback
    return 0
  fi
  echo "[autostart] installed control-plane autostart script: $script_path"

  if ! command -v systemctl >/dev/null 2>&1; then
    echo "[autostart] skip systemd install because systemctl is missing"
    print_systemd_fallback
    return 0
  fi
  if ! sudo mkdir -p /etc/systemd/system 2>/dev/null; then
    echo "[autostart] skip systemd install because /etc/systemd/system is not writable"
    print_systemd_fallback
    return 0
  fi
  if ! sudo sh -c 'touch /etc/systemd/system/.glm51-autostart-write-test && rm -f /etc/systemd/system/.glm51-autostart-write-test' 2>/dev/null; then
    echo "[autostart] skip systemd install because /etc/systemd/system is not writable"
    print_systemd_fallback
    return 0
  fi

  if ! sudo tee "$service_path" >/dev/null <<EOF
[Unit]
Description=GLM51 autostart and self-heal service
Wants=network-online.target docker.service containerd.service
After=network-online.target docker.service containerd.service

[Service]
Type=simple
EnvironmentFile=-${env_path}
ExecStart=${script_path}
KillMode=process
Restart=always
RestartSec=30
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
  then
    echo "[autostart] failed to write $service_path"
    print_systemd_fallback
    return 0
  fi

  if ! sudo systemctl daemon-reload; then
    echo "[autostart] warning: systemctl daemon-reload failed; skip autostart enable"
    print_systemd_fallback
    return 0
  fi
  if ! sudo systemctl enable "${service_name}.service"; then
    echo "[autostart] warning: systemctl enable ${service_name}.service failed"
    print_systemd_fallback
    return 0
  fi
  echo "[autostart] installed and enabled ${service_name}.service"
}

install_systemd_autostart_if_enabled
echo "[preflight] writing control-plane bootstrap/autostart/env completed"

export TMUX_SOCKET="${TMUX_SOCKET:-${TMUX_TMPDIR}/glm51.sock}"
export TMUX_SESSION="${TMUX_SESSION:-glm51}"
cleanup_existing_tmux_if_enabled() {
  if [ "${AUTO_KILL_EXISTING_TMUX_SESSION:-1}" != "1" ]; then
    echo "[tmux-cleanup] skip because AUTO_KILL_EXISTING_TMUX_SESSION=${AUTO_KILL_EXISTING_TMUX_SESSION:-unset}"
    return 0
  fi

  echo "[tmux-cleanup] cleaning managed tmux session/socket only: session=${TMUX_SESSION} socket=${TMUX_SOCKET}"
  tmux -S "$TMUX_SOCKET" kill-session -t "$TMUX_SESSION" >/dev/null 2>&1 || true

  if [ -S "$TMUX_SOCKET" ] || [ -e "$TMUX_SOCKET" ]; then
    echo "[tmux-cleanup] removing stale socket: $TMUX_SOCKET"
    rm -f "$TMUX_SOCKET" 2>/dev/null || true
  fi
  if [ -e "${TMUX_SOCKET}.lock" ]; then
    echo "[tmux-cleanup] removing stale socket lock: ${TMUX_SOCKET}.lock"
    rm -f "${TMUX_SOCKET}.lock" 2>/dev/null || true
  fi
}
print_outer_tmux_observe_guide() {
  echo
  echo "[observe] tmux attach:"
  echo "  sudo tmux -S $TMUX_SOCKET attach -t $TMUX_SESSION"
  echo "[observe] list windows:"
  echo "  sudo tmux -S $TMUX_SOCKET list-windows -t $TMUX_SESSION"
  echo "[observe] capture first-run-check/bootstrap log:"
  echo "  sudo tmux -S $TMUX_SOCKET capture-pane -pt ${TMUX_SESSION}:0 -S -3000 | grep -F '[first-run-check]' || true"
  echo "  sudo tmux -S $TMUX_SOCKET capture-pane -pt ${TMUX_SESSION}:resume -S -3000 | grep -F '[first-run-check]' || true"
  echo "[observe] capture recent prep log:"
  echo "  sudo tmux -S $TMUX_SOCKET capture-pane -pt ${TMUX_SESSION}:${PREP_WINDOW:-prep-download-build} -S -200"
  echo "[observe] capture recent serve log:"
  echo "  sudo tmux -S $TMUX_SOCKET capture-pane -pt ${TMUX_SESSION}:${SERVE_WINDOW:-sglang-serve} -S -200"
  echo "  sudo tail -200 /data/glm51-control/logs/sglang-serve.log"
  echo "  sudo tail -200 /data/glm51-control/logs/latest/sglang-serve.log"
  echo "[observe] capture recent lm_eval log:"
  echo "  sudo tmux -S $TMUX_SOCKET capture-pane -pt ${TMUX_SESSION}:${LMEVAL_WINDOW:-lm-eval} -S -200"
  echo "[observe] endpoint check:"
  echo "  curl -fsS http://127.0.0.1:${SGLANG_PORT}/v1/models"
  echo "[observe] state dir:"
  echo "  ${HOST_WORKDIR}/.glm51_resume_state"
  echo
}
RESUME_TMUX_CMD='if [ -r "${RUN_FIRST_RUN_CHECK_LOG_PERSIST:-}" ]; then cat "$RUN_FIRST_RUN_CHECK_LOG_PERSIST"; elif [ -r "${FIRST_RUN_CHECK_LOG_PERSIST:-}" ]; then cat "$FIRST_RUN_CHECK_LOG_PERSIST"; fi; "${GENERATED_SCRIPT_DIR}/glm51_resume.sh"; exec bash'
if command -v tmux >/dev/null 2>&1; then
  echo "[preflight] cleaning existing tmux / starting tmux"
  cleanup_existing_tmux_if_enabled

  if tmux -S "$TMUX_SOCKET" has-session -t "$TMUX_SESSION" 2>/dev/null; then
    tmux -S "$TMUX_SOCKET" kill-window -t "${TMUX_SESSION}:${PREP_WINDOW:-prep-download-build}" >/dev/null 2>&1 || true
    tmux -S "$TMUX_SOCKET" kill-window -t "${TMUX_SESSION}:${SERVE_WINDOW:-sglang-serve}" >/dev/null 2>&1 || true
    tmux -S "$TMUX_SOCKET" kill-window -t "${TMUX_SESSION}:${LMEVAL_WINDOW:-lm-eval}" >/dev/null 2>&1 || true
    tmux -S "$TMUX_SOCKET" new-window -d -t "$TMUX_SESSION" -n "resume" -c "$HOST_WORKDIR" "$RESUME_TMUX_CMD"
  else
    tmux -S "$TMUX_SOCKET" new-session -d -s "$TMUX_SESSION" -c "$HOST_WORKDIR" "$RESUME_TMUX_CMD"
  fi
  echo "tmux session $TMUX_SESSION is running with socket $TMUX_SOCKET"
  print_outer_tmux_observe_guide
else
  echo "tmux not found; running directly in current shell"
  "${GENERATED_SCRIPT_DIR}/glm51_resume.sh"
fi
BOOTSTRAP
echo "[preflight] writing control-plane bootstrap/autostart/env"
sudo install -d -m 0755 "$GLM51_OPT_DIR"
sudo install -d -m 0755 "$GENERATED_SCRIPT_DIR"
sudo chown "$(id -u):$(id -g)" "$GENERATED_SCRIPT_DIR" 2>/dev/null || true
sudo install -m 0755 "$BOOTSTRAP_TMP" "$BOOTSTRAP_PATH"
rm -f "$BOOTSTRAP_TMP"

persist_root_writable_outer() {
  local dir="$1"
  [ -n "$dir" ] || return 1
  [ -d "$dir" ] || return 1
  sudo sh -c 'touch "$1/.glm51-persist-log-rw" && rm -f "$1/.glm51-persist-log-rw"' sh "$dir" 2>/dev/null
}

choose_persist_log_root_outer() {
  local candidate
  if [ -n "${PERSIST_LOG_ROOT:-}" ]; then
    if persist_root_writable_outer "$PERSIST_LOG_ROOT"; then
      printf '%s
' "$PERSIST_LOG_ROOT"
      return 0
    fi
    echo "[markdown] warning: PERSIST_LOG_ROOT is not writable, falling back: $PERSIST_LOG_ROOT" >&2
  fi
  for candidate in "${CONTROL_PLANE_DIR:-}" /data /data2 /mnt /opt/glm51; do
    if persist_root_writable_outer "$candidate"; then
      printf '%s
' "$candidate"
      return 0
    fi
  done
  printf '%s
' "/opt/glm51"
}

PERSIST_LOG_ROOT="$(choose_persist_log_root_outer)"
PERSIST_LOG_DIR="${PERSIST_LOG_ROOT}/glm51-logs"
sudo mkdir -p "$PERSIST_LOG_DIR"
sudo chmod 0755 "$PERSIST_LOG_DIR" 2>/dev/null || true

sudo install -m 0600 /dev/null "$GLM51_ENV_FILE"
{
  printf 'ALLOW_REDOWNLOAD_ON_LOCAL_NVME_LOSS=%q
' "$ALLOW_REDOWNLOAD_ON_LOCAL_NVME_LOSS"
  printf 'GLM51_SECRETS_FILE=%q
' "$GLM51_SECRETS_FILE"
  printf 'HF_TOKEN_FILE=%q
' "$HF_TOKEN_FILE"
  printf 'GLM51_SCRIPT_VERSION=%q
' "$GLM51_SCRIPT_VERSION"
  printf 'CONTROL_DIR=%q
' "$CONTROL_DIR"
  printf 'CONTROL_PLANE_DIR=%q
' "$CONTROL_PLANE_DIR"
  printf 'GENERATED_SCRIPT_DIR=%q
' "$GENERATED_SCRIPT_DIR"
  printf 'PERSIST_LOG_ROOT=%q
' "$PERSIST_LOG_ROOT"
  printf 'PERSIST_LOG_DIR=%q
' "$PERSIST_LOG_DIR"
  printf 'EXPERIMENT_NAME=%q
' "$EXPERIMENT_NAME"
  printf 'LOG_TIMEZONE=%q
' "$LOG_TIMEZONE"
  printf 'FIRST_RUN_LOG_COPY_TIMEOUT_SECONDS=%q
' "$FIRST_RUN_LOG_COPY_TIMEOUT_SECONDS"
  printf 'COPY_FIRST_RUN_LOG_TO_LOCAL_NVME=%q
' "$COPY_FIRST_RUN_LOG_TO_LOCAL_NVME"
  printf 'AUTOSTART_CREATE_OBSERVE_WINDOW=%q
' "$AUTOSTART_CREATE_OBSERVE_WINDOW"
  printf 'SGLANG_IMAGE=%q
' "$SGLANG_IMAGE"
  printf 'ATOM_REPO_URL=%q
' "$ATOM_REPO_URL"
  printf 'ATOM_REF=%q
' "$ATOM_REF"
  printf 'ATOM_REPO_ROOT=%q
' "$ATOM_REPO_ROOT"
  printf 'ATOM_REPO_HOST=%q
' "$ATOM_REPO_HOST"
  printf 'ATOM_REPO_CONTAINER=%q
' "$ATOM_REPO_CONTAINER"
  printf 'ATOM_PLUGIN_PYTHONPATH=%q
' "$ATOM_PLUGIN_PYTHONPATH"
  printf 'SGLANG_EXTERNAL_MODEL_PACKAGE=%q
' "$SGLANG_EXTERNAL_MODEL_PACKAGE"
  printf 'AITER_QUICK_REDUCE_QUANTIZATION=%q
' "$AITER_QUICK_REDUCE_QUANTIZATION"
  printf 'SGLANG_AITER_FP8_PREFILL_ATTN=%q
' "$SGLANG_AITER_FP8_PREFILL_ATTN"
  printf 'PATCH_SGLANG_QUARK_FUSED_QKV_A_PROJ=%q
' "$PATCH_SGLANG_QUARK_FUSED_QKV_A_PROJ"
  printf 'RUN_TINKER_MERGE_QUANT=%q
' "$RUN_TINKER_MERGE_QUANT"
  printf 'GPU_LEASE_BASE_URL=%q
' "$GPU_LEASE_BASE_URL"
  printf 'LORA_DIR=%q
' "$LORA_DIR"
  printf 'MERGED_MODEL_DIR=%q
' "$MERGED_MODEL_DIR"
  printf 'QUANT_MODEL_DIR=%q
' "$QUANT_MODEL_DIR"
  printf 'QUANT_SCHEME=%q
' "$QUANT_SCHEME"
  printf 'GLM5_REFERENCE_SCRIPTS_DIR=%q
' "$GLM5_REFERENCE_SCRIPTS_DIR"
  printf 'GLM5_MERGE_QUANT_GPUS=%q
' "$GLM5_MERGE_QUANT_GPUS"
  printf 'GLM5_MERGE_QUANT_WORKERS_PER_GPU=%q
' "$GLM5_MERGE_QUANT_WORKERS_PER_GPU"
  printf 'GLM5_LOCAL_QUANT_ROOT=%q
' "$GLM5_LOCAL_QUANT_ROOT"
} | sudo tee "$GLM51_ENV_FILE" >/dev/null
sudo chmod 0600 "$GLM51_ENV_FILE"
echo "[markdown] installed $BOOTSTRAP_PATH and persisted private env/log config $GLM51_ENV_FILE; logs=$PERSIST_LOG_DIR"
sudo env \
  GLM51_OPT_DIR="$GLM51_OPT_DIR" \
  GLM51_ENV_FILE="$GLM51_ENV_FILE" \
  CONTROL_PLANE_DIR="$CONTROL_PLANE_DIR" \
  CONTROL_DIR="$CONTROL_DIR" \
  GENERATED_SCRIPT_DIR="$GENERATED_SCRIPT_DIR" \
  GLM51_SECRETS_FILE="$GLM51_SECRETS_FILE" \
  HF_TOKEN_FILE="$HF_TOKEN_FILE" \
  BASE_IMAGE="$BASE_IMAGE" \
  GLM51_SCRIPT_VERSION="$GLM51_SCRIPT_VERSION" \
  SGLANG_IMAGE="$SGLANG_IMAGE" \
  SGLANG_CONTAINER="$SGLANG_CONTAINER" \
  LMEVAL_IMAGE="$LMEVAL_IMAGE" \
  LMEVAL_CONTAINER="$LMEVAL_CONTAINER" \
  MOUNT_POINT="$MOUNT_POINT" \
  MD_DEV="$MD_DEV" \
  RAID_DEVICES="$RAID_DEVICES" \
  NVME_DEVS="${NVME_DEVS:-}" \
  HOST_WORKDIR="$HOST_WORKDIR" \
  CONTAINER_WORKDIR="$CONTAINER_WORKDIR" \
  HOST_TMPDIR="$HOST_TMPDIR" \
  MODEL_ID="$MODEL_ID" \
  MODEL_REVISION="$MODEL_REVISION" \
  MODEL_DIR="$MODEL_DIR" \
  SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
  RUN_TINKER_MERGE_QUANT="$RUN_TINKER_MERGE_QUANT" \
  GPU_LEASE_BASE_URL="$GPU_LEASE_BASE_URL" \
  LORA_DIR="$LORA_DIR" \
  MERGED_MODEL_DIR="$MERGED_MODEL_DIR" \
  QUANT_MODEL_DIR="$QUANT_MODEL_DIR" \
  QUANT_SCHEME="$QUANT_SCHEME" \
  GLM5_REFERENCE_SCRIPTS_DIR="$GLM5_REFERENCE_SCRIPTS_DIR" \
  GLM5_MERGE_QUANT_GPUS="$GLM5_MERGE_QUANT_GPUS" \
  GLM5_MERGE_QUANT_WORKERS_PER_GPU="$GLM5_MERGE_QUANT_WORKERS_PER_GPU" \
  GLM5_LOCAL_QUANT_ROOT="$GLM5_LOCAL_QUANT_ROOT" \
  SGLANG_PORT="$SGLANG_PORT" \
  ATOM_REPO_URL="$ATOM_REPO_URL" \
  ATOM_REF="$ATOM_REF" \
  ATOM_REPO_ROOT="$ATOM_REPO_ROOT" \
  ATOM_REPO_HOST="$ATOM_REPO_HOST" \
  ATOM_REPO_CONTAINER="$ATOM_REPO_CONTAINER" \
  ATOM_PLUGIN_PYTHONPATH="$ATOM_PLUGIN_PYTHONPATH" \
  SGLANG_EXTERNAL_MODEL_PACKAGE="$SGLANG_EXTERNAL_MODEL_PACKAGE" \
  AITER_QUICK_REDUCE_QUANTIZATION="$AITER_QUICK_REDUCE_QUANTIZATION" \
  SGLANG_AITER_FP8_PREFILL_ATTN="$SGLANG_AITER_FP8_PREFILL_ATTN" \
  PATCH_SGLANG_QUARK_FUSED_QKV_A_PROJ="$PATCH_SGLANG_QUARK_FUSED_QKV_A_PROJ" \
  INSTALL_SYSTEMD_AUTOSTART="$INSTALL_SYSTEMD_AUTOSTART" \
  AUTOSTART_SERVICE_NAME="$AUTOSTART_SERVICE_NAME" \
  AUTOSTART_CHECK_INTERVAL_SECONDS="$AUTOSTART_CHECK_INTERVAL_SECONDS" \
  SGLANG_SERVE_ARGS="$SGLANG_SERVE_ARGS" \
  LMEVAL_WORKDIR="$LMEVAL_WORKDIR" \
  PERSIST_LOG_ROOT="$PERSIST_LOG_ROOT" \
  PERSIST_LOG_DIR="$PERSIST_LOG_DIR" \
  EXPERIMENT_NAME="$EXPERIMENT_NAME" \
  LOG_TIMEZONE="$LOG_TIMEZONE" \
  FIRST_RUN_LOG_COPY_TIMEOUT_SECONDS="$FIRST_RUN_LOG_COPY_TIMEOUT_SECONDS" \
  COPY_FIRST_RUN_LOG_TO_LOCAL_NVME="$COPY_FIRST_RUN_LOG_TO_LOCAL_NVME" \
  AUTOSTART_CREATE_OBSERVE_WINDOW="$AUTOSTART_CREATE_OBSERVE_WINDOW" \
  AUTO_FIX_DOCKER_CACHE="$AUTO_FIX_DOCKER_CACHE" \
  AUTO_FREE_SPACE="$AUTO_FREE_SPACE" \
  AUTO_DELETE_MODEL_IF_LOW_SPACE="$AUTO_DELETE_MODEL_IF_LOW_SPACE" \
  MIN_HOST_FREE_GB="$MIN_HOST_FREE_GB" \
  MIN_MODEL_DOWNLOAD_FREE_GB="$MIN_MODEL_DOWNLOAD_FREE_GB" \
  MIN_DOCKER_BUILD_FREE_GB="$MIN_DOCKER_BUILD_FREE_GB" \
  AUTO_KILL_STALE_DOWNLOADS="$AUTO_KILL_STALE_DOWNLOADS" \
  HF_DOWNLOAD_MAX_WORKERS="$HF_DOWNLOAD_MAX_WORKERS" \
  AUTO_KILL_EXISTING_CONTAINERS="$AUTO_KILL_EXISTING_CONTAINERS" \
  AUTO_KILL_EXISTING_TMUX_SESSION="$AUTO_KILL_EXISTING_TMUX_SESSION" \
  RUN_LMEVAL="$RUN_LMEVAL" \
  TMUX_TMPDIR="$TMUX_TMPDIR" \
  TMUX_SOCKET="$TMUX_SOCKET" \
  TMUX_SESSION="$TMUX_SESSION" \
  PREP_WINDOW="$PREP_WINDOW" \
  SERVE_WINDOW="$SERVE_WINDOW" \
  LMEVAL_WINDOW="$LMEVAL_WINDOW" \
  ALLOW_REDOWNLOAD_ON_LOCAL_NVME_LOSS="$ALLOW_REDOWNLOAD_ON_LOCAL_NVME_LOSS" \
  "$BOOTSTRAP_PATH"


echo "sudo tmux -S /data/glm51-control/tmux/glm51.sock attach -t glm51"
if [ "${AUTO_ATTACH_TMUX:-0}" = "1" ]; then
  echo "AUTO_ATTACH_TMUX=1; attaching tmux now. Detach with Ctrl-b d; this does not stop tasks."
  sudo tmux -S /data/glm51-control/tmux/glm51.sock attach -t glm51 || true
else
  echo "AUTO_ATTACH_TMUX=${AUTO_ATTACH_TMUX:-0}; not attaching automatically. Use the command above when you want to observe."
fi
