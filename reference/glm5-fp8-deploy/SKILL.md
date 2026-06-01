---
name: glm5-fp8-deploy
description: "Unified GLM-5.1 LoRA deployment skill for Tinker adapters: resolve adapter from PFS or GPU Lease HTTP archive, merge into BF16, export repaired FineGrainedFP8, auto-detect SGLang or vLLM, and serve through the selected backend."
---

# GLM-5.1 FP8 Deploy

Use this skill when deploying a GLM-5.1 LoRA from a `tinker://.../weights/<name>` URL or an existing adapter directory, especially on H20/Mindverse/Mint hosts where the final serving backend may be either SGLang or vLLM.

All required deploy automation lives in this skill's `scripts/` directory; do not call or load separate deployment skills for this workflow.

## Default Workflow

Run preflight first on the target host:

```bash
/Users/liuqihan/.agents/skills/glm5-fp8-deploy/scripts/preflight_glm51_deploy.sh \
  --backend auto \
  --tinker-url "$TINKER_URL"
```

Run the full merge, quant, checkpoint-validate, and serve flow:

```bash
/Users/liuqihan/.agents/skills/glm5-fp8-deploy/scripts/merge_quant_serve_glm51_fp8.sh \
  --backend auto \
  --tinker-url "$TINKER_URL"
```

For a remote host, pass the SSH command. The script copies the skill scripts to the remote host and then performs backend and adapter detection there:

```bash
/Users/liuqihan/.agents/skills/glm5-fp8-deploy/scripts/preflight_glm51_deploy.sh \
  --ssh "ssh -p 20510 root@ssh4.vast.ai" \
  --backend auto \
  --tinker-url "$TINKER_URL" \
  --base /root/.cache/huggingface/hub/models--zai-org--GLM-5.1/snapshots/26e1bd6e011feb778d25ae34b09b07074139d92d
```

```bash
/Users/liuqihan/.agents/skills/glm5-fp8-deploy/scripts/merge_quant_serve_glm51_fp8.sh \
  --ssh "ssh -p 20510 root@ssh4.vast.ai" \
  --backend auto \
  --tinker-url "$TINKER_URL" \
  --base /root/.cache/huggingface/hub/models--zai-org--GLM-5.1/snapshots/26e1bd6e011feb778d25ae34b09b07074139d92d
```

When actively operating SSH sessions yourself, use the `ssh-via-piloty` skill.

If `--base` is omitted, scripts try to auto-detect a GLM-5.1 BF16 snapshot from `GLM5_BASE`, the Mindverse PFS cache, or local HuggingFace cache paths under `/root/.cache/huggingface` and `$HOME/.cache/huggingface`. Pass `--base` when multiple snapshots exist or the host uses a custom path.

## Backend Selection

`--backend auto` detects the runtime on the target machine:

- Prefer SGLang when `python3 -m sglang.launch_server` is importable.
- Otherwise use vLLM when `/usr/local/bin/vllm` or `vllm` is available.
- Use `--backend sglang` or `--backend vllm` to force one backend.

Default serve ports:

- SGLang: `30000`
- vLLM: `8000`

SGLang launches with:

```bash
python3 -m sglang.launch_server \
  --model-path <model_path> \
  --served-model-name <model_name> \
  --host 0.0.0.0 \
  --port 30000 \
  --tp 8 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --mem-fraction-static 0.85 \
  --enable-metrics \
  --enable-mfu-metrics
```

vLLM launches with `/usr/local/bin/vllm serve`, GLM reasoning/tool parser flags, no `--enforce-eager`, and MTP speculative decoding enabled by default. When vLLM MTP is enabled, checkpoint validation also runs with `--require-mtp`.

## Adapter Resolution

For `tinker://<run-id>/weights/<name>`:

1. Check the PFS adapter cache:

```text
/vePFS-Mindverse/share/tinker_runtime_checkpoints/persistent_cache/admin/<run-id>/<name>
```

2. If PFS is missing or unavailable, download via GPU Lease HTTP archive into:

```text
/data0/glm51_adapters/<name>
```

3. Extract only deploy-time adapter and metadata files:

```text
*_adapter.pt
adapter_config.json
metadata.json
training_meta.json
```

The HTTP archive converter uses the asynchronous GPU Lease Manager transfer jobs API through `scripts/tinker_to_http_archive.py`. The scripts set the known GPU Lease base URL by default; `GPU_LEASE_API_KEY` must be provided from the environment or an ops secret store.

## Merge And Quantization Contracts

- Use a GLM-5.1 BF16 base snapshot for LoRA merge, not an already quantized FP8 model.
- Convert Megatron/MBridge adapter shards to PEFT when `adapter_model.safetensors` is absent.
- Merge locally with `/usr/bin/python3.12`.
- Export repaired FineGrainedFP8 with sparse-MoE expert shard repair.
- Validate the FP8 checkpoint before serving.
- Keep merged and local quant outputs under `/root/tmp` by default.

Useful overrides:

```bash
--base PATH
GLM5_BASE=PATH
--adapter-root PATH
--merged PATH
--fp8 PATH
--served-model-name NAME
--force-download-adapter
--force-merge
--force-quant
--skip-serve
```

## Failure Notes

Read `references/failure-analysis.md` before changing merge, quantization, checkpoint validation, or serve defaults. It records the known H20 GLM-5.1 failure modes, including sparse-MoE FP8 repair, stale safetensor keys, vLLM MTP validation, missing `/root/tmp`, and SGLang custom all-reduce CUDA graph failures.

After the server is ready, use `scripts/test_glm51_endpoint_matrix.py` against the selected port for endpoint validation.
