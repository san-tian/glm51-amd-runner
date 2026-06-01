# GLM-5.1 FP8 Deployment Failure Analysis

This reference records the deployment bugs that actually occurred on the H20 GLM-5.1 merge+quant path and the checkpoint invariants that must remain true before SGLang or vLLM serving. Use `SKILL.md` and `scripts/serve_glm51_fp8.sh` for the current backend selection and launch commands.

## Current Runtime Contract

- GPU: 8 x NVIDIA H20-3e
- Helper Python: `/usr/bin/python3.12` for merge, quantization, and checkpoint validation scripts
- Backend auto-detection: prefer SGLang if `python3 -m sglang.launch_server` is importable; otherwise use vLLM if `/usr/local/bin/vllm` or `vllm` is available
- SGLang serve entrypoint: `python3 -m sglang.launch_server`
- SGLang serve command:
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
- vLLM serve entrypoint: `/usr/local/bin/vllm serve`
- vLLM serve command:
```bash
/usr/local/bin/vllm serve <model_path> \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 8 \
  --trust-remote-code \
  --served-model-name <model_name> \
  --max-model-len 131072 \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice \
  --chat-template-content-format string \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```
- Do not carry vLLM-only flags into the SGLang serve path.
- Do not carry SGLang-only flags into the vLLM serve path.

## Failure Modes And Fixes

1. Merged BF16 checkpoints can have dead metadata or shard symlinks.
- Some merged outputs were not self-contained HuggingFace snapshots.
- `config.json`, `model.safetensors.index.json`, tokenizer files, or unchanged shard symlinks could point at worker-image paths.
- Fix: `quantize_glm5_finegrained_fp8_parallel.py` resolves the authoritative source from `merge_summary.json`, materializes metadata/tokenizer files, and relinks only shards marked unchanged by the merge summary. The original serial `quantize_glm5_finegrained_fp8.py` remains a fallback/reference path.

2. The adapter shared-expert key names differ from the DeepSeek/GLM runtime names.
- Adapter keys can use singular `mlp.shared_expert.*`.
- Runtime/base keys use plural `mlp.shared_experts.*`.
- Fix: the merge script rewrites singular to plural before binding LoRA tensors.

3. Sparse expert LoRA may need expansion.
- Some adapters store strided representative sparse experts instead of every routed expert.
- Fix: local merge expands those representative LoRA tensors to all matching sparse experts when the stride pattern is valid.
- Do not pass `--no-expand-sparse-experts` except for a deliberate control experiment.

4. Naive FineGrainedFP8 save output is not deployable for GLM-5.1 sparse MoE.
- The runtime expects per-expert 2D `gate_proj`, `up_proj`, `down_proj` weights plus 2D `weight_scale_inv`.
- Naive export produced malformed or overrange expert tensors in the main shards.
- Fix: rewrite all sparse-MoE expert tensors from the merged BF16 source into `model-experts-fix-layer-*` shards.

5. Redirecting the index is not enough.
- Runtime loaders can scan safetensors files that are mentioned anywhere in the index, so stale physical keys in main shards can still be loaded.
- Earlier workaround: a loader patch in `sitecustomize.py` filtered at key level.
- Current fix: physically scrub redirected and `expert_id >= n_routed_experts` keys from the main shards. The final serve path must not depend on loader patches to hide malformed checkpoint keys.

6. MTP validation is backend-specific.
- vLLM MTP serve needs `model.layers.78.*` speculative weights and `validate_glm51_fp8_checkpoint.py --require-mtp`.
- The SGLang command in this skill does not pass speculative decoding flags and does not require MTP validation by default.
- Do not use one backend's MTP assumption as proof that the other backend is broken.

7. Endpoint quirks must be separated from checkpoint corruption.
- `/v1/completions` is raw continuation and can look repetitive for chat-style prompts.
- Tool-call parser behavior can vary by runtime/API path.
- Treat those as API/parser behavior until proven otherwise. Use the full endpoint matrix plus checkpoint validator to judge deployment health.

8. Minimal remote containers may not have `/root/tmp`.
- Symptom: preflight fails during disk checks with `df: /root/tmp: No such file or directory`, even though the root filesystem has enough free space.
- Fix: create `/root/tmp` before disk checks or deployment. The preflight script should do `mkdir -p /root/tmp` before running `df -h /root/tmp`.
- Do not treat this as a model/cache problem or a disk-space failure unless the follow-up free-space check still fails.

9. SGLang custom all-reduce can fail during CUDA graph capture.
- Symptom: after a checkpoint validates and weights load, SGLang exits with `custom_all_reduce.cuh:37: CUDA error: invalid argument` and `Capture cuda graph failed`.
- Fix: retry serving only with `--disable-custom-all-reduce` while keeping the same FP8 checkpoint, `--tp 8`, GLM parser flags, `--mem-fraction-static 0.85`, and the default Prometheus metrics flags.
- If that still fails, then try smaller CUDA graph settings such as `--cuda-graph-max-bs`; do not rerun merge or quantization unless checkpoint validation failed.

10. vLLM must use the machine entrypoint and disabled Mint import patches.
- Symptom: serving through the wrong Python or patched import path can hide the actual vLLM package, break child process startup, or load unexpected patches.
- Fix: launch through `/usr/local/bin/vllm` when available, set `MINT_VLLM_REAL_PYTHON_EXECUTABLE=/usr/bin/python3.12`, set `VLLM_USE_FLASHINFER_MOE_FP8=0`, and unset `MINT_ENABLE_VLLM_IMPORT_PATCHES`.
- Do not pass `--enforce-eager` in the normal vLLM path; compile should remain enabled by omission.

11. Missing PFS is not an adapter failure.
- Symptom: `/vePFS-Mindverse/share/tinker_runtime_checkpoints/...` does not exist on a Vast or non-Mindverse host.
- Fix: convert the `tinker://` URL through the GPU Lease asynchronous HTTP archive job, download the `.tar.gz`, and extract only adapter shards plus metadata under `/data0/glm51_adapters/<name>`.
- Do not require `/vePFS-Mindverse/share` for HTTP fallback deployments. Pass `--base` when the GLM-5.1 BF16 snapshot is local to another path.

12. Missing PFS is not a base-model failure if a local HuggingFace snapshot exists.
- Symptom: a Vast host has no `/vePFS-Mindverse/share`, but the BF16 base is present under `/root/.cache/huggingface/hub/models--zai-org--GLM-5.1/snapshots/...`.
- Fix: auto-detect `GLM5_BASE`, the Mindverse PFS snapshot, and local HuggingFace cache snapshots before failing. Pass `--base` only when auto-detection is ambiguous or the snapshot is stored elsewhere.
- Do not fall back to the official FP8 model as a merge base.

## Hard Invariants

- Use the official GLM-5.1 BF16 base snapshot for LoRA merge, not the official FP8 model.
- Do offline FineGrainedFP8 export and sparse-MoE repair before serving.
- Do not rely on loader patches to hide stale checkpoint keys; the checkpoint itself must validate cleanly.
- Do not call an export complete until `fp8_quant_meta.json` exists and `validate_glm51_fp8_checkpoint.py` passes.
- Require MTP validation only when serving with vLLM MTP enabled.
- Do not use vLLM-only serve flags in SGLang.
- Do not use SGLang-only serve flags in vLLM.
- Serving-only runtime failures after a clean checkpoint validation should be fixed by changing serving flags, not by recreating merge or quant artifacts.
