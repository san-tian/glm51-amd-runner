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
