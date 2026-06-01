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
