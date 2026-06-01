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
