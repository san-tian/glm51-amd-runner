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
