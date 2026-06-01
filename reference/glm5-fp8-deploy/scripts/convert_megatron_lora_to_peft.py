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
