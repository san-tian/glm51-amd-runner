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
