"""Inspect local JSON metadata, without loading weights or executing model code."""
import argparse
import json
from pathlib import Path


def read_object(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path.name}")
    return value


def inspect_checkpoint(model, adapter=None, tp=2):
    if tp not in (1, 2, 4):
        raise ValueError("TP must be 1, 2 or 4")
    config = read_object(model / "config.json")
    text = config.get("text_config", config)
    if not isinstance(text, dict):
        raise ValueError("text_config must be an object")
    warnings = ["Metadata only; tensors, scales, visual adapter coverage and TP execution NOT validated"]
    checks = {}
    for key in ("num_attention_heads", "num_key_value_heads", "linear_num_key_heads",
                "linear_num_value_heads", "intermediate_size"):
        value = text.get(key)
        valid = type(value) is int and value > 0
        divisible = value % tp == 0 if valid else None
        checks[key] = {"value": value, "divisible": divisible}
        if divisible is not True:
            warnings.append(f"Partition requires inspection: {key}")
    quant = config.get("quantization_config", text.get("quantization_config"))
    if quant is None:
        warnings.append("FP8 format unknown: no quantization_config")
    adapter_config = read_object(adapter / "adapter_config.json") if adapter else None
    if adapter_config is not None:
        warnings.append("Check actual visual/aligner tensor keys and modules_to_save before merging")
    return {
        "runtime_ready": False, "target": "A100 SM80", "tp": tp,
        "planned_precision": "FP8 storage / BF16 compute (W8A16)",
        "architectures": config.get("architectures"), "model_type": config.get("model_type"),
        "vision_config_present": isinstance(config.get("vision_config"), dict),
        "quantization_config": quant, "partition_checks": checks,
        "adapter_config": adapter_config, "warnings": warnings,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--tp", type=int, choices=(1, 2, 4), default=2)
    args = parser.parse_args()
    try:
        report = inspect_checkpoint(args.model, args.adapter, args.tp)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Inspection failed: {exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
