import json
import logging
from pathlib import Path
from typing import Any

from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def _flatten_samples(items):
    for item in items:
        if isinstance(item, Sample):
            yield item
        elif isinstance(item, list):
            yield from _flatten_samples(item)


def _reward_to_jsonable(value: Any):
    if isinstance(value, dict):
        return {k: _reward_to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def get_sample_id(sample: Sample) -> str:
    metadata = sample.metadata or {}
    sample_id = metadata.get("sample_id", sample.index)
    return str(sample_id)


def is_sample_correct(args, sample: Sample) -> bool:
    metadata = sample.metadata or {}
    if "rollout_correct" in metadata:
        return bool(metadata["rollout_correct"])

    if sample.reward is None:
        return False

    reward_value = sample.get_reward_value(args)
    return reward_value is True or reward_value == 1


def summarize_samples(args, samples: list[Sample]) -> dict[str, Any]:
    rows = []
    correct_count = 0

    for sample in samples:
        correct = is_sample_correct(args, sample)
        correct_count += int(correct)
        rows.append(
            {
                "sample_id": get_sample_id(sample),
                "sample_index": sample.index,
                "group_index": sample.group_index,
                "correct": correct,
                "reward": _reward_to_jsonable(sample.reward),
                "status": sample.status.value,
                "removed": sample.remove_sample,
                "response_length": sample.response_length,
                "label": sample.label,
                "prompt_preview": str(sample.prompt)[:200],
                "response_preview": sample.response[:200],
            }
        )

    total = len(rows)
    accuracy = correct_count / total if total else 0.0
    return {
        "num_samples": total,
        "num_correct": correct_count,
        "accuracy": accuracy,
        "samples": rows,
    }


def _output_dir(args) -> Path | None:
    save_dir = getattr(args, "save", None)
    if not save_dir:
        return None
    path = Path(save_dir) / "rollout" / "per_sample_accuracy"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_snapshot(base_path: Path, summary: dict[str, Any]) -> None:
    summary_path = base_path.with_suffix(".summary.json")
    details_path = base_path.with_suffix(".jsonl")

    summary_payload = {
        "num_samples": summary["num_samples"],
        "num_correct": summary["num_correct"],
        "accuracy": summary["accuracy"],
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with details_path.open("w", encoding="utf-8") as f:
        for row in summary["samples"]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def process_all_samples(args, all_samples, data_source) -> None:
    samples = list(_flatten_samples(all_samples))
    for sample in samples:
        sample.metadata["rollout_correct"] = is_sample_correct(args, sample)

    out_dir = _output_dir(args)
    if out_dir is None:
        return

    summary = summarize_samples(args, samples)
    _write_snapshot(out_dir / "latest_all_samples", summary)


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    summary = summarize_samples(args, samples)

    if rollout_extra_metrics is not None:
        rollout_extra_metrics["sample_accuracy/accuracy"] = summary["accuracy"]
        rollout_extra_metrics["sample_accuracy/num_correct"] = summary["num_correct"]
        rollout_extra_metrics["sample_accuracy/num_samples"] = summary["num_samples"]

    hardest_samples = [row for row in summary["samples"] if not row["correct"]][:3]
    logger.info(
        "rollout %s sample accuracy: %.4f (%s/%s)%s",
        rollout_id,
        summary["accuracy"],
        summary["num_correct"],
        summary["num_samples"],
        f", hardest_samples={hardest_samples}" if hardest_samples else "",
    )

    out_dir = _output_dir(args)
    if out_dir is not None:
        _write_snapshot(out_dir / f"rollout_{rollout_id}", summary)

    return False
