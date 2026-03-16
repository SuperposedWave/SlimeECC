import json
import logging
from pathlib import Path

from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def _reward_to_jsonable(value):
    if isinstance(value, dict):
        return {k: _reward_to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sample_to_record(sample: Sample) -> dict:
    return {
        "group_index": sample.group_index,
        "index": sample.index,
        "prompt": sample.prompt,
        "response": sample.response,
        "label": sample.label,
        "reward": _reward_to_jsonable(sample.reward),
        "response_length": sample.response_length,
        "status": sample.status.value,
        "removed": sample.remove_sample,
        "metadata": {k: _reward_to_jsonable(v) for k, v in (sample.metadata or {}).items()},
    }


def _output_dir(args) -> Path | None:
    save_dir = getattr(args, "save", None)
    if not save_dir:
        return None
    path = Path(save_dir) / "rollout" / "rollout_data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_and_save_rollout(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    """Save every rollout's samples to a JSON file.

    Wire via: --custom-rollout-log-function-path examples.ECC.save_rollout.log_and_save_rollout
    Files are written to <save>/rollout/rollout_data/rollout_{id}.json
    """
    out_dir = _output_dir(args)
    if out_dir is None:
        logger.warning("--save is not set; skipping rollout data dump")
        return False

    records = [_sample_to_record(s) for s in samples]
    payload = {
        "rollout_id": rollout_id,
        "rollout_time": rollout_time,
        "num_samples": len(records),
        "samples": records,
    }

    out_path = out_dir / f"rollout_{rollout_id}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Saved %d samples to %s", len(records), out_path)

    return False
