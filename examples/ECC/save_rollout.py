import json
import logging
from statistics import mean
from pathlib import Path

from slime.utils.tensorboard_utils import _TensorboardAdapter
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


def _extract_min_distances(samples: list[Sample]) -> list[float]:
    distances: list[float] = []
    for sample in samples:
        reward = sample.reward if isinstance(sample.reward, dict) else {}
        value = reward.get("min_distance")
        if value is None:
            metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
            ecc_reward = metadata.get("ecc_tool_reward")
            if isinstance(ecc_reward, dict):
                value = ecc_reward.get("min_distance")
        if isinstance(value, (int, float)):
            distances.append(float(value))
    return distances


def _extract_parsed_steps(samples: list[Sample]) -> list[float]:
    parsed_steps: list[float] = []
    for sample in samples:
        reward = sample.reward if isinstance(sample.reward, dict) else {}
        value = reward.get("parsed_steps")
        if value is None:
            metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
            ecc_reward = metadata.get("ecc_tool_reward")
            if isinstance(ecc_reward, dict):
                value = ecc_reward.get("parsed_steps")
        if isinstance(value, (int, float)):
            parsed_steps.append(float(value))
    return parsed_steps


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

    if args.use_tensorboard:
        tb_metrics = {}
        min_distances = _extract_min_distances(samples)
        if min_distances:
            tb_metrics["rollout/ecc_min_distance_max"] = max(min_distances)
            tb_metrics["rollout/ecc_min_distance_mean"] = mean(min_distances)
        parsed_steps = _extract_parsed_steps(samples)
        if parsed_steps:
            tb_metrics["rollout/ecc_parsed_steps_mean"] = mean(parsed_steps)
        if tb_metrics:
            _TensorboardAdapter(args).log(tb_metrics, step=rollout_id)

    return False
