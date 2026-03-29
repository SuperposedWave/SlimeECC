import asyncio
import hashlib
import json
import logging
import math
import os
import random
import re
import threading
from argparse import Namespace
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from slime.utils.types import Sample

logger = logging.getLogger(__name__)

ENV_FIXED_PROMPT = "ECC_FIXED_PROMPT"
ENV_FIXED_MESSAGES = "ECC_FIXED_MESSAGES"
ENV_FIXED_LABEL = "ECC_FIXED_LABEL"
ENV_FIXED_METADATA = "ECC_FIXED_METADATA"

ENV_CONTINUATION_ENABLED = "ECC_CONTINUATION_ENABLED"
ENV_CONTINUATION_RATIO = "ECC_CONTINUATION_RATIO"
ENV_HISTORY_POOL_MAX_SIZE = "ECC_HISTORY_POOL_MAX_SIZE"
ENV_HISTORY_SUCCESS_RATIO = "ECC_HISTORY_SUCCESS_RATIO"
ENV_HISTORY_DIM_ERROR_RATIO = "ECC_HISTORY_DIM_ERROR_RATIO"
ENV_HISTORY_OTHER_FAIL_RATIO = "ECC_HISTORY_OTHER_FAIL_RATIO"
ENV_CONTINUATION_INCLUDE_PROGRAM = "ECC_CONTINUATION_INCLUDE_PROGRAM"
ENV_HISTORY_MAX_ROLLOUT_AGE = "ECC_HISTORY_MAX_ROLLOUT_AGE"

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_DIMENSION_ERROR_PREFIXES = {
    "wrong_dimension",
    "wrong_length",
    "wrong_length_and_dimension",
}


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _load_fixed_prompt() -> str | list[dict[str, Any]]:
    messages = os.getenv(ENV_FIXED_MESSAGES)
    if messages:
        value = json.loads(messages)
        if not isinstance(value, list):
            raise ValueError(f"{ENV_FIXED_MESSAGES} must be a JSON list of chat messages.")
        return value

    prompt = os.getenv(ENV_FIXED_PROMPT)
    if prompt:
        return prompt

    raise ValueError(
        f"Please set either {ENV_FIXED_PROMPT} for plain-text prompts or "
        f"{ENV_FIXED_MESSAGES} for chat-format prompts."
    )


def _load_optional_json_dict(env_name: str) -> dict[str, Any]:
    value = os.getenv(env_name)
    if not value:
        return {}

    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{env_name} must be a JSON object.")
    return parsed


def _messages_to_text(messages: list[dict[str, Any]]) -> str:
    parts = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str) and content.strip():
            parts.append(content.strip())
    return "\n\n".join(parts).strip()


def _strip_code_fence(text: str) -> str:
    match = _CODE_BLOCK_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalized_program_from_sample(sample: Sample) -> str:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    steps = metadata.get("ecc_steps")
    if isinstance(steps, list):
        program = "\n".join(str(step).strip() for step in steps if str(step).strip()).strip()
        if program:
            return program
    return _strip_code_fence(sample.response)


def _base_category(value: str | None) -> str:
    if not value:
        return "unknown"
    return value[:-10] if value.endswith("_duplicate") else value


@dataclass(frozen=True)
class ContinuationConfig:
    enabled: bool = True
    continuation_ratio: float = 0.5
    history_pool_max_size: int = 256
    history_success_ratio: float = 0.4
    history_dim_error_ratio: float = 0.4
    history_other_fail_ratio: float = 0.2
    include_program: bool = True
    history_max_rollout_age: int | None = None

    @classmethod
    def from_env(cls) -> "ContinuationConfig":
        ratio = min(max(float(os.getenv(ENV_CONTINUATION_RATIO, "0.5")), 0.0), 1.0)
        history_pool_max_size = max(int(os.getenv(ENV_HISTORY_POOL_MAX_SIZE, "256")), 1)
        success_ratio = max(float(os.getenv(ENV_HISTORY_SUCCESS_RATIO, "0.4")), 0.0)
        dim_error_ratio = max(float(os.getenv(ENV_HISTORY_DIM_ERROR_RATIO, "0.4")), 0.0)
        other_fail_ratio = max(float(os.getenv(ENV_HISTORY_OTHER_FAIL_RATIO, "0.2")), 0.0)
        age = os.getenv(ENV_HISTORY_MAX_ROLLOUT_AGE)
        return cls(
            enabled=_env_flag(ENV_CONTINUATION_ENABLED, True),
            continuation_ratio=ratio,
            history_pool_max_size=history_pool_max_size,
            history_success_ratio=success_ratio,
            history_dim_error_ratio=dim_error_ratio,
            history_other_fail_ratio=other_fail_ratio,
            include_program=_env_flag(ENV_CONTINUATION_INCLUDE_PROGRAM, True),
            history_max_rollout_age=int(age) if age else None,
        )


@dataclass(frozen=True)
class ContinuationSeed:
    key: str
    rollout_id: int
    bucket: str
    category: str
    message: str | None
    program: str
    target_n: int | None
    target_k: int | None
    target_distance: int | None
    actual_n: int | None
    actual_k: int | None
    min_distance: int | None
    reward_score: float | None
    rollout_correct: bool


class ContinuationHistoryPool:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, ContinuationSeed] = OrderedDict()

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    def counts_by_bucket(self) -> dict[str, int]:
        with self._lock:
            counts = {"success": 0, "dim_error": 0, "other_fail": 0}
            for seed in self._entries.values():
                counts[seed.bucket] = counts.get(seed.bucket, 0) + 1
            return counts

    def record_samples(self, samples: list[Sample], rollout_id: int, config: ContinuationConfig) -> int:
        added = 0
        with self._lock:
            for sample in samples:
                if sample.status != Sample.Status.COMPLETED:
                    continue
                seed = build_continuation_seed(sample, rollout_id)
                if seed is None:
                    continue
                if seed.key in self._entries:
                    self._entries.move_to_end(seed.key)
                    continue
                self._entries[seed.key] = seed
                added += 1
                while len(self._entries) > config.history_pool_max_size:
                    self._entries.popitem(last=False)
        return added

    def sample_seed(
        self, config: ContinuationConfig, rollout_id: int, rng: random.Random
    ) -> ContinuationSeed | None:
        with self._lock:
            candidates = list(self._entries.values())

        if config.history_max_rollout_age is not None:
            candidates = [
                seed for seed in candidates if rollout_id - seed.rollout_id <= config.history_max_rollout_age
            ]
        if not candidates:
            return None

        buckets = {
            "success": [seed for seed in candidates if seed.bucket == "success"],
            "dim_error": [seed for seed in candidates if seed.bucket == "dim_error"],
            "other_fail": [seed for seed in candidates if seed.bucket == "other_fail"],
        }
        bucket = _choose_bucket(
            buckets=buckets,
            desired_ratios={
                "success": config.history_success_ratio,
                "dim_error": config.history_dim_error_ratio,
                "other_fail": config.history_other_fail_ratio,
            },
            rng=rng,
        )
        if bucket is None:
            return None
        ranked = _rank_bucket_candidates(bucket, buckets[bucket])
        top_k = ranked[: min(8, len(ranked))]
        return rng.choice(top_k) if top_k else None


def _bucket_for_seed(category: str, rollout_correct: bool) -> str:
    if rollout_correct:
        return "success"
    if _base_category(category) in _DIMENSION_ERROR_PREFIXES:
        return "dim_error"
    return "other_fail"


def build_continuation_seed(sample: Sample, rollout_id: int) -> ContinuationSeed | None:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    reward = sample.reward if isinstance(sample.reward, dict) else {}
    ecc_reward = metadata.get("ecc_tool_reward")
    reward_dict = ecc_reward if isinstance(ecc_reward, dict) else reward

    category = str(reward_dict.get("category") or metadata.get("ecc_category") or "unknown")
    program = _normalized_program_from_sample(sample)
    if not program:
        return None
    if bool(reward_dict.get("duplicate", False)):
        return None

    target_n = _safe_int(metadata.get("n"))
    target_k = _safe_int(metadata.get("k"))
    target_distance = _safe_int(
        metadata.get("target_min_distance", metadata.get("d", reward_dict.get("target_distance")))
    )
    actual_n = _safe_int(reward_dict.get("n", metadata.get("ecc_actual_n")))
    actual_k = _safe_int(reward_dict.get("k", metadata.get("ecc_actual_k")))
    min_distance = _safe_int(reward_dict.get("min_distance", metadata.get("ecc_min_distance")))
    reward_score = _safe_float(reward_dict.get("score", reward_dict.get("raw_score")))
    rollout_correct = bool(metadata.get("rollout_correct", False))
    message = reward_dict.get("message") or metadata.get("ecc_message") or metadata.get("ecc_error")
    key = "program:" + hashlib.sha256(program.encode("utf-8")).hexdigest()

    return ContinuationSeed(
        key=key,
        rollout_id=rollout_id,
        bucket=_bucket_for_seed(category, rollout_correct),
        category=category,
        message=str(message) if message is not None else None,
        program=program,
        target_n=target_n,
        target_k=target_k,
        target_distance=target_distance,
        actual_n=actual_n,
        actual_k=actual_k,
        min_distance=min_distance,
        reward_score=reward_score,
        rollout_correct=rollout_correct,
    )


def _choose_bucket(
    buckets: dict[str, list[ContinuationSeed]], desired_ratios: dict[str, float], rng: random.Random
) -> str | None:
    available = {name: items for name, items in buckets.items() if items}
    if not available:
        return None
    total = sum(desired_ratios.get(name, 0.0) for name in available)
    if total <= 0:
        return rng.choice(sorted(available))

    threshold = rng.random()
    cumulative = 0.0
    for name in ("success", "dim_error", "other_fail"):
        if name not in available:
            continue
        cumulative += desired_ratios.get(name, 0.0) / total
        if threshold <= cumulative:
            return name
    return next(iter(available))


def _rank_bucket_candidates(bucket: str, seeds: list[ContinuationSeed]) -> list[ContinuationSeed]:
    if bucket == "success":
        return sorted(
            seeds,
            key=lambda seed: (
                seed.min_distance if seed.min_distance is not None else -1,
                seed.reward_score if seed.reward_score is not None else -1.0,
                seed.rollout_id,
            ),
            reverse=True,
        )
    return sorted(
        seeds,
        key=lambda seed: (
            seed.reward_score if seed.reward_score is not None else -1.0,
            seed.rollout_id,
        ),
        reverse=True,
    )


def assign_group_sources(
    num_groups: int,
    continuation_ratio: float,
    continuation_available: bool,
    rollout_id: int,
) -> list[str]:
    if num_groups <= 0:
        return []
    if not continuation_available or continuation_ratio <= 0.0:
        return ["cold"] * num_groups
    if continuation_ratio >= 1.0:
        return ["continuation"] * num_groups

    desired = num_groups * continuation_ratio
    base = int(math.floor(desired))
    fractional = desired - base
    rng = random.Random(f"{rollout_id}:source-assignment")
    continuation_count = base + int(rng.random() < fractional)
    continuation_count = max(0, min(num_groups, continuation_count))
    indices = set(rng.sample(range(num_groups), continuation_count))
    return ["continuation" if idx in indices else "cold" for idx in range(num_groups)]


def render_continuation_prompt(base_prompt: str, seed: ContinuationSeed, include_program: bool = True) -> str:
    lines = [
        base_prompt.strip(),
        "",
        "This is a continuation attempt for the same ECC construction task.",
        "Do not restart from scratch unless the previous construction is clearly unusable.",
        "Use the previous result and feedback below to repair dimension/length issues or improve the minimum Hamming distance.",
        "Output a complete new program, not a diff.",
        "",
        "Previous feedback summary:",
        f"- Previous result category: {seed.category}",
        f"- Target (n, k): ({seed.target_n}, {seed.target_k})",
        f"- Actual (n, k): ({seed.actual_n}, {seed.actual_k})",
        f"- Minimum Hamming distance: {seed.min_distance}",
    ]
    if seed.target_distance is not None:
        lines.append(f"- Target minimum Hamming distance: {seed.target_distance}")
    if seed.message:
        lines.append(f"- Execution/validation feedback: {seed.message}")
    if include_program:
        lines.extend(
            [
                "",
                "Previous program:",
                "```python",
                seed.program.strip(),
                "```",
            ]
        )
    return "\n".join(lines).strip()


def _apply_chat_template_if_needed(
    args: Namespace, state: Any, text_prompt: str, use_chat_template: bool
) -> str:
    if not use_chat_template:
        return text_prompt
    return state.tokenizer.apply_chat_template(
        [{"role": "user", "content": text_prompt}],
        tokenize=False,
        add_generation_prompt=True,
        **(args.apply_chat_template_kwargs or {}),
    )


def _prepare_group(
    group: list[Sample],
    prompt: str,
    fixed_label: str | None,
    fixed_metadata: dict[str, Any],
    prompt_source: str,
    seed: ContinuationSeed | None,
) -> list[Sample]:
    for sample in group:
        sample.prompt = prompt
        sample.label = fixed_label
        merged_metadata = dict(sample.metadata) if isinstance(sample.metadata, dict) else {}
        merged_metadata.update(fixed_metadata)
        merged_metadata["ecc_prompt_source"] = prompt_source
        if seed is not None:
            merged_metadata.update(
                {
                    "ecc_continuation_seed_key": seed.key,
                    "ecc_continuation_seed_rollout_id": seed.rollout_id,
                    "ecc_continuation_seed_bucket": seed.bucket,
                    "ecc_continuation_seed_category": seed.category,
                    "ecc_continuation_seed_message": seed.message,
                    "ecc_continuation_seed_n": seed.actual_n,
                    "ecc_continuation_seed_k": seed.actual_k,
                    "ecc_continuation_seed_min_distance": seed.min_distance,
                }
            )
        sample.metadata = merged_metadata
    return group


def _sample_reward_dict(sample: Sample) -> dict[str, Any]:
    if isinstance(sample.reward, dict):
        return sample.reward
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    ecc_reward = metadata.get("ecc_tool_reward")
    return ecc_reward if isinstance(ecc_reward, dict) else {}


def _sample_dim_match(sample: Sample) -> bool:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    reward = _sample_reward_dict(sample)
    target_n = _safe_int(metadata.get("n"))
    target_k = _safe_int(metadata.get("k"))
    actual_n = _safe_int(reward.get("n"))
    actual_k = _safe_int(reward.get("k"))
    return target_n is not None and target_k is not None and actual_n == target_n and actual_k == target_k


def _build_rollout_metrics(samples: list[list[Sample]], pool: ContinuationHistoryPool) -> dict[str, Any]:
    cold_groups = 0
    continuation_groups = 0
    seed_bucket_counts = {"success": 0, "dim_error": 0, "other_fail": 0}
    improved_distance_groups = 0
    repaired_dimension_groups = 0
    continuation_with_seed = 0

    for group in samples:
        if not group:
            continue
        metadata = group[0].metadata if isinstance(group[0].metadata, dict) else {}
        prompt_source = metadata.get("ecc_prompt_source", "cold")
        if prompt_source == "continuation":
            continuation_groups += 1
            continuation_with_seed += int("ecc_continuation_seed_key" in metadata)
            bucket = metadata.get("ecc_continuation_seed_bucket")
            if isinstance(bucket, str):
                seed_bucket_counts[bucket] = seed_bucket_counts.get(bucket, 0) + 1
            seed_min_distance = _safe_int(metadata.get("ecc_continuation_seed_min_distance"))
            if seed_min_distance is not None:
                group_best_distance = max((_safe_int(_sample_reward_dict(s).get("min_distance")) or -1) for s in group)
                improved_distance_groups += int(group_best_distance > seed_min_distance)
            seed_n = _safe_int(metadata.get("ecc_continuation_seed_n"))
            seed_k = _safe_int(metadata.get("ecc_continuation_seed_k"))
            target_n = _safe_int(metadata.get("n"))
            target_k = _safe_int(metadata.get("k"))
            seed_dim_ok = (
                seed_n is not None and seed_k is not None and target_n == seed_n and target_k == seed_k
            )
            group_dim_ok = any(_sample_dim_match(sample) for sample in group)
            repaired_dimension_groups += int((not seed_dim_ok) and group_dim_ok)
        else:
            cold_groups += 1

    counts = pool.counts_by_bucket()
    metrics = {
        "ecc/cold_groups": cold_groups,
        "ecc/continuation_groups": continuation_groups,
        "ecc/history_pool_size": pool.size(),
        "ecc/history_pool_success": counts.get("success", 0),
        "ecc/history_pool_dim_error": counts.get("dim_error", 0),
        "ecc/history_pool_other_fail": counts.get("other_fail", 0),
        "ecc/continuation_seed_success": seed_bucket_counts.get("success", 0),
        "ecc/continuation_seed_dim_error": seed_bucket_counts.get("dim_error", 0),
        "ecc/continuation_seed_other_fail": seed_bucket_counts.get("other_fail", 0),
    }
    if continuation_with_seed > 0:
        metrics["ecc/continuation_distance_improve_rate"] = improved_distance_groups / continuation_with_seed
        metrics["ecc/continuation_dimension_repair_rate"] = repaired_dimension_groups / continuation_with_seed
    return metrics


_HISTORY_POOL = ContinuationHistoryPool()


async def _generate_continuation_rollout_async(
    args: Namespace, rollout_id: int, data_source: Any
) -> tuple[Any, list[list[Sample]]]:
    from tqdm import tqdm

    from slime.rollout.base_types import RolloutFnTrainOutput
    from slime.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
    from slime.rollout.sglang_rollout import GenerateState, abort
    from slime.utils.misc import load_function

    state = GenerateState(args)
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
    )
    metric_gatherer = MetricGatherer()
    config = ContinuationConfig.from_env()

    fixed_prompt = _load_fixed_prompt()
    fixed_label = os.getenv(ENV_FIXED_LABEL)
    fixed_metadata = _load_optional_json_dict(ENV_FIXED_METADATA)
    base_prompt_text = _messages_to_text(fixed_prompt) if isinstance(fixed_prompt, list) else fixed_prompt
    raw_base_prompt = (
        state.tokenizer.apply_chat_template(
            fixed_prompt,
            tokenize=False,
            add_generation_prompt=True,
            **(args.apply_chat_template_kwargs or {}),
        )
        if isinstance(fixed_prompt, list) and args.apply_chat_template
        else fixed_prompt
    )
    if not isinstance(raw_base_prompt, str):
        raise ValueError("Rendered ECC base prompt must be a string.")
    continuation_uses_chat_template = isinstance(fixed_prompt, list) and args.apply_chat_template

    target_data_size = args.rollout_batch_size
    data: list[list[Sample]] = []
    all_data: list[list[Sample]] = []
    do_print = True
    pbar = tqdm(total=target_data_size * args.n_samples_per_prompt, desc="ECC rollout generation")
    seed_rng = random.Random(f"{rollout_id}:seed-selection")
    source_plan = assign_group_sources(
        num_groups=target_data_size,
        continuation_ratio=config.continuation_ratio,
        continuation_available=config.enabled and _HISTORY_POOL.size() > 0,
        rollout_id=rollout_id,
    )

    while len(data) < target_data_size:
        while state.remaining_batch_size < target_data_size:
            num_groups = args.over_sampling_batch_size
            sample_groups = data_source.get_samples(num_groups)
            start_idx = len(all_data) + state.remaining_batch_size
            prepared_groups = []

            for offset, group in enumerate(sample_groups):
                source = source_plan[start_idx + offset] if start_idx + offset < len(source_plan) else "cold"
                seed = None
                prompt = raw_base_prompt
                if source == "continuation":
                    seed = _HISTORY_POOL.sample_seed(config=config, rollout_id=rollout_id, rng=seed_rng)
                    if seed is None:
                        source = "cold"
                    else:
                        prompt = _apply_chat_template_if_needed(
                            args=args,
                            state=state,
                            text_prompt=render_continuation_prompt(
                                base_prompt=base_prompt_text,
                                seed=seed,
                                include_program=config.include_program,
                            ),
                            use_chat_template=continuation_uses_chat_template,
                        )
                prepared_groups.append(
                    _prepare_group(
                        group=group,
                        prompt=prompt,
                        fixed_label=fixed_label,
                        fixed_metadata=fixed_metadata,
                        prompt_source=source,
                        seed=seed,
                    )
                )
            state.submit_generate_tasks(prepared_groups)

        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            group = task.result()

            if do_print:
                sample = group[0]
                logger.info(
                    "First ECC continuation rollout sample: %s, label: %s, reward: %s",
                    str(sample.prompt) + sample.response,
                    str(sample.label)[:100],
                    sample.reward,
                )
                do_print = False

            all_data.append(group)
            dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
            if not dynamic_filter_output.keep:
                metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
                state.remaining_batch_size -= 1
                continue

            if len(data) < target_data_size:
                data.append(group)
                pbar.update(args.n_samples_per_prompt)

    pbar.close()

    aborted_samples = await abort(args, rollout_id)

    assert len(data) == args.rollout_batch_size, f"Got {len(data)} samples, expected {args.rollout_batch_size}"
    data = sorted(data, key=lambda group: group[0].index)
    all_samples = sorted(all_data, key=lambda group: group[0].index)

    state.reset()
    if args.rollout_sample_filter_path is not None:
        filter_func = load_function(args.rollout_sample_filter_path)
        filter_func(args, data)

    if args.rollout_all_samples_process_path is not None:
        process_func = load_function(args.rollout_all_samples_process_path)
        process_func(args, all_samples, data_source)

    flat_samples = [sample for group in data for sample in group]
    _HISTORY_POOL.record_samples(flat_samples, rollout_id=rollout_id, config=config)
    metrics = metric_gatherer.collect()
    metrics.update(_build_rollout_metrics(data, _HISTORY_POOL))

    return RolloutFnTrainOutput(samples=data, metrics=metrics), aborted_samples


def generate_rollout(
    args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False
) -> Any:
    from slime.rollout.sglang_rollout import eval_rollout
    from slime.utils.async_utils import run

    if evaluation:
        output, _ = run(eval_rollout(args, rollout_id))
        return output

    output, aborted_samples = run(_generate_continuation_rollout_async(args, rollout_id, data_source))
    data_source.add_samples(aborted_samples)
    return output
