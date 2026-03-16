import asyncio
import json
import logging
import os
from argparse import Namespace
from copy import deepcopy
from typing import Any

from tqdm import tqdm

from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from slime.rollout.sglang_rollout import GenerateState, abort, eval_rollout
from slime.utils.async_utils import run
from slime.utils.misc import load_function
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

ENV_FIXED_PROMPT = "ECC_FIXED_PROMPT"
ENV_FIXED_MESSAGES = "ECC_FIXED_MESSAGES"
ENV_FIXED_LABEL = "ECC_FIXED_LABEL"
ENV_FIXED_METADATA = "ECC_FIXED_METADATA"


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


def _prepare_samples(
    sample_groups: list[list[Sample]],
    fixed_prompt: str | list[dict[str, Any]],
    fixed_label: str | None,
    fixed_metadata: dict[str, Any],
) -> list[list[Sample]]:
    for group in sample_groups:
        for sample in group:
            sample.prompt = deepcopy(fixed_prompt)
            sample.label = fixed_label
            merged_metadata = dict(sample.metadata) if isinstance(sample.metadata, dict) else {}
            merged_metadata.update(fixed_metadata)
            sample.metadata = merged_metadata
    return sample_groups


async def _generate_fixed_prompt_rollout_async(
    args: Namespace, rollout_id: int, data_source: Any
) -> tuple[RolloutFnTrainOutput, list[list[Sample]]]:
    state = GenerateState(args)
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
    )
    metric_gatherer = MetricGatherer()

    fixed_prompt = _load_fixed_prompt()
    fixed_label = os.getenv(ENV_FIXED_LABEL)
    fixed_metadata = _load_optional_json_dict(ENV_FIXED_METADATA)

    target_data_size = args.rollout_batch_size
    data = []
    all_data = []
    do_print = True
    pbar = tqdm(total=target_data_size * args.n_samples_per_prompt, desc="ECC rollout generation")

    while len(data) < target_data_size:
        while state.remaining_batch_size < target_data_size:
            sample_groups = data_source.get_samples(args.over_sampling_batch_size)
            sample_groups = _prepare_samples(sample_groups, fixed_prompt, fixed_label, fixed_metadata)
            state.submit_generate_tasks(sample_groups)

        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            group: list[Sample] = task.result()

            if do_print:
                sample = group[0]
                logger.info(
                    "First ECC rollout sample: %s, label: %s, reward: %s",
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

    return RolloutFnTrainOutput(samples=data, metrics=metric_gatherer.collect()), aborted_samples


def generate_rollout(
    args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    if evaluation:
        output, _ = run(eval_rollout(args, rollout_id))
        return output

    output, aborted_samples = run(_generate_fixed_prompt_rollout_async(args, rollout_id, data_source))
    data_source.add_samples(aborted_samples)
    return output
