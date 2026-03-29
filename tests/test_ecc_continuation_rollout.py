import random

import pytest

from examples.ECC.continuation_rollout import (
    ContinuationConfig,
    ContinuationHistoryPool,
    ContinuationSeed,
    assign_group_sources,
    build_continuation_seed,
    render_continuation_prompt,
)
from slime.utils.types import Sample


@pytest.mark.unit
def test_build_continuation_seed_uses_feedback_summary():
    sample = Sample(
        response="```python\nc0 = build_family('hamming_binary', r=3)\n```",
        reward={"score": 0.75, "category": "wrong_dimension", "n": 18, "k": 6, "min_distance": 4},
        metadata={
            "n": 18,
            "k": 7,
            "target_min_distance": 5,
            "rollout_correct": False,
            "ecc_steps": ["c0 = build_family('hamming_binary', r=3)"],
        },
        status=Sample.Status.COMPLETED,
    )

    seed = build_continuation_seed(sample, rollout_id=12)

    assert seed is not None
    assert seed.bucket == "dim_error"
    assert seed.program == "c0 = build_family('hamming_binary', r=3)"
    assert seed.target_n == 18
    assert seed.target_k == 7
    assert seed.actual_k == 6
    assert seed.min_distance == 4


@pytest.mark.unit
def test_history_pool_deduplicates_by_program():
    pool = ContinuationHistoryPool()
    config = ContinuationConfig(history_pool_max_size=8)
    sample = Sample(
        response="c0 = foo()",
        reward={"score": 0.5, "category": "wrong_length", "n": 17, "k": 7, "min_distance": 3},
        metadata={"n": 18, "k": 7, "rollout_correct": False},
        status=Sample.Status.COMPLETED,
    )

    added_first = pool.record_samples([sample], rollout_id=1, config=config)
    added_second = pool.record_samples([sample], rollout_id=2, config=config)

    assert added_first == 1
    assert added_second == 0
    assert pool.size() == 1


@pytest.mark.unit
def test_history_pool_prefers_success_bucket_when_only_success_exists():
    pool = ContinuationHistoryPool()
    config = ContinuationConfig(
        history_pool_max_size=8,
        history_success_ratio=1.0,
        history_dim_error_ratio=0.0,
        history_other_fail_ratio=0.0,
    )

    good = Sample(
        response="c1 = good()",
        reward={"score": 1.1, "category": "success", "n": 18, "k": 7, "min_distance": 6},
        metadata={"n": 18, "k": 7, "rollout_correct": True},
        status=Sample.Status.COMPLETED,
    )
    better = Sample(
        response="c2 = better()",
        reward={"score": 1.2, "category": "success", "n": 18, "k": 7, "min_distance": 7},
        metadata={"n": 18, "k": 7, "rollout_correct": True},
        status=Sample.Status.COMPLETED,
    )
    pool.record_samples([good, better], rollout_id=3, config=config)

    seed = pool.sample_seed(config=config, rollout_id=4, rng=random.Random(0))

    assert seed is not None
    assert seed.bucket == "success"
    assert seed.min_distance == 7


@pytest.mark.unit
def test_render_continuation_prompt_contains_feedback_and_program():
    seed = ContinuationSeed(
        key="program:test",
        rollout_id=8,
        bucket="dim_error",
        category="wrong_dimension",
        message="Expected k=7, got k=6.",
        program="c0 = build_family('hamming_binary', r=3)",
        target_n=18,
        target_k=7,
        target_distance=5,
        actual_n=18,
        actual_k=6,
        min_distance=4,
        reward_score=0.2,
        rollout_correct=False,
    )

    prompt = render_continuation_prompt("Construct the code.", seed, include_program=True)

    assert "Construct the code." in prompt
    assert "Previous result category: wrong_dimension" in prompt
    assert "Expected k=7, got k=6." in prompt
    assert "c0 = build_family('hamming_binary', r=3)" in prompt
    assert "Output a complete new program, not a diff." in prompt


@pytest.mark.unit
def test_assign_group_sources_is_deterministic_for_odd_batch():
    first = assign_group_sources(num_groups=3, continuation_ratio=0.5, continuation_available=True, rollout_id=11)
    second = assign_group_sources(num_groups=3, continuation_ratio=0.5, continuation_available=True, rollout_id=11)

    assert first == second
    assert len(first) == 3
    assert set(first) <= {"cold", "continuation"}
    assert first.count("continuation") in {1, 2}
