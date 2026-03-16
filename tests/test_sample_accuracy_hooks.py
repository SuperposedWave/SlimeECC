import json
from pathlib import Path
from types import SimpleNamespace

from slime.rollout.sample_accuracy import log_rollout_data, process_all_samples
from slime.utils.types import Sample


def make_args(tmp_path: Path):
    return SimpleNamespace(save=str(tmp_path), reward_key=None)


def make_sample(index: int, reward=1.0, *, sample_id: str | None = None):
    metadata = {}
    if sample_id is not None:
        metadata["sample_id"] = sample_id
    return Sample(
        index=index,
        group_index=index // 2,
        prompt=f"prompt-{index}",
        response=f"response-{index}",
        response_length=1,
        reward=reward,
        status=Sample.Status.COMPLETED,
        metadata=metadata,
    )


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_process_all_samples_marks_correctness_and_writes_latest_snapshot(tmp_path):
    args = make_args(tmp_path)
    kept = [make_sample(0, reward=1.0, sample_id="a"), make_sample(1, reward=0.0, sample_id="b")]
    filtered = [make_sample(2, reward=1.0, sample_id="c")]

    process_all_samples(args, [kept, filtered], data_source=object())

    assert kept[0].metadata["rollout_correct"] is True
    assert kept[1].metadata["rollout_correct"] is False
    assert filtered[0].metadata["rollout_correct"] is True

    out_dir = tmp_path / "rollout" / "per_sample_accuracy"
    summary = json.loads((out_dir / "latest_all_samples.summary.json").read_text(encoding="utf-8"))
    details = read_jsonl(out_dir / "latest_all_samples.jsonl")

    assert summary["num_samples"] == 3
    assert summary["num_correct"] == 2
    assert summary["accuracy"] == 2 / 3
    assert [row["sample_id"] for row in details] == ["a", "b", "c"]


def test_log_rollout_data_updates_metrics_and_writes_rollout_snapshot(tmp_path):
    args = make_args(tmp_path)
    samples = [make_sample(0, reward=1.0, sample_id="a"), make_sample(1, reward=0.0, sample_id="b")]
    extra_metrics = {"source": "test"}

    should_skip_default = log_rollout_data(7, args, samples, extra_metrics, rollout_time=0.3)

    assert should_skip_default is False
    assert extra_metrics["sample_accuracy/accuracy"] == 0.5
    assert extra_metrics["sample_accuracy/num_correct"] == 1
    assert extra_metrics["sample_accuracy/num_samples"] == 2

    out_dir = tmp_path / "rollout" / "per_sample_accuracy"
    summary = json.loads((out_dir / "rollout_7.summary.json").read_text(encoding="utf-8"))
    details = read_jsonl(out_dir / "rollout_7.jsonl")

    assert summary["accuracy"] == 0.5
    assert [row["correct"] for row in details] == [True, False]


def test_log_rollout_data_uses_reward_key_for_dict_rewards(tmp_path):
    args = SimpleNamespace(save=str(tmp_path), reward_key="score")
    samples = [
        make_sample(0, reward={"score": 1.0, "aux": 0.1}, sample_id="a"),
        make_sample(1, reward={"score": 0.0, "aux": 0.9}, sample_id="b"),
    ]
    extra_metrics = {}

    log_rollout_data(3, args, samples, extra_metrics, rollout_time=0.1)

    assert extra_metrics["sample_accuracy/accuracy"] == 0.5
