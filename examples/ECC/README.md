# ECC Fixed-Prompt Rollout

This example shows how to run slime without a prompt dataset, or with the same
question repeated for every rollout group.

It adds a custom `--rollout-function-path` that:

- fetches empty `Sample` objects from the normal data source
- injects one fixed prompt into every sample
- reuses slime's default async SGLang generation and reward flow

## Required environment variables

Choose one prompt format:

```bash
export ECC_FIXED_PROMPT='What is the next action?'
```

or

```bash
export ECC_FIXED_MESSAGES='[{"role":"user","content":"What is the next action?"}]'
```

Optional extras:

```bash
export ECC_FIXED_LABEL='expected answer'
export ECC_FIXED_METADATA='{"task":"ecc"}'
```

## Required rollout flags

```bash
--disable-rollout-global-dataset
--num-rollout 100
--rollout-function-path examples.ECC.fixed_prompt_rollout.generate_rollout
```

Notes:

- Do not use `--num-epoch` in this mode. Without a global dataset, slime
  requires `--num-rollout`.
- If your prompt is chat-format messages, keep `--apply-chat-template`.
- Reward computation is still controlled by your existing `--rm-type` or
  `--custom-rm-path`.

## ECC custom reward

For ECC tasks you can use:

```bash
--custom-rm-path examples.ECC.ecc_reward.custom_rm
```

The reward function expects the model response to contain a GF(2) generator
matrix in one of these forms:

```text
[[1,0,1,1],[0,1,1,0]]
```

or

```text
1011
0110
```

It then enumerates all non-zero codewords and computes the true minimum
Hamming distance.

Target distance can be provided in either place:

- `label`: a plain integer like `3`
- `metadata`: one of `target_min_distance`, `min_distance`, `minimum_distance`, `target_d`, `d`

Optional shape checks:

- `metadata["n"]`: expected code length
- `metadata["k"]`: expected generator row count

Reward behavior:

- if target distance exists: reward is `1.0` when `d_min >= target`, else `0.0`
- if target distance is missing: reward is the computed minimum distance itself
- parse failure, wrong shape, or rank-deficient generator matrix: reward is `0.0`

Diagnostics are written into `sample.metadata`, including:

- `ecc_computed_min_distance`
- `ecc_rank`
- `ecc_shape_ok`
- `ecc_rank_ok`
- `ecc_error` on failure

## ECC tool-use reward skeleton

If the model output is not a final generator matrix, but a sequence of
tool-use calls for `code_ops.py`, use:

```bash
--custom-rm-path examples.ECC.tool_reward.custom_rm
--reward-key score
--log-reward-category category
```

This reward skeleton is designed for the workflow:

1. parse model output into tool-call steps
2. whitelist-check function names
3. execute steps through your own `code_ops` dispatcher
4. verify the final code's `n`, `k`, and minimum distance

Current status:

- parsing is implemented
- whitelist validation is implemented
- actual execution is intentionally left as a stub in `execute_tool_steps()`

Expected response format is a simple Python-like sequence such as:

```python
from code_ops import build_family, puncture_code

c0 = build_family("hamming_binary", r=3)
c1 = puncture_code(c0, 6)
```

Reward categories are intended to be:

- `parse_error`
- `forbidden_function:<name>`
- `parsed_but_not_executed`
- `execution_failed`
- `distance_too_small`
- `success`

Task targets should be passed through metadata, for example:

```json
{
  "target_min_distance": 3,
  "n": 6,
  "k": 3
}
```

The reward returns a dict with at least:

```json
{
  "score": 0.0,
  "category": "parse_error"
}
```

so you should enable:

```bash
--reward-key score
--log-reward-category category
```

## Minimal example

```bash
export ECC_FIXED_PROMPT='Solve this task from the same initial question.'

python train.py \
  --disable-rollout-global-dataset \
  --num-rollout 100 \
  --rollout-batch-size 8 \
  --n-samples-per-prompt 4 \
  --rollout-function-path examples.ECC.fixed_prompt_rollout.generate_rollout \
  --rm-type deepscaler
```
