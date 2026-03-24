#!/bin/bash

# ============================================================================
# ECC Tool-Use GRPO 训练脚本
# ============================================================================

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex
export PYTHONBUFFERED=16
# export WANDB_KEY=$

TIMESTAMP=$(date +%Y%m%d%H%M%S)
PROJECT_NAME="ecc-tool-grpo"
EXP_NAME="${PROJECT_NAME}-${TIMESTAMP}"

# ============================================================================
# 路径配置
# ============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"
SLIME_DIR="$(cd -- "${PROJECT_DIR}/../.." &>/dev/null && pwd)"

MODEL_BASE="/inspire/hdd/project/qproject-multireasoning/zhouzhixiang-240107010008/Model"
MEGATRON_LM_PATH="/root/Megatron-LM/"
MODEL_ARGS_SCRIPT="/inspire/hdd/project/qproject-multireasoning/zhouzhixiang-240107010008/Project/slime/scripts/models/qwen3-4B.sh"

HF_CHECKPOINT="${MODEL_BASE}/Qwen3-4B"
TORCH_DIST_CHECKPOINT="${MODEL_BASE}/Qwen3-4B_torch_dist"
CHECKPOINT="${PROJECT_DIR}/checkpoint/${EXP_NAME}"
TENSORBOARD_DIR="${PROJECT_DIR}/tensorboard/${EXP_NAME}"
WANDB_DIR="${PROJECT_DIR}/wandb/${EXP_NAME}"
PROMPT_FILE="${PROJECT_DIR}/llm_tool_prompt.txt"

mkdir -p "${CHECKPOINT}" "${TENSORBOARD_DIR}" "${WANDB_DIR}"

if [[ ! -f "${PROMPT_FILE}" ]]; then
    echo "Prompt file not found: ${PROMPT_FILE}" >&2
    exit 1
fi

if [[ ! -f "${MODEL_ARGS_SCRIPT}" ]]; then
    echo "Model args script not found: ${MODEL_ARGS_SCRIPT}" >&2
    exit 1
fi

# ============================================================================
# ECC 任务参数
# ============================================================================

TARGET_N="18"
TARGET_K="7"

ECC_DIVERSITY_PENALTY="0.1"
ECC_HISTORY_MAX_SIZE="16384"
ECC_USE_CHAT_TEMPLATE="1"

# Qwen3: passed through to tokenizer.apply_chat_template (rollout + dataset when used).
APPLY_CHAT_TEMPLATE_KWARGS="{\"enable_thinking\": false}"

PROMPT_PREFIX="Construct a binary linear code over GF(2) with length n=${TARGET_N} and dimension k=${TARGET_K}. Try to maximize the minimum Hamming distance of the final code."
PROMPT_BODY="${PROMPT_PREFIX}

$(cat "${PROMPT_FILE}")"

if [[ -z "${ECC_FIXED_PROMPT:-}" && -z "${ECC_FIXED_MESSAGES:-}" ]]; then
    if [[ "${ECC_USE_CHAT_TEMPLATE}" == "1" ]]; then
        export ECC_FIXED_MESSAGES="$(ECC_PROMPT_CONTENT="${PROMPT_BODY}" python3 -c 'import json, os; print(json.dumps([{"role": "user", "content": os.environ["ECC_PROMPT_CONTENT"]}]))')"
    else
        export ECC_FIXED_PROMPT="${PROMPT_BODY}"
    fi
fi

export ECC_FIXED_METADATA="{\"task\":\"ecc\",\"n\":${TARGET_N},\"k\":${TARGET_K}}"

# ============================================================================
# 硬件配置
# ============================================================================

NUM_GPUS="8"
TP_SIZE="2"
ROLLOUT_GPUS_PER_ENGINE="2"
NUM_ROLLOUT="100"
ROLLOUT_BATCH_SIZE="1"
N_SAMPLES_PER_PROMPT="128"
NUM_STEPS_PER_ROLLOUT="1"
GLOBAL_BATCH_SIZE="128"
ROLLOUT_MAX_RESPONSE_LEN="4096"
ROLLOUT_TEMPERATURE="1.0"
ROLLOUT_TOP_P="1.0"
ENTROPY_COEF="0.001"
LR="1e-6"
MASTER_ADDR="127.0.0.1"

# ============================================================================
# NVLink 检测
# ============================================================================

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# ============================================================================
# 模型架构参数
# ============================================================================

source "${MODEL_ARGS_SCRIPT}"

# ============================================================================
# Checkpoint 参数
# ============================================================================

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${TORCH_DIST_CHECKPOINT}"
   --load "${CHECKPOINT}"
   --save "${CHECKPOINT}"
   --save-interval 10086
)

# ============================================================================
# Rollout 参数
# ============================================================================

ROLLOUT_ARGS=(
   --disable-rollout-global-dataset
   --rollout-function-path examples.ECC.fixed_prompt_rollout.generate_rollout
   --custom-rm-path examples.ECC.tool_reward.custom_rm
   --custom-rollout-log-function-path examples.ECC.save_rollout.log_and_save_rollout
   --reward-key score
   --log-reward-category category

   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT}"
   --global-batch-size "${GLOBAL_BATCH_SIZE}"

   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
   --rollout-temperature "${ROLLOUT_TEMPERATURE}"
   --rollout-top-p "${ROLLOUT_TOP_P}"

   --apply-chat-template-kwargs "${APPLY_CHAT_TEMPLATE_KWARGS}"

   --balance-data
)

if [[ "${ECC_USE_CHAT_TEMPLATE}" == "1" ]]; then
   ROLLOUT_ARGS+=(--apply-chat-template)
fi

# ============================================================================
# 评估参数
# ============================================================================

EVAL_ARGS=(
   # --eval-interval 100
)

# ============================================================================
# 并行度 / 性能参数
# ============================================================================

PERF_ARGS=(
   --tensor-model-parallel-size ${TP_SIZE}
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 32768
)

# ============================================================================
# GRPO 算法参数
# ============================================================================

GRPO_ARGS=(
   --advantage-estimator grpo
   --disable-rewards-normalization
   --entropy-coef "${ENTROPY_COEF}"
   --eps-clip 0.2
   --eps-clip-high 0.28
)

# ============================================================================
# 优化器参数
# ============================================================================

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR}"
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

# ============================================================================
# W&B + TensorBoard
# ============================================================================

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project "${PROJECT_NAME}"
   # --wandb-group "${EXP_NAME}"
   # --wandb-dir "${WANDB_DIR}"
   # --wandb-key "${WANDB_KEY}"
)

TB_ARGS=(
   --use-tensorboard
   --tb-project-name "${PROJECT_NAME}"
   --tb-experiment-name "${EXP_NAME}"
)

# ============================================================================
# SGLang 推理引擎参数
# ============================================================================

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine ${ROLLOUT_GPUS_PER_ENGINE}
   --sglang-mem-fraction-static 0.85
)

# ============================================================================
# 其他参数
# ============================================================================

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# ============================================================================
# 启动 Ray 集群并提交训练任务
# ============================================================================

export MASTER_ADDR="${MASTER_ADDR}"
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_LM_PATH}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"0\",
    \"TENSORBOARD_DIR\": \"${TENSORBOARD_DIR}\",
    \"ECC_FIXED_PROMPT\": $(printf '%s' "${ECC_FIXED_PROMPT}" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'),
    \"ECC_FIXED_MESSAGES\": $(printf '%s' "${ECC_FIXED_MESSAGES}" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'),
    \"ECC_FIXED_METADATA\": $(printf '%s' "${ECC_FIXED_METADATA}" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'),
    \"ECC_DIVERSITY_PENALTY\": \"${ECC_DIVERSITY_PENALTY}\",
    \"ECC_HISTORY_MAX_SIZE\": \"${ECC_HISTORY_MAX_SIZE}\"
  }
}"

cd "${SLIME_DIR}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${TB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"
