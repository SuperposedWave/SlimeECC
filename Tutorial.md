# slime 框架详细教程

> slime 是为 RL Scaling 设计的 LLM 后训练（post-training）框架，通过连接 **Megatron**（训练）与 **SGLang**（推理/rollout），实现高性能的强化学习训练。本教程将从架构、数据、参数、训练流程、自定义扩展等方面进行全面讲解。

---

## 目录

- [一、架构总览](#一架构总览)
  - [1.1 核心设计理念](#11-核心设计理念)
  - [1.2 三大模块](#12-三大模块)
  - [1.3 代码目录结构](#13-代码目录结构)
- [二、环境搭建](#二环境搭建)
  - [2.1 依赖项](#21-依赖项)
  - [2.2 安装步骤](#22-安装步骤)
- [三、输入数据格式](#三输入数据格式)
  - [3.1 基础格式：JSONL](#31-基础格式jsonl)
  - [3.2 对话格式（Chat Template）](#32-对话格式chat-template)
  - [3.3 多模态数据](#33-多模态数据)
  - [3.4 工具调用数据](#34-工具调用数据)
  - [3.5 数据预处理示例](#35-数据预处理示例)
- [四、Checkpoint 格式转换](#四checkpoint-格式转换)
  - [4.1 HuggingFace → torch_dist](#41-huggingface--torch_dist)
  - [4.2 torch_dist → HuggingFace](#42-torch_dist--huggingface)
- [五、训练流程详解](#五训练流程详解)
  - [5.1 同步训练流程 (train.py)](#51-同步训练流程-trainpy)
  - [5.2 异步训练流程 (train_async.py)](#52-异步训练流程-train_asyncpy)
  - [5.3 训练循环详细步骤](#53-训练循环详细步骤)
- [六、RL 算法详解](#六rl-算法详解)
  - [6.1 GRPO（Group Relative Policy Optimization）](#61-grpogroup-relative-policy-optimization)
  - [6.2 PPO（Proximal Policy Optimization）](#62-ppoproximal-policy-optimization)
  - [6.3 REINFORCE++](#63-reinforce)
  - [6.4 GSPO](#64-gspo)
  - [6.5 KL 散度控制](#65-kl-散度控制)
- [七、参数完整说明](#七参数完整说明)
  - [7.1 参数的三个来源](#71-参数的三个来源)
  - [7.2 集群 / Ray 参数](#72-集群--ray-参数)
  - [7.3 训练参数](#73-训练参数)
  - [7.4 Rollout 参数](#74-rollout-参数)
  - [7.5 数据参数](#75-数据参数)
  - [7.6 算法参数](#76-算法参数)
  - [7.7 评估参数](#77-评估参数)
  - [7.8 奖励模型参数](#78-奖励模型参数)
  - [7.9 SGLang 推理引擎参数](#79-sglang-推理引擎参数)
  - [7.10 W&B / TensorBoard 参数](#710-wb--tensorboard-参数)
  - [7.11 调试参数](#711-调试参数)
- [八、奖励函数（Reward Function）](#八奖励函数reward-function)
  - [8.1 内置奖励类型](#81-内置奖励类型)
  - [8.2 自定义奖励函数](#82-自定义奖励函数)
  - [8.3 远程奖励模型](#83-远程奖励模型)
  - [8.4 奖励后处理](#84-奖励后处理)
- [九、自定义 Rollout 函数](#九自定义-rollout-函数)
  - [9.1 自定义生成函数](#91-自定义生成函数)
  - [9.2 自定义完整 Rollout](#92-自定义完整-rollout)
- [十、工具调用与 Agent 训练](#十工具调用与-agent-训练)
  - [10.1 ReTool：代码解释器](#101-retool代码解释器)
  - [10.2 Search-R1：搜索增强推理](#102-search-r1搜索增强推理)
  - [10.3 Tau-Bench：Agent 环境交互](#103-tau-benchagent-环境交互)
  - [10.4 Multi-Agent：多智能体训练](#104-multi-agent多智能体训练)
- [十一、评估系统](#十一评估系统)
  - [11.1 基本评估配置](#111-基本评估配置)
  - [11.2 多任务评估配置（YAML）](#112-多任务评估配置yaml)
- [十二、高级特性](#十二高级特性)
  - [12.1 Colocate 模式](#121-colocate-模式)
  - [12.2 动态批大小](#122-动态批大小)
  - [12.3 数据均衡](#123-数据均衡)
  - [12.4 On-Policy Distillation](#124-on-policy-distillation)
  - [12.5 容错机制](#125-容错机制)
  - [12.6 Routing Replay（MoE 专用）](#126-routing-replaymoe-专用)
  - [12.7 SlimeRouter 中间件](#127-slimerouter-中间件)
- [十三、完整训练脚本示例](#十三完整训练脚本示例)
- [十四、常见问题](#十四常见问题)

---

## 一、架构总览

### 1.1 核心设计理念

slime 的核心目标是解决 LLM 强化学习训练中的两个关键挑战：

1. **高性能训练**：利用 Megatron-LM 的分布式训练能力（TP/PP/CP/EP），配合 SGLang 的高吞吐推理引擎，实现训练与推理的高效协同。
2. **灵活的数据生成**：提供自定义接口（Rollout 函数、Reward 函数、Generate 函数），支持任意形式的训练数据生成流程，包括多轮对话、工具调用、Agent 交互等。

### 1.2 三大模块

slime 的架构围绕三个核心模块构建：

```
┌──────────────────────────────────────────────────────────────────┐
│                        slime 架构                                │
│                                                                  │
│  ┌─────────────────┐    ┌──────────────┐    ┌────────────────┐  │
│  │  Training        │    │  Data Buffer  │    │  Rollout       │  │
│  │  (Megatron)      │◄───│  (桥梁模块)    │◄───│  (SGLang +     │  │
│  │                  │    │              │    │   Router)      │  │
│  │  · Actor Model   │───►│  · Prompt 管理 │───►│  · 推理引擎     │  │
│  │  · Critic Model  │    │  · 数据转换    │    │  · 奖励计算     │  │
│  │  · 参数同步       │    │  · 缓冲管理    │    │  · 数据采样     │  │
│  └─────────────────┘    └──────────────┘    └────────────────┘  │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  Ray：分布式编排、Placement Group 管理、远程调用               │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

- **Training（Megatron）**：负责模型的前向/反向传播和参数更新。支持 Actor-Critic 双模型架构。训练完成后将参数同步至 Rollout 模块。
- **Rollout（SGLang + Router）**：利用 SGLang 推理引擎生成新数据，计算奖励（reward/verifier），将结果存入 Data Buffer。
- **Data Buffer**：连接 Training 和 Rollout 的桥梁，管理 prompt 初始化、rollout 数据的转换和分发。

底层使用 **Ray** 进行分布式编排，管理 GPU 分配（Placement Group）和跨节点通信。

### 1.3 代码目录结构

```
slime/
├── train.py                    # 同步训练入口
├── train_async.py              # 异步训练入口
├── setup.py                    # 包安装配置 (v0.2.2)
├── build_conda.sh              # Conda 环境构建脚本
│
├── slime/                      # 核心包
│   ├── backends/
│   │   ├── megatron_utils/     # Megatron 训练后端
│   │   │   ├── actor.py        #   训练 Actor（前向/反向/参数更新）
│   │   │   ├── model.py        #   模型定义
│   │   │   ├── loss.py         #   损失函数（policy_loss, value_loss, sft_loss）
│   │   │   ├── data.py         #   数据处理
│   │   │   ├── checkpoint.py   #   Checkpoint 加载/保存
│   │   │   ├── model_provider.py  # 模型提供器（支持 bridge 模式）
│   │   │   ├── initialize.py   #   Megatron 初始化
│   │   │   ├── megatron_to_hf/ #   Megatron ↔ HF 格式转换
│   │   │   └── update_weight/  #   权重同步（Training → Rollout）
│   │   └── sglang_utils/       # SGLang 推理配置
│   │       ├── sglang_config.py   # SGLang 配置解析
│   │       └── sglang_engine.py   # SGLang 引擎管理
│   │
│   ├── ray/                    # Ray 分布式编排
│   │   ├── placement_group.py  #   GPU Placement Group
│   │   ├── rollout.py          #   RolloutManager（核心调度器）
│   │   ├── train_actor.py      #   训练 Actor 封装
│   │   └── actor_group.py      #   Actor Group（多 GPU 训练组）
│   │
│   ├── rollout/                # Rollout 数据生成
│   │   ├── sglang_rollout.py   #   默认 SGLang Rollout 实现
│   │   ├── sft_rollout.py      #   SFT Rollout（监督微调用）
│   │   ├── data_source.py      #   数据源管理
│   │   ├── base_types.py       #   Sample 等基础类型定义
│   │   ├── filter_hub/         #   动态采样过滤器
│   │   ├── generate_hub/       #   Benchmark 生成
│   │   └── rm_hub/             #   奖励模型集合
│   │       ├── __init__.py     #     奖励分发入口
│   │       ├── deepscaler.py   #     DeepScaler 奖励
│   │       ├── math_utils.py   #     数学验证工具
│   │       ├── math_dapo_utils.py # DAPO 数学奖励
│   │       ├── f1.py           #     F1 分数奖励
│   │       └── gpqa.py         #     GPQA 奖励
│   │
│   ├── router/                 # 请求路由
│   │   ├── router.py           #   SlimeRouter
│   │   └── middleware_hub/     #   中间件（如 Radix Tree）
│   │
│   └── utils/                  # 工具函数
│       ├── arguments.py        #   命令行参数定义
│       ├── data.py             #   数据加载与处理
│       ├── ppo_utils.py        #   PPO/GRPO 算法工具
│       ├── eval_config.py      #   评估配置
│       └── metric_utils.py     #   指标工具
│
├── slime_plugins/              # 插件系统
│   ├── mbridge/                #   模型桥接（Megatron ↔ HuggingFace）
│   ├── models/                 #   自定义模型组件
│   └── rollout_buffer/         #   Rollout Buffer 插件
│
├── tools/                      # 工具脚本
│   ├── convert_hf_to_torch_dist.py     # HF → Megatron 格式
│   ├── convert_torch_dist_to_hf_*.py   # Megatron → HF 格式
│   └── profile_rollout.py              # 性能分析
│
├── examples/                   # 示例
│   ├── math_test/              #   数学推理 GRPO 训练
│   ├── retool/                 #   工具调用训练
│   ├── tau-bench/              #   Agent 环境交互
│   ├── search-r1/              #   搜索增强推理
│   ├── multi_agent/            #   多智能体训练
│   ├── fully_async/            #   全异步训练
│   ├── on_policy_distillation/ #   在线蒸馏
│   └── ...
│
├── scripts/                    # 运行脚本
│   ├── models/                 #   模型架构配置
│   └── run-*.sh                #   各模型训练脚本
│
└── docs/                       # 文档
    ├── en/                     #   英文文档
    └── zh/                     #   中文文档
```

---

## 二、环境搭建

### 2.1 依赖项

slime 依赖以下核心组件：

| 组件 | 说明 |
|------|------|
| Python ≥ 3.10 | 基础运行环境 |
| PyTorch ≥ 2.9 | 深度学习框架 |
| CUDA ≥ 12.9 | GPU 加速 |
| Megatron-LM | 分布式训练后端 |
| SGLang | 高性能推理引擎 |
| Ray | 分布式任务编排 |
| Flash Attention | 高效注意力计算 |
| Transformer Engine | FP8 训练支持 |

### 2.2 安装步骤

推荐使用项目提供的 `build_conda.sh` 一键安装：

```bash
bash build_conda.sh
```

该脚本会依次完成：

1. 创建 Python 3.12 的 conda 环境
2. 安装 CUDA 12.9、cuDNN、NCCL
3. 安装 PyTorch 2.9.1
4. 克隆并安装 SGLang（指定 commit）
5. 安装 Flash Attention、Transformer Engine、Apex
6. 克隆并安装 Megatron-LM（指定 commit）
7. 安装 slime 本体（`pip install -e .`）
8. 应用 SGLang 和 Megatron 的补丁

手动安装时，确保 Megatron-LM 路径在 `PYTHONPATH` 中：

```bash
export PYTHONPATH=/path/to/Megatron-LM/:$PYTHONPATH
```

---

## 三、输入数据格式

slime 支持 **JSONL**（每行一个 JSON 对象）和 **Parquet** 两种数据格式。

### 3.1 基础格式：JSONL

最简单的数据格式包含 prompt 和 label 两个字段：

```jsonl
{"prompt": "1 + 1 等于多少？", "label": "2"}
{"prompt": "计算 sin(π/6) 的值", "label": "0.5"}
```

通过 `--input-key` 和 `--label-key` 指定对应的字段名：

```bash
--input-key prompt --label-key label
```

### 3.2 对话格式（Chat Template）

当使用 `--apply-chat-template` 时，`input-key` 对应的字段应为消息列表：

```jsonl
{"prompt": [{"role": "user", "content": "计算 1+1 等于多少"}], "label": "2"}
{"prompt": [{"role": "system", "content": "你是一个数学助手"}, {"role": "user", "content": "求解 x^2 = 4"}], "label": "2"}
```

每条消息包含：
- `role`：角色，通常为 `"system"`、`"user"` 或 `"assistant"`
- `content`：消息内容

slime 会自动调用模型 tokenizer 的 `apply_chat_template` 方法将消息列表转换为模型所需的格式。

### 3.3 多模态数据

对于视觉语言模型（VLM），通过 `--multimodal-keys` 指定媒体字段映射：

```bash
--multimodal-keys '{"image": "image_path"}'
```

数据格式示例：

```jsonl
{"image_path": "/data/images/geo_001.png", "prompt": [{"role": "user", "content": "<image>请描述这张图片中的几何图形"}], "label": "三角形"}
```

### 3.4 工具调用数据

当训练需要工具调用能力时，通过 `--tool-key` 指定工具定义字段：

```jsonl
{
  "prompt": [{"role": "user", "content": "帮我计算 2^10"}],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "code_interpreter",
        "description": "执行 Python 代码",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}}
      }
    }
  ],
  "label": "1024"
}
```

### 3.5 数据预处理示例

以 math_test 为例，将原始数据转换为 slime 所需格式：

```python
from datasets import load_dataset

DATA_DIR = "/path/to/data"

def transform(example):
    return {
        "prompt": example["prompt"],
        "label": example["reward_model"]["ground_truth"]["target"],
    }

for split in ["train", "test"]:
    ds = load_dataset("parquet", data_files=f"{DATA_DIR}/{split}.parquet", split="train")
    ds2 = ds.map(transform, remove_columns=ds.column_names)
    ds2.to_json(f"{DATA_DIR}/{split}_processed.jsonl", orient="records", lines=True)
```

核心要点：将原始数据映射到 `prompt`（输入）和 `label`（标签/答案）两个字段。

### 3.6 数据字段总结

| 字段 | 参数指定 | 类型 | 说明 |
|------|---------|------|------|
| prompt（输入） | `--input-key` | `str` 或 `list[dict]` | 原始文本或消息列表 |
| label（标签） | `--label-key` | `str` / `float` / `dict` | 奖励计算所需的真实答案 |
| metadata | `--metadata-key` | `dict` | 可选，附加元数据 |
| tools | `--tool-key` | `list[dict]` | 可选，工具定义 |

---

## 四、Checkpoint 格式转换

slime 使用 Megatron 进行训练，需要将 HuggingFace 格式的模型权重转换为 Megatron 的 `torch_dist` 格式。

### 4.1 HuggingFace → torch_dist

使用 `tools/convert_hf_to_torch_dist.py`：

```bash
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    --swiglu \
    --num-layers 36 \
    --hidden-size 2560 \
    --ffn-hidden-size 9728 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 8 \
    --use-rotary-position-embeddings \
    --disable-bias-linear \
    --normalization "RMSNorm" \
    --norm-epsilon 1e-6 \
    --vocab-size 151936 \
    --kv-channels 128 \
    --qk-layernorm \
    --hf-checkpoint "/path/to/Qwen3-4B" \
    --save "/path/to/Qwen3-4B_torch_dist"
```

模型架构参数（`--num-layers`、`--hidden-size` 等）需要与模型的实际配置一致。slime 在 `scripts/models/` 目录下提供了常见模型的预设配置。

### 4.2 torch_dist → HuggingFace

训练完成后，如果需要将 Megatron 格式的 checkpoint 转回 HuggingFace 格式：

```bash
python tools/convert_torch_dist_to_hf_qwen3.py \
    --load "/path/to/slime_checkpoint" \
    --save "/path/to/hf_output" \
    --hf-checkpoint "/path/to/original_hf_model"
```

slime 为不同模型提供了专门的转换脚本（`convert_torch_dist_to_hf_qwen3.py`、`convert_torch_dist_to_hf_glm4.py` 等）。

---

## 五、训练流程详解

### 5.1 同步训练流程 (train.py)

`train.py` 是主训练入口，执行同步的"rollout → train"循环：

```
初始化阶段：
  1. parse_args()                → 解析所有参数
  2. create_placement_groups()   → 分配 GPU（训练/推理）
  3. create_rollout_manager()    → 创建 RolloutManager + SGLang 引擎
  4. create_training_models()    → 创建 Actor（+ Critic）模型
  5. actor_model.update_weights()→ 将训练权重同步到推理引擎

训练循环（for rollout_id in range(num_rollout)）：
  6. rollout_manager.generate()  → 生成 rollout 数据（推理 + 奖励）
  7. actor_model.async_train()   → Actor 训练（前向/反向/更新）
  8. critic_model.async_train()  → Critic 训练（如果启用 PPO）
  9. save()                      → 定期保存 checkpoint
  10. actor_model.update_weights()→ 将更新后的权重同步到推理引擎
  11. rollout_manager.eval()     → 定期评估
```

### 5.2 异步训练流程 (train_async.py)

`train_async.py` 通过流水线化 rollout 和 train 来提升 GPU 利用率：

```
当前 rollout_id 的训练与下一个 rollout_id 的数据生成并行执行：

  时间线：
  ├─ rollout(id=0) ─────────────────────────────────────────┤
  │                  ├─ train(id=0) ─┤─ rollout(id=1) ──────┤
  │                  │               │  ├─ train(id=1) ─┤   │
  │                  │               │  │               │   │
```

注意：异步训练**不支持** `--colocate` 模式（推理和训练不能共享同一组 GPU）。

### 5.3 训练循环详细步骤

每个 rollout 步骤中，Actor 的训练过程如下：

1. **获取 Rollout 数据**：从 RolloutManager 获取生成的 samples，包含 tokens、response_lengths、rewards、loss_masks 等。
2. **计算参考模型 log_probs**：使用冻结的参考模型（`--ref-load`）计算 token 级别的对数概率。
3. **计算当前策略 log_probs**：当前 Actor 模型的前向传播。
4. **计算优势函数**：根据选择的算法（GRPO/PPO/REINFORCE++）计算 advantages 和 returns。
5. **训练步**：使用 PPO Clip 目标函数计算 loss，反向传播更新参数。

---

## 六、RL 算法详解

slime 通过 `--advantage-estimator` 参数选择不同的 RL 算法。

### 6.1 GRPO（Group Relative Policy Optimization）

**默认算法**。核心思想：对同一 prompt 生成的多个回复（group），用组内相对奖励作为优势估计。

```
returns = reward - group_mean(reward)
（可选）returns = returns / group_std(reward)
```

关键参数：

```bash
--advantage-estimator grpo
--n-samples-per-prompt 8       # 每个 prompt 生成 8 个回复
--eps-clip 0.2                 # PPO clip 范围
--eps-clip-high 0.28           # 非对称上界 clip（可选）
```

GRPO 不需要 Critic 模型，训练效率更高。

### 6.2 PPO（Proximal Policy Optimization）

需要额外的 Critic（价值函数）模型，使用 GAE（Generalized Advantage Estimation）计算优势函数。

```bash
--advantage-estimator ppo
--critic-load /path/to/critic_checkpoint
--critic-save /path/to/critic_save
--gamma 1.0                    # GAE 折扣因子
--lambd 1.0                    # GAE lambda
--value-clip 0.2               # 价值函数 clip 范围
```

### 6.3 REINFORCE++

REINFORCE 的改进版，使用逐 token 折扣回报：

```bash
--advantage-estimator reinforce_plus_plus
# 或带 baseline 的版本：
--advantage-estimator reinforce_plus_plus_baseline
```

### 6.4 GSPO

与 GRPO 类似，但使用 sequence-level KL 散度：

```bash
--advantage-estimator gspo
```

### 6.5 KL 散度控制

slime 支持多种 KL 散度计算方式来约束策略更新幅度：

```bash
--use-kl-loss                  # 启用 KL loss
--kl-loss-coef 0.01            # KL loss 系数
--kl-loss-type low_var_kl      # KL 类型：k1, k2, k3, low_var_kl
--kl-coef 0.0                  # 奖励上的 KL 惩罚系数
```

| KL 类型 | 说明 |
|---------|------|
| `k1` | 标准 KL 散度 |
| `k2` | 反向 KL 散度 |
| `k3` | Jensen-Shannon 散度 |
| `low_var_kl` | 低方差 KL 估计（推荐） |

---

## 七、参数完整说明

### 7.1 参数的三个来源

slime 的参数来自三个层面：

1. **Megatron 参数**：所有 Megatron-LM 原生参数均可直接使用，如 `--tensor-model-parallel-size`、`--pipeline-model-parallel-size` 等。
2. **SGLang 参数**：SGLang 推理引擎参数需添加 `--sglang-` 前缀，如 `--sglang-mem-fraction-static`。
3. **slime 自身参数**：定义在 `slime/utils/arguments.py` 中，下面详细列出。

### 7.2 集群 / Ray 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--actor-num-nodes` | int | 1 | 训练 Actor 的节点数 |
| `--actor-num-gpus-per-node` | int | 8 | 每个节点用于训练的 GPU 数 |
| `--critic-num-nodes` | int | 同 Actor | Critic 的节点数 |
| `--critic-num-gpus-per-node` | int | 同 Actor | Critic 每节点 GPU 数 |
| `--rollout-num-gpus` | int | - | 推理用 GPU 总数（colocate 时忽略） |
| `--rollout-num-gpus-per-engine` | int | 1 | 每个推理引擎的 GPU 数（即 TP 大小） |
| `--num-gpus-per-node` | int | 8 | 每节点 GPU 数（rollout 用） |
| `--colocate` | flag | False | 推理与训练共享 GPU |
| `--offload` | flag | False | 启用 CPU offload（等价于同时开启 offload-train 和 offload-rollout） |
| `--offload-train` | bool | - | 训练时将模型 offload 到 CPU |
| `--offload-rollout` | bool | - | 推理时将模型 offload 到 CPU |

### 7.3 训练参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--megatron-to-hf-mode` | str | `"raw"` | Megatron→HF 转换模式：`raw`（原始）或 `bridge`（桥接） |
| `--custom-model-provider-path` | str | - | 自定义模型提供器路径 |
| `--recompute-loss-function` | flag | False | 重新计算 loss 以节省显存 |
| `--log-probs-chunk-size` | int | -1 | 分块计算 log-prob 的 chunk 大小 |
| `--only-train-params-name-list` | str[] | - | 只训练匹配的参数（正则表达式） |
| `--freeze-params-name-list` | str[] | - | 冻结匹配的参数（正则表达式） |
| `--train-env-vars` | JSON | `{}` | 训练进程的额外环境变量 |

### 7.4 Rollout 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--hf-checkpoint` | str | - | HuggingFace checkpoint 路径（用于 SGLang 和 tokenizer） |
| `--model-name` | str | - | 模型名称（用于 Megatron→HF 转换） |
| `--rollout-function-path` | str | `slime.rollout.sglang_rollout.generate_rollout` | Rollout 生成函数路径 |
| `--rollout-temperature` | float | 1.0 | 采样温度 |
| `--rollout-top-p` | float | 1.0 | Top-p 采样 |
| `--rollout-top-k` | int | -1 | Top-k 采样 |
| `--rollout-max-context-len` | int | - | 最大上下文长度 |
| `--rollout-max-prompt-len` | int | - | 最大 prompt 长度 |
| `--rollout-max-response-len` | int | - | 最大回复长度（max_tokens） |
| `--rollout-stop` | str[] | - | 停止字符串列表 |
| `--rollout-stop-token-ids` | int[] | - | 停止 token ID 列表 |
| `--rollout-shuffle` | flag | False | 是否打乱 prompt 顺序 |
| `--rollout-seed` | int | 42 | 随机种子 |
| `--rollout-skip-special-tokens` | flag | False | 解码时跳过特殊 token |
| `--over-sampling-batch-size` | int | rollout_batch_size | 过采样批大小 |
| `--custom-generate-function-path` | str | - | 自定义生成函数路径 |
| `--rollout-external` | flag | False | 使用外部 SGLang 实例 |
| `--update-weights-interval` | int | 1 | 权重更新间隔 |

### 7.5 数据参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--prompt-data` | str | - | Prompt 数据文件路径（jsonl/parquet） |
| `--input-key` | str | `"input"` | 数据中 prompt 对应的 JSON key |
| `--label-key` | str | - | 数据中标签对应的 JSON key |
| `--metadata-key` | str | `"metadata"` | 元数据 JSON key |
| `--tool-key` | str | `"tools"` | 工具定义 JSON key |
| `--apply-chat-template` | flag | False | 使用 tokenizer 的 chat template |
| `--num-rollout` | int | - | Rollout 总步数 |
| `--num-epoch` | int | - | 训练轮数（用于计算 num_rollout） |
| `--rollout-batch-size` | int | **必填** | 每步 rollout 的 prompt 数 |
| `--n-samples-per-prompt` | int | 1 | 每个 prompt 生成的回复数 |
| `--global-batch-size` | int | - | 全局批大小 |
| `--micro-batch-size` | int | 1 | 微批大小 |
| `--balance-data` | flag | False | 跨 DP rank 均衡 token 数 |
| `--use-dynamic-batch-size` | flag | False | 动态微批大小 |
| `--max-tokens-per-gpu` | int | - | 每 GPU 最大 token 数（动态批大小用） |

### 7.6 算法参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--advantage-estimator` | str | `"grpo"` | 优势估计器：grpo / gspo / ppo / reinforce_plus_plus / reinforce_plus_plus_baseline |
| `--eps-clip` | float | 0.2 | PPO clip 范围 ε |
| `--eps-clip-high` | float | 同 eps-clip | 非对称上界 clip |
| `--eps-clip-c` | float | - | Dual-clip PPO 下界 |
| `--value-clip` | float | 0.2 | 价值函数 clip |
| `--kl-coef` | float | 0.0 | 奖励上的 KL 惩罚系数 |
| `--use-kl-loss` | flag | False | 启用 KL loss |
| `--kl-loss-coef` | float | 0.0 | KL loss 系数 |
| `--kl-loss-type` | str | `"k1"` | KL 类型 |
| `--entropy-coef` | float | 0.0 | 熵正则化系数 |
| `--gamma` | float | 1.0 | GAE 折扣因子 |
| `--lambd` | float | 1.0 | GAE lambda |
| `--normalize-advantages` | flag | False | 标准化优势函数 |
| `--disable-grpo-std-normalization` | flag | False | 禁用 Dr.GRPO 标准差归一化 |
| `--disable-rewards-normalization` | flag | False | 禁用奖励归一化 |
| `--loss-type` | str | `"policy_loss"` | 损失类型：policy_loss / sft_loss / custom_loss |
| `--custom-loss-function-path` | str | - | 自定义损失函数路径 |
| `--ref-load` | str | - | 参考模型 checkpoint |
| `--ref-update-interval` | int | - | 参考模型更新间隔 |
| `--use-rollout-logprobs` | flag | False | 使用 rollout 阶段的 logprobs 计算重要性采样比率 |
| `--reset-optimizer-states` | flag | False | 每轮 rollout 重置优化器状态 |

### 7.7 评估参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--eval-interval` | int | - | 评估间隔（每 N 个 rollout 评估一次） |
| `--eval-prompt-data` | str[] | - | 评估数据集（格式：name path） |
| `--eval-config` | str | - | 评估配置 YAML 文件 |
| `--n-samples-per-eval-prompt` | int | 1 | 每个评估 prompt 生成的回复数 |
| `--eval-temperature` | float | - | 评估采样温度 |
| `--eval-top-p` | float | - | 评估 top-p |
| `--eval-max-response-len` | int | - | 评估最大回复长度 |
| `--skip-eval-before-train` | flag | False | 跳过训练前评估 |

### 7.8 奖励模型参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--rm-type` | str | - | 奖励类型（deepscaler / math / dapo / f1 / gpqa / remote_rm 等） |
| `--custom-rm-path` | str | - | 自定义奖励函数路径 |
| `--rm-url` | str | - | 远程奖励模型 URL |
| `--reward-key` | str | - | 奖励字典中的 key |
| `--group-rm` | flag | False | 按组计算奖励 |
| `--custom-reward-post-process-path` | str | - | 自定义奖励后处理函数 |

### 7.9 SGLang 推理引擎参数

所有 SGLang 参数需要加 `--sglang-` 前缀。常用参数：

| 参数 | 说明 |
|------|------|
| `--sglang-mem-fraction-static` | KV Cache 占用的显存比例（默认 0.7） |
| `--sglang-context-length` | 模型上下文长度 |
| `--sglang-trust-remote-code` | 信任远程代码 |
| `--sglang-quantization` | 量化方法 |
| `--sglang-dtype` | 数据类型 |

高级配置可通过 `--sglang-config` 传入 YAML 文件，支持多模型、PD 分离等复杂部署。

### 7.10 W&B / TensorBoard 参数

```bash
# Weights & Biases
--use-wandb
--wandb-project "my-project"
--wandb-group "experiment-group"
--wandb-key "your-api-key"
--log-passrate                 # 记录 pass@n
--log-multi-turn               # 记录多轮对话

# TensorBoard
--use-tensorboard
--tb-project-name "my-project"
--tb-experiment-name "exp-001"
```

### 7.11 调试参数

| 参数 | 说明 |
|------|------|
| `--debug-rollout-only` | 只运行 rollout，不训练 |
| `--debug-train-only` | 只运行训练，不 rollout |
| `--save-debug-rollout-data PATH` | 保存 rollout 数据到文件 |
| `--load-debug-rollout-data PATH` | 从文件加载 rollout 数据 |
| `--save-debug-train-data PATH` | 保存训练数据到文件 |
| `--dump-details PATH` | 导出训练详情 |

---

## 八、奖励函数（Reward Function）

奖励函数是 RL 训练的核心组件，决定了模型优化的方向。

### 8.1 内置奖励类型

通过 `--rm-type` 选择：

| 类型 | 说明 | 适用场景 |
|------|------|---------|
| `deepscaler` | 提取 `</think>` 后的答案，与标签比较 | 数学推理（支持 CoT） |
| `math` | 使用 `grade_answer_verl` 验证数学答案 | 通用数学题 |
| `dapo` | DAPO 风格数学打分 | 数学题 |
| `f1` | F1 分数 | 文本匹配/抽取式 QA |
| `gpqa` | GPQA 奖励 | GPQA 基准测试 |
| `ifbench` | IFBench 奖励 | 指令遵循 |
| `remote_rm` | HTTP POST 到远程奖励服务 | 需要外部模型打分 |
| `random` | 随机 0/1 | 调试用 |

使用 `boxed_` 前缀可先提取 `\boxed{}` 中的答案再应用奖励，如 `boxed_math`。

### 8.2 自定义奖励函数

通过 `--custom-rm-path` 指定自定义奖励函数：

```bash
--custom-rm-path my_reward_module.reward_func
```

函数签名：

```python
async def reward_func(args, sample, **kwargs):
    """
    Args:
        args: 训练参数
        sample: Sample 对象，包含以下属性：
            - sample.prompt: 原始 prompt
            - sample.response: 模型生成的回复
            - sample.label: 标签/真实答案
            - sample.metadata: 元数据
    Returns:
        float 或 dict: 奖励值。如果返回 dict，通过 --reward-key 指定取用的 key
    """
    # 示例：数学答案验证
    predicted = extract_answer(sample.response)
    correct = (predicted == sample.label)
    return 1.0 if correct else 0.0
```

实际示例（retool 工具调用奖励）：

```python
async def reward_func(args, sample, **kwargs):
    score = compute_score_dapo(sample.response, sample.label)
    has_tool_call = "<tool_call>" in sample.response
    tool_bonus = 0.1 if has_tool_call else 0.0
    return {"score": score + tool_bonus}
```

### 8.3 远程奖励模型

当奖励需要由外部模型计算时（如 reward model server），使用远程奖励：

```bash
--rm-type remote_rm
--rm-url "http://reward-server:8080/reward"
```

slime 会将 sample 信息以 HTTP POST 发送到指定 URL，接收返回的奖励值。

### 8.4 奖励后处理

通过 `--custom-reward-post-process-path` 自定义奖励后处理逻辑：

```python
def reward_post_process(args, samples):
    """
    Args:
        args: 训练参数
        samples: Sample 列表
    Returns:
        (raw_rewards, processed_rewards): 原始奖励和处理后的奖励
    """
    raw_rewards = [s.reward for s in samples]
    # 自定义归一化或变换
    processed = normalize(raw_rewards)
    return raw_rewards, processed
```

默认情况下，GRPO/GSPO 会自动进行组内奖励归一化（减去均值，可选除以标准差）。

---

## 九、自定义 Rollout 函数

slime 提供两个层级的自定义接口，满足不同程度的定制需求。

### 9.1 自定义生成函数

通过 `--custom-generate-function-path` 替换**单个 sample 的生成逻辑**，适合多轮对话、工具调用等场景：

```bash
--custom-generate-function-path my_generate.generate
```

函数签名：

```python
async def generate(args, sample, sampling_params, **kwargs):
    """
    Args:
        args: 训练参数
        sample: Sample 对象（包含 prompt、tokenized 信息等）
        sampling_params: 采样参数（temperature, top_p, max_tokens 等）
    Returns:
        Sample: 包含生成结果的 Sample 对象
    """
    # 第一轮：生成初始回复
    response = await call_sglang(sample.prompt, sampling_params)
    
    # 多轮交互：解析工具调用，执行工具，拼接结果
    while has_tool_call(response):
        tool_result = execute_tool(response)
        response = await call_sglang(
            sample.prompt + response + tool_result,
            sampling_params
        )
    
    sample.response = response
    return sample
```

### 9.2 自定义完整 Rollout

通过 `--rollout-function-path` 替换**整个 rollout 流程**，适合需要完全控制数据生成的场景：

```bash
--rollout-function-path my_rollout.generate_rollout
```

函数签名：

```python
def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    """
    Args:
        args: 训练参数
        rollout_id: 当前 rollout 步骤编号
        data_buffer: 数据缓冲区（含 prompt 数据源）
        evaluation: 是否为评估模式
    Returns:
        list[Sample]: 生成的 Sample 列表
    """
    samples = []
    prompts = data_buffer.get_samples(args.rollout_batch_size)
    
    for prompt in prompts:
        # 自定义生成逻辑
        response = custom_inference(prompt)
        reward = custom_reward(response)
        samples.append(Sample(
            prompt=prompt,
            response=response,
            reward=reward,
            ...
        ))
    
    return samples
```

slime 内置了多种 rollout 实现：

| 路径 | 用途 |
|------|------|
| `slime.rollout.sglang_rollout.generate_rollout` | 默认 SGLang rollout |
| `slime.rollout.sft_rollout.generate_rollout` | SFT 数据 rollout（无推理） |
| `slime.rollout.sleep_rollout.generate_rollout` | 调试用空 rollout |

---

## 十、工具调用与 Agent 训练

slime 支持多种 Agent 训练范式，通过自定义 generate 和 reward 函数实现。

### 10.1 ReTool：代码解释器

`examples/retool/` 实现了带代码解释器的多轮工具调用训练。

**工作流程：**

```
用户 Prompt → LLM 生成 → 检测 <tool_call> → 执行代码 → 返回结果 → LLM 继续生成 → ...
```

**核心文件：**
- `generate_with_retool.py`：自定义生成函数（多轮工具交互）
- `tool_sandbox.py`：代码沙箱执行器

**工具调用格式：**

```
<tool_call>
{"name": "code_interpreter", "arguments": {"code": "print(2**10)"}}
</tool_call>
```

**执行结果格式：**

```
<interpreter>
1024
</interpreter>
```

**关键配置：**

```bash
--custom-generate-function-path generate_with_retool.generate
--custom-rm-path generate_with_retool.reward_func
--reward-key score
```

### 10.2 Search-R1：搜索增强推理

`examples/search-r1/` 实现了多轮搜索增强的推理训练。

**工作流程：**

```
用户 Prompt → LLM 生成 → 检测 <search>query</search> → 执行搜索 → 返回 <information>... → LLM 继续 → <answer>...
```

模型学会在推理过程中主动发起搜索查询，利用检索到的信息来辅助回答。

**关键配置：**

```bash
--custom-generate-function-path generate_with_search.generate
--custom-rm-path qa_em_format.reward_func
--input-key prompt
--label-key reward_model
```

### 10.3 Tau-Bench：Agent 环境交互

`examples/tau-bench/` 实现了在模拟环境中训练 Agent 的能力（如零售/航空客服场景）。

**工作流程：**

```
任务描述 → Agent 调用 API → 环境返回结果 → Agent 决策 → ... → 环境给出奖励
```

**核心特点：**
- 使用 OpenAI 风格的 tool calling（function calling）
- Agent 通过 `trainable_agents.py` 中的 `TrainableAgentMixin` 包装
- 环境提供标准化的 API 和奖励信号

**关键配置：**

```bash
--custom-generate-function-path generate_with_tau.generate
--input-key index
--dynamic-sampling-filter-path ...check_reward_nonzero_std
```

`--dynamic-sampling-filter-path` 过滤掉奖励方差为零的组（即所有回复都对或都错的 prompt），提高训练效率。

### 10.4 Multi-Agent：多智能体训练

`examples/multi_agent/` 支持多个 Agent 协同训练。

**工作流程：**

```
任务 → Agent 1 处理 → Agent 2 协作 → ... → 综合奖励
```

**关键配置：**

```bash
--rollout-function-path rollout_with_multi_agents.generate_with_multi_agents
--custom-config-path multi_agent_config.yaml
```

---

## 十一、评估系统

### 11.1 基本评估配置

在训练脚本中配置评估：

```bash
--eval-interval 20                          # 每 20 个 rollout 评估一次
--eval-prompt-data dataset_name /path/to/eval.jsonl   # 评估数据集
--n-samples-per-eval-prompt 16              # 每个评估 prompt 生成 16 个回复
--eval-max-response-len 16384              # 评估最大回复长度
--eval-temperature 0.6                      # 评估采样温度
--eval-top-p 0.95                           # 评估 top-p
```

`--eval-prompt-data` 接受 `name path` 对，name 用于 W&B/TensorBoard 中的标识。

### 11.2 多任务评估配置（YAML）

使用 `--eval-config` 传入 YAML 文件，为每个评估数据集设定独立参数：

```yaml
datasets:
  - name: math500
    path: /data/eval/math500.jsonl
    rm_type: deepscaler
    n_samples_per_eval_prompt: 8
    temperature: 0.6
    max_response_len: 8192

  - name: gpqa
    path: /data/eval/gpqa.jsonl
    rm_type: gpqa
    n_samples_per_eval_prompt: 4
    temperature: 0.0
    max_response_len: 4096

  - name: aime
    path: /data/eval/aime.jsonl
    rm_type: math
    input_key: problem
    label_key: answer
    n_samples_per_eval_prompt: 16
    temperature: 1.0
```

每个数据集可以覆盖的配置项包括：

| 字段 | 类型 | 说明 |
|------|------|------|
| `name` | str | 数据集名称 |
| `path` | str | 数据文件路径 |
| `rm_type` | str | 奖励模型类型 |
| `input_key` | str | 输入字段 key |
| `label_key` | str | 标签字段 key |
| `tool_key` | str | 工具字段 key |
| `n_samples_per_eval_prompt` | int | 每 prompt 采样数 |
| `temperature` | float | 采样温度 |
| `top_p` | float | Top-p |
| `top_k` | int | Top-k |
| `max_response_len` | int | 最大回复长度 |
| `stop` | list[str] | 停止字符串 |
| `min_new_tokens` | int | 最小新 token 数 |

---

## 十二、高级特性

### 12.1 Colocate 模式

`--colocate` 让训练和推理共享同一组 GPU，通过 offload 机制在两个阶段之间切换：

```bash
--colocate                     # 启用 colocate
--actor-num-gpus-per-node 8    # 所有 8 张 GPU 同时用于训练和推理
```

工作原理：
1. Rollout 阶段：加载推理模型到 GPU，训练模型 offload 到 CPU
2. Training 阶段：加载训练模型到 GPU，推理模型 offload 到 CPU

适用于 GPU 资源有限的场景，代价是增加了 offload 开销。

### 12.2 动态批大小

`--use-dynamic-batch-size` 根据序列长度动态调整 micro batch size，避免 OOM：

```bash
--use-dynamic-batch-size
--max-tokens-per-gpu 18432     # 每 GPU 最大 token 数
```

实际的 micro batch size 会根据当前 batch 中序列的实际长度动态计算，确保每 GPU 处理的 token 总数不超过 `--max-tokens-per-gpu`。

### 12.3 数据均衡

`--balance-data` 在多 DP rank 间均衡 token 数量，避免负载不均：

```bash
--balance-data
```

当不同 sample 的长度差异较大时，某些 rank 可能分到更多 token 而变成瓶颈。启用此选项后，slime 会重新分配 sample 使各 rank 的 token 数尽量一致。

### 12.4 On-Policy Distillation

`--use-opd` 在 RL 训练的同时从教师模型蒸馏知识：

```bash
--use-opd
--opd-type sglang              # 使用 SGLang 推理教师模型
--opd-kl-coef 1.0              # 蒸馏 KL 系数
--opd-teacher-load /path/to/teacher
```

### 12.5 容错机制

启用 rollout 阶段的容错能力：

```bash
--use-fault-tolerance
--rollout-health-check-interval 30     # 健康检查间隔（秒）
--rollout-health-check-timeout 30      # 健康检查超时（秒）
```

当 SGLang 推理引擎异常时，slime 会自动重启失败的引擎，保证训练不中断。

### 12.6 Routing Replay（MoE 专用）

对于 MoE（Mixture of Experts）模型，`--use-routing-replay` 在训练时重放 rollout 阶段的路由决策：

```bash
--use-routing-replay
```

这确保了训练阶段使用与 rollout 阶段相同的 expert 路由，减少训练-推理不一致性。

### 12.7 SlimeRouter 中间件

`--use-slime-router` 启用 slime 自定义的请求路由器，支持中间件扩展：

```bash
--use-slime-router
--slime-router-middleware-paths "slime.router.middleware_hub.radix_tree"
```

中间件可以实现如 Radix Tree 缓存等高级路由策略。

---

## 十三、完整训练脚本示例

以下是一个完整的 Qwen3-4B GRPO 数学推理训练脚本：

```bash
#!/bin/bash

# 清理旧进程
pkill -9 sglang; sleep 3
ray stop --force; pkill -9 ray; pkill -9 python; sleep 3

set -ex

# ========================= 路径配置 =========================
HF_CHECKPOINT="/path/to/Qwen3-4B"
TORCH_DIST_CHECKPOINT="/path/to/Qwen3-4B_torch_dist"
SLIME_CHECKPOINT="/path/to/Qwen3-4B_slime"
TRAIN_DATA="/path/to/train_processed.jsonl"
EVAL_DATA="/path/to/test_processed.jsonl"

NUM_GPUS=8
TP_SIZE=2
ROLLOUT_GPUS_PER_ENGINE=2

# ========================= 模型架构 =========================
MODEL_ARGS=(
   --swiglu
   --num-layers 36
   --hidden-size 2560
   --ffn-hidden-size 9728
   --num-attention-heads 32
   --group-query-attention
   --num-query-groups 8
   --use-rotary-position-embeddings
   --disable-bias-linear
   --normalization "RMSNorm"
   --norm-epsilon 1e-6
   --vocab-size 151936
   --kv-channels 128
   --qk-layernorm
)

# ========================= Checkpoint =========================
CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${TORCH_DIST_CHECKPOINT}"
   --load "${SLIME_CHECKPOINT}"
   --save "${SLIME_CHECKPOINT}"
   --save-interval 20
)

# ========================= Rollout 配置 =========================
ROLLOUT_ARGS=(
   --prompt-data "${TRAIN_DATA}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle

   --rm-type deepscaler

   --num-rollout 3000
   --rollout-batch-size 64
   --n-samples-per-prompt 8
   --rollout-max-response-len 16384
   --rollout-temperature 1

   --global-batch-size 512
   --balance-data
)

# ========================= 评估配置 =========================
EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data math_test "${EVAL_DATA}"
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 1
)

# ========================= 并行度与性能 =========================
PERF_ARGS=(
   --tensor-model-parallel-size ${TP_SIZE}
   --sequence-parallel
   --pipeline-model-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 18432
)

# ========================= GRPO 算法 =========================
GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

# ========================= 优化器 =========================
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

# ========================= SGLang =========================
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine ${ROLLOUT_GPUS_PER_ENGINE}
   --sglang-mem-fraction-static 0.85
)

# ========================= 启动训练 =========================
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} \
    --num-gpus ${NUM_GPUS} --disable-usage-stats

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   --attention-dropout 0.0 \
   --hidden-dropout 0.0 \
   --accumulate-allreduce-grads-in-fp32 \
   --attention-softmax-in-fp32 \
   --attention-backend flash
```

**脚本执行流程：**

1. 清理旧进程，确保 GPU 资源干净
2. 配置路径（模型、数据、checkpoint）
3. 设定模型架构参数（需与模型实际架构匹配）
4. 启动 Ray 集群
5. 通过 `ray job submit` 提交训练任务，运行 `train.py`

**关键参数关系：**

```
global_batch_size = rollout_batch_size × n_samples_per_prompt
         512     =        64          ×         8
```

---

## 十四、常见问题

**Q1：Checkpoint 格式有哪些？它们之间是什么关系？**

slime 涉及三种 checkpoint 格式：

| 格式 | 用途 | 说明 |
|------|------|------|
| HuggingFace | 推理引擎（SGLang） | 标准 HF 格式，用于 `--hf-checkpoint` |
| torch_dist | 参考模型 | Megatron 分布式格式，用于 `--ref-load` |
| slime | 训练 checkpoint | 包含优化器状态等，用于 `--load` / `--save` |

转换路径：`HuggingFace → torch_dist`（convert_hf_to_torch_dist.py），训练后可 `slime → HuggingFace`。

**Q2：如何设置合理的 `global-batch-size`？**

`global-batch-size` 应等于一个 rollout 步骤产出的 sample 总数：

```
global_batch_size = rollout_batch_size × n_samples_per_prompt
```

如果设置了 `--num-steps-per-rollout`，则一个 rollout 的数据会分多个 step 训练。

**Q3：`--colocate` 和独立 GPU 分配如何选择？**

- **Colocate**：GPU 少时使用，训练和推理共享 GPU，通过 offload 切换。开销：CPU-GPU 数据搬运。
- **独立分配**：GPU 充裕时推荐，训练和推理各占一组 GPU，可并行执行。需要设置 `--rollout-num-gpus`。

**Q4：多节点训练如何配置？**

```bash
# 主节点
ray start --head --node-ip-address $MASTER_ADDR --num-gpus 8

# 工作节点
ray start --address=$MASTER_ADDR:6379 --num-gpus 8

# 训练脚本
--actor-num-nodes 2
--actor-num-gpus-per-node 8
```

**Q5：训练过程中 reward 一直为 0 怎么办？**

1. 检查 `--rm-type` 是否匹配数据格式
2. 检查 `--label-key` 是否正确指向答案字段
3. 使用 `--save-debug-rollout-data` 保存 rollout 数据进行检查
4. 确认奖励函数能正确提取和比较答案

**Q6：如何只评估不训练？**

```bash
--num-rollout 0
--eval-interval 1
--eval-prompt-data dataset_name /path/to/eval.jsonl
```

设置 `--num-rollout 0` 且配置评估参数即可。

**Q7：如何从中断的训练恢复？**

slime 支持从 checkpoint 恢复训练。设置 `--load` 指向已保存的 checkpoint 路径，`--start-rollout-id` 指定恢复的 rollout 步骤：

```bash
--load /path/to/checkpoint
--start-rollout-id 100
```

---

> 更多信息请参考：
> - [快速开始指南](./docs/zh/get_started/quick_start.md)
> - [使用文档](./docs/zh/get_started/usage.md)
> - [调试指南](./docs/zh/developer_guide/debug.md)
> - [示例代码](./examples/)
> - [参数源码](./slime/utils/arguments.py)
