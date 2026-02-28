# FrequencyBias: Comparing Post-Training Methods for Entailment Judgement

This project studies how different **post-training methods** affect an LLM's ability to judge textual entailment — specifically, whether frequency bias in pre-training data influences entailment predictions. We use **Llama-3.2-1B-Instruct** as the base model and compare Supervised Fine-Tuning (SFT) against Reinforcement Learning (GRPO) on entailment datasets.

---

## Task

Given a **premise** and a **hypothesis**, predict whether the hypothesis is entailed:

```
If "france showed to add job in christmas-day", then "france show import during christmas-day", is that true or false?
→ True
```

---

## Datasets

### EG (Training)
`data/EGs/EG_train-balanced.json` — **10,952 examples** (balanced: 5,476 True / 5,476 False)

Each entry:
```json
{"prem": "france showed to add job in christmas-day",
 "hypo": "france show import during christmas-day",
 "label": true}
```

### Levy/Holt (Evaluation)
`data/LevyHolt/` — standard entailment benchmark for out-of-distribution evaluation.

| Split | Examples |
|-------|----------|
| `dev.txt` | 1,098 |
| `test.txt` | 12,921 |
| `lvholt_test_llama2.json` | 1,784 |

LevyHolt entries also include negated versions of premise/hypothesis (`neg_prem`, `neg_hypo`) for probing negation sensitivity.

---

## Post-Training Methods

### 1. Supervised Fine-Tuning (SFT) — `finetune/`

Standard instruction fine-tuning using **LLaMA Factory** with **LoRA**.

**Setup:**
```bash
# Step 1: Convert raw dataset to alpaca instruction format
cd finetune
python convert_data.py

# Step 2: Run LoRA SFT
bash run_train.sh 0          # single GPU
bash run_train.sh 0,1        # two GPUs
```

Key settings ([train_lora.yaml](finetune/train_lora.yaml)):

| Parameter | Value |
|-----------|-------|
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target modules | all linear |
| Learning rate | 2e-4 |
| Epochs | 3 |
| Batch size (effective) | 32 |
| Eval split | 5% |

Checkpoints saved to `finetune/saves/llama3.2-1b-lora-EG/`.

---

### 2. Reinforcement Learning — GRPO — `rl/`

**Group Relative Policy Optimization** (DeepSeekMath, Shao et al. 2024) with a rule-based reward signal — no reward model needed.

**Algorithm:**
1. For each prompt, generate **G = 4** completions via sampling
2. Score each with a rule-based reward: `+1` correct, `−1` wrong, `−0.5` unparseable
3. Compute **group-relative advantage**: `A_i = (r_i − mean(r)) / std(r)`
4. Optimize: `L = −E[clip(ratio, 1±ε) · A] + β · KL(π_θ ∥ π_ref)`

The reference policy `π_ref` is the base model without LoRA adapters (`model.disable_adapter()`), so only **one model copy** is needed in GPU memory.

**Setup:**
```bash
# From base model
bash rl/run_grpo.sh 0

# From SFT checkpoint (recommended for warm start)
bash rl/run_grpo.sh 0 --sft_lora_path finetune/saves/llama3.2-1b-lora-EG

# Custom hyperparams
bash rl/run_grpo.sh 0 --num_generations 8 --kl_coeff 0.1 --learning_rate 1e-6
```

Key settings ([grpo_train.py](rl/grpo_train.py)):

| Parameter | Value |
|-----------|-------|
| LoRA rank | 16 |
| Generations per prompt (G) | 4 |
| Temperature | 0.8 |
| KL coefficient (β) | 0.04 |
| PPO clip ε | 0.2 |
| Learning rate | 5e-7 |
| Effective batch | 16 sequences |

Checkpoints saved to `rl/saves/llama3.2-1b-grpo-EG/`.

---

## Method Comparison

| | SFT | GRPO |
|---|---|---|
| Supervision | Ground-truth label (cross-entropy) | Rule-based reward (±1) |
| Reference model | None | Base model (LoRA disabled) |
| Exploration | None | G=4 samples per prompt |
| Reward model | Not needed | Not needed |
| Learning rate | 2e-4 | 5e-7 |
| Package | LLaMA Factory | transformers + peft |

---

## Project Structure

```
FrequenctBias/
├── data/
│   ├── EGs/
│   │   ├── EG_train-balanced.json   # training set (10,952 examples)
│   │   └── train.txt                # raw format
│   └── LevyHolt/
│       ├── dev.txt                  # dev split (1,098)
│       ├── test.txt                 # test split (12,921)
│       └── lvholt_test_llama2.json  # pre-processed test (1,784)
│
├── finetune/                        # SFT with LLaMA Factory
│   ├── convert_data.py              # convert EG JSON → alpaca format
│   ├── train_lora.yaml              # LLaMA Factory training config
│   ├── run_train.sh                 # launch script
│   └── data/
│       ├── EG_entailment_train.json # converted dataset
│       └── dataset_info.json        # LLaMA Factory registry
│
└── rl/                              # GRPO (self-contained)
    ├── grpo_train.py                # GRPO training script
    └── run_grpo.sh                  # launch script
```

---

## Environment

```bash
conda activate tuneEG
# Key packages: transformers, peft, accelerate, trl, vllm
```

- **Base model:** `/mnt/raid0hdd1/liang/models/Llama-3.2-1B-Instruct`
- **LLaMA Factory:** `/mnt/nvme0n1/luzhan/finetune-qwen/LLaMA-Factory` (v0.9.3)
- **Python:** `/home/liang/miniconda3/envs/tuneEG/bin/python`
