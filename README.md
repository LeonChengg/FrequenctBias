# FrequencyBias: Comparing Post-Training Methods for Entailment Judgement

This project studies how different **post-training methods** affect an LLM's ability to judge textual entailment — specifically, whether frequency bias in pre-training data influences entailment predictions. We use **Llama-3.2-1B-Instruct** as the base model and compare three post-training methods: Supervised Fine-Tuning (SFT), Reinforcement Learning (GRPO), and Knowledge Distillation (KD).

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

### 3. Knowledge Distillation (KD) — `distill/`

**Token-level forward KL distillation** from a larger teacher model. The student learns to match the teacher's full output distribution over the vocabulary at each response token — richer than SFT's hard labels.

**Teacher:** `Llama-3.1-8B-Instruct` (frozen)
**Student:** `Llama-3.2-1B-Instruct` + LoRA (trained)
Both share the same 131,072-token vocabulary, so per-token KL is directly applicable.

**Loss (Hinton et al., 2015):**
```
L = α · KD(T) + (1 − α) · CE
KD(T) = KL( p_teacher(T) ‖ p_student(T) ) · T²
```
- `KD(T)`: forward KL with temperature `T` to soften distributions, scaled by `T²` to keep gradient magnitude stable
- `CE`: standard cross-entropy against the gold label, for grounding
- Both losses applied **only to response tokens** (not the prompt)

**Setup:**
```bash
bash distill/run_distill.sh 0          # single GPU

# Tune distillation strength
bash distill/run_distill.sh 0 --temperature 4.0 --alpha 0.7
bash distill/run_distill.sh 0 --alpha 1.0   # pure KD, no CE loss
```

Key settings ([distill_train.py](distill/distill_train.py)):

| Parameter | Value |
|-----------|-------|
| Teacher | Llama-3.1-8B-Instruct |
| LoRA rank | 16 |
| Temperature (T) | 2.0 |
| α (KD weight) | 0.9 |
| Learning rate | 1e-4 |
| Epochs | 3 |
| Effective batch | 32 |

Checkpoints saved to `distill/saves/llama3.2-1b-distill-EG/`.

---

## Method Comparison

| | SFT | GRPO | Distillation |
|---|---|---|---|
| Supervision signal | Hard label (CE) | Rule-based reward (±1) | Teacher distribution (KL) |
| Teacher model | None | None | Llama-3.1-8B (frozen) |
| Reference model | None | Base model (LoRA off) | None |
| Exploration | None | G=4 samples/prompt | None |
| Reward model | Not needed | Not needed | Not needed |
| Learning rate | 2e-4 | 5e-7 | 1e-4 |
| Conda env | `tuneEG` | `tuneEG` | `distll` |
| Package | LLaMA Factory | transformers + peft | transformers + peft |

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
├── rl/                              # GRPO reinforcement learning
│   ├── grpo_train.py                # GRPO training script
│   └── run_grpo.sh                  # launch script
│
└── distill/                         # Knowledge distillation (8B → 1B)
    ├── distill_train.py             # KD training script
    └── run_distill.sh               # launch script
```

---

## Environment

| Method | Conda env | Key packages |
|--------|-----------|--------------|
| SFT | `tuneEG` | transformers 4.57, peft 0.11, trl 0.9.4, accelerate 0.31 |
| GRPO | `tuneEG` | transformers 4.57, peft 0.11, accelerate 0.31 |
| Distillation | `distll` | transformers 4.56, peft 0.17, accelerate 1.10, torch 2.8 |

- **Student / base model:** `/mnt/raid0hdd1/liang/models/Llama-3.2-1B-Instruct`
- **Teacher model (KD):** `/mnt/raid0hdd1/liang/models/Llama-3.1-8B-Instruct`
- **LLaMA Factory (SFT):** `/mnt/nvme0n1/luzhan/finetune-qwen/LLaMA-Factory` (v0.9.3)
