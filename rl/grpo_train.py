"""
GRPO (Group Relative Policy Optimization) for EG Entailment.

Algorithm (DeepSeekMath, Shao et al. 2024):
  For each batch of prompts:
  1. Generate G completions per prompt
  2. Score each completion: +1 correct, -1 wrong, -0.5 format error
  3. Compute group-relative advantage:  A_i = (r_i - mean(r)) / std(r)
  4. GRPO loss = -E[min(ratio*A, clip(ratio, 1±ε)*A)] + β*KL(π_θ || π_ref)
     where π_ref = base model (LoRA disabled), so no extra memory needed.

Usage:
    python grpo_train.py [--config grpo_config overrides via argparse]
"""

import json
import re
import os
import argparse
import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GRPOConfig:
    # Model
    model_name_or_path: str = "/mnt/raid0hdd1/liang/models/Llama-3.2-1B-Instruct"
    # Optionally start from an SFT LoRA checkpoint (set to "" to use base model)
    sft_lora_path: str = ""

    # LoRA (applied on top of base model)
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Data
    data_path: str = "../data/EGs/EG_train-balanced.json"
    val_ratio: float = 0.05

    # GRPO core
    num_generations: int = 4     # G: completions per prompt
    max_new_tokens: int = 16     # just "True" or "False" + buffer
    max_prompt_len: int = 128
    temperature: float = 0.8
    kl_coeff: float = 0.04       # β: KL penalty weight
    clip_eps: float = 0.2        # PPO clipping ε

    # Training
    num_epochs: int = 5
    batch_size: int = 4          # prompts per step (× G = total sequences)
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-7
    warmup_steps: int = 20
    max_grad_norm: float = 1.0

    # Output
    output_dir: str = "saves/llama3.2-1b-grpo-EG"
    logging_steps: int = 10
    save_steps: int = 1000
    seed: int = 42


# ---------------------------------------------------------------------------
# Prompt & reward
# ---------------------------------------------------------------------------

INSTRUCTION_TEMPLATE = (
    'If "{prem}", then "{hypo}", is that true or false? '
    'Answer with only "True" or "False".'
)


def make_chat_prompt(tokenizer, prem: str, hypo: str) -> str:
    messages = [{"role": "user", "content": INSTRUCTION_TEMPLATE.format(prem=prem, hypo=hypo)}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def compute_reward(response: str, label: bool) -> float:
    """
    +1.0  : correct prediction
    -1.0  : wrong prediction
    -0.5  : cannot parse True/False from response
    """
    text = response.strip().lower()
    has_true = bool(re.search(r"\btrue\b", text))
    has_false = bool(re.search(r"\bfalse\b", text))

    if has_true and not has_false:
        pred = True
    elif has_false and not has_true:
        pred = False
    else:
        return -0.5  # ambiguous or empty

    return 1.0 if (pred == label) else -1.0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EGDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_prompt_len: int, split: str = "train", val_ratio: float = 0.05, seed: int = 42):
        with open(data_path) as f:
            raw = json.load(f)

        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(raw)).tolist()
        n_val = int(len(raw) * val_ratio)

        if split == "val":
            indices = indices[:n_val]
        else:
            indices = indices[n_val:]

        self.items = [raw[i] for i in indices]
        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        prompt_text = make_chat_prompt(self.tokenizer, item["prem"], item["hypo"])
        return {"prompt": prompt_text, "label": item["label"]}


def collate_fn(batch):
    return {
        "prompts": [b["prompt"] for b in batch],
        "labels": [b["label"] for b in batch],
    }


# ---------------------------------------------------------------------------
# Core GRPO helpers
# ---------------------------------------------------------------------------

def get_response_logprobs(
    model,
    input_ids: torch.Tensor,       # (B, total_len)
    attention_mask: torch.Tensor,  # (B, total_len)
    prompt_len: int,               # number of prompt tokens (same for all after left-padding)
) -> torch.Tensor:
    """
    Return per-token log-probs for response tokens only.
    Shape: (B, response_len)
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (B, total_len, V)

    # Shift: logits[:, t, :] predicts token at position t+1
    # Response starts at prompt_len; first response token is at position prompt_len
    # Its logit is at position prompt_len - 1
    response_logits = logits[:, prompt_len - 1 : -1, :]  # (B, response_len, V)
    response_ids = input_ids[:, prompt_len:]              # (B, response_len)

    log_probs = F.log_softmax(response_logits, dim=-1)
    token_logprobs = log_probs.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)
    return token_logprobs  # (B, response_len)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over non-padding positions."""
    return (values * mask).sum() / mask.sum().clamp(min=1)


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

def train(cfg: GRPOConfig):
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.output_dir, exist_ok=True)

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
    tokenizer.padding_side = "left"  # required for generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Model ----
    logger.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules="all-linear",
        bias="none",
    )
    model = get_peft_model(base_model, lora_cfg)

    if cfg.sft_lora_path:
        logger.info(f"Loading SFT LoRA weights from {cfg.sft_lora_path}")
        from peft import set_peft_model_state_dict
        import safetensors.torch as st
        weights = st.load_file(os.path.join(cfg.sft_lora_path, "adapter_model.safetensors"))
        set_peft_model_state_dict(model, weights)

    model.print_trainable_parameters()

    # ---- Dataset ----
    train_dataset = EGDataset(cfg.data_path, tokenizer, cfg.max_prompt_len, "train", cfg.val_ratio, cfg.seed)
    val_dataset   = EGDataset(cfg.data_path, tokenizer, cfg.max_prompt_len, "val",   cfg.val_ratio, cfg.seed)
    train_loader  = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    logger.info(f"Train: {len(train_dataset)}  Val: {len(val_dataset)}")

    # ---- Optimizer & scheduler ----
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=0.01,
    )
    total_steps = len(train_loader) * cfg.num_epochs // cfg.gradient_accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg.warmup_steps, total_steps)

    # ---- Training loop ----
    global_step = 0
    running = {"loss": 0.0, "kl": 0.0, "reward": 0.0, "acc": 0.0}

    for epoch in range(cfg.num_epochs):
        model.train()
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            prompts: list[str] = batch["prompts"]   # B strings
            labels:  list[bool] = batch["labels"]   # B bools
            B = len(prompts)

            # ----------------------------------------------------------
            # Step 1: Tokenize prompts (left-padded for generation)
            # ----------------------------------------------------------
            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=cfg.max_prompt_len,
            ).to(device)
            prompt_len = enc["input_ids"].shape[1]  # padded prompt length

            # ----------------------------------------------------------
            # Step 2: Generate G completions per prompt
            #   Repeat each prompt G times → shape (G*B, prompt_len)
            # ----------------------------------------------------------
            rep_input_ids = enc["input_ids"].repeat_interleave(cfg.num_generations, dim=0)
            rep_attn_mask = enc["attention_mask"].repeat_interleave(cfg.num_generations, dim=0)

            with torch.no_grad():
                generated = model.generate(
                    input_ids=rep_input_ids,
                    attention_mask=rep_attn_mask,
                    max_new_tokens=cfg.max_new_tokens,
                    do_sample=True,
                    temperature=cfg.temperature,
                    pad_token_id=tokenizer.eos_token_id,
                )
            # generated: (G*B, prompt_len + response_len)
            response_ids = generated[:, prompt_len:]   # (G*B, response_len)
            response_len = response_ids.shape[1]

            # Decode responses
            response_texts = tokenizer.batch_decode(response_ids, skip_special_tokens=True)

            # ----------------------------------------------------------
            # Step 3: Compute rewards  →  shape (G*B,)
            # ----------------------------------------------------------
            rewards = torch.zeros(cfg.num_generations * B, device=device)
            for i, (text, label) in enumerate(zip(response_texts, labels * cfg.num_generations)):
                # labels is repeated: [l0,l1,...,lB, l0,l1,...,lB, ...]
                rewards[i] = compute_reward(text, label)

            # Rearrange to (G, B) for group statistics
            rewards_gb = rewards.view(cfg.num_generations, B)  # (G, B)

            # ----------------------------------------------------------
            # Step 4: Group-relative advantages  (G, B)
            # ----------------------------------------------------------
            mean_r = rewards_gb.mean(dim=0, keepdim=True)   # (1, B)
            std_r  = rewards_gb.std(dim=0, keepdim=True).clamp(min=1e-8)
            advantages_gb = (rewards_gb - mean_r) / std_r   # (G, B)
            advantages = advantages_gb.view(-1)              # (G*B,)

            # ----------------------------------------------------------
            # Step 5: Build full sequences for log-prob computation
            #   full_seq = [padded_prompt | response] (G*B, total_len)
            # ----------------------------------------------------------
            response_mask = (response_ids != tokenizer.pad_token_id).long()  # (G*B, response_len)

            full_ids  = generated                                              # (G*B, total_len)
            full_mask = torch.cat([rep_attn_mask, response_mask], dim=1)       # (G*B, total_len)

            # ----------------------------------------------------------
            # Step 6: Old log-probs  (at generation time, no grad)
            # ----------------------------------------------------------
            with torch.no_grad():
                old_logprobs = get_response_logprobs(model, full_ids, full_mask, prompt_len)
                # (G*B, response_len)

            # ----------------------------------------------------------
            # Step 7: Reference log-probs  (base model, LoRA disabled)
            # ----------------------------------------------------------
            with torch.no_grad(), model.disable_adapter():
                ref_logprobs = get_response_logprobs(model, full_ids, full_mask, prompt_len)
                # (G*B, response_len)

            # ----------------------------------------------------------
            # Step 8: GRPO loss  (with gradient)
            # ----------------------------------------------------------
            new_logprobs = get_response_logprobs(model, full_ids, full_mask, prompt_len)
            # (G*B, response_len)

            # Per-token advantage: broadcast advantage per sequence to all its tokens
            adv = advantages.unsqueeze(1).expand_as(new_logprobs)  # (G*B, response_len)
            mask = response_mask.float()

            # Policy loss (PPO-style clipping)
            log_ratio = new_logprobs - old_logprobs.detach()
            ratio = torch.exp(log_ratio)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv
            policy_loss = -masked_mean(torch.min(surr1, surr2), mask)

            # Per-token KL: KL(π_θ || π_ref) ≈ log(π_θ/π_ref)
            kl = masked_mean(new_logprobs - ref_logprobs.detach(), mask)

            loss = policy_loss + cfg.kl_coeff * kl
            loss = loss / cfg.gradient_accumulation_steps
            loss.backward()

            # ----------------------------------------------------------
            # Step 9: Optimizer step (every gradient_accumulation_steps)
            # ----------------------------------------------------------
            if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Accumulate metrics
                with torch.no_grad():
                    acc = ((rewards > 0).float().mean()).item()
                running["loss"]   += policy_loss.item()
                running["kl"]     += kl.item()
                running["reward"] += rewards.mean().item()
                running["acc"]    += acc

                if global_step % cfg.logging_steps == 0:
                    n = cfg.logging_steps
                    logger.info(
                        f"step {global_step} | "
                        f"loss={running['loss']/n:.4f} | "
                        f"kl={running['kl']/n:.4f} | "
                        f"reward={running['reward']/n:.3f} | "
                        f"acc={running['acc']/n:.3f}"
                    )
                    running = {k: 0.0 for k in running}

                if global_step % cfg.save_steps == 0:
                    save_path = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    logger.info(f"Saved checkpoint to {save_path}")

    # ---- Final save ----
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    logger.info(f"Training complete. Model saved to {cfg.output_dir}")

    # ---- Validation ----
    evaluate(model, tokenizer, val_dataset, cfg, device)


# ---------------------------------------------------------------------------
# Evaluation (greedy decode, report accuracy)
# ---------------------------------------------------------------------------

def evaluate(model, tokenizer, dataset: EGDataset, cfg: GRPOConfig, device):
    model.eval()
    correct = 0
    parse_fail = 0
    loader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            enc = tokenizer(
                batch["prompts"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=cfg.max_prompt_len,
            ).to(device)
            prompt_len = enc["input_ids"].shape[1]
            generated = model.generate(
                **enc,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            responses = tokenizer.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)
            for resp, label in zip(responses, batch["labels"]):
                r = compute_reward(resp, label)
                if r == -0.5:
                    parse_fail += 1
                elif r == 1.0:
                    correct += 1

    total = len(dataset)
    logger.info(
        f"\n=== Validation ===\n"
        f"  Accuracy    : {correct}/{total} = {correct/total:.4f}\n"
        f"  Parse fails : {parse_fail}/{total} = {parse_fail/total:.4f}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> GRPOConfig:
    cfg = GRPOConfig()
    parser = argparse.ArgumentParser(description="GRPO fine-tuning for EG Entailment")
    for field_name, field_val in cfg.__dict__.items():
        t = type(field_val)
        if t == bool:
            parser.add_argument(f"--{field_name}", type=lambda x: x.lower() == "true", default=field_val)
        else:
            parser.add_argument(f"--{field_name}", type=t, default=field_val)
    args = parser.parse_args()
    for k, v in vars(args).items():
        setattr(cfg, k, v)
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    logger.info("Config:\n" + "\n".join(f"  {k}={v}" for k, v in cfg.__dict__.items()))
    train(cfg)
