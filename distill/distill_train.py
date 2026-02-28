"""
Token-level Knowledge Distillation for EG Entailment.

Teacher : Llama-3.1-8B-Instruct  (frozen)
Student : Llama-3.2-1B-Instruct + LoRA  (trained)

Both models share the same 131 072-token vocabulary, so per-token KL
divergence is computed directly over the full distribution.

Loss (Hinton et al., 2015):
    L = α · KD(T) + (1 - α) · CE
    KD(T) = KL(p_teacher(T) ‖ p_student(T)) · T²
    CE    = cross-entropy against the hard ground-truth label

KD is applied only to response tokens (everything after the prompt).

Usage:
    python distill_train.py [--option value ...]
"""

import json
import os
import logging
import argparse
from dataclasses import dataclass

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
class DistillConfig:
    # Models
    teacher_model: str = "/mnt/raid0hdd1/liang/models/Llama-3.1-8B-Instruct"
    student_model: str = "/mnt/raid0hdd1/liang/models/Llama-3.2-1B-Instruct"

    # LoRA (student only)
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Data
    data_path: str = "../data/EGs/EG_train-balanced.json"
    val_ratio: float = 0.05
    max_seq_len: int = 160   # prompt (≤128 tokens) + response ("True"/"False" = 1-2 tokens)

    # Distillation
    temperature: float = 2.0   # T — softens both distributions before KL
    alpha: float = 0.9         # weight on KD loss; (1-alpha) goes to hard-label CE

    # Training
    num_epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # Output
    output_dir: str = "saves/llama3.2-1b-distill-EG"
    logging_steps: int = 50
    save_steps: int = 500
    seed: int = 42


# ---------------------------------------------------------------------------
# Prompt format  (same as SFT and GRPO for fair comparison)
# ---------------------------------------------------------------------------

INSTRUCTION_TEMPLATE = (
    'If "{prem}", then "{hypo}", is that true or false? '
    'Answer with only "True" or "False".'
)


def make_chat_prompt(tokenizer, prem: str, hypo: str) -> str:
    messages = [{"role": "user", "content": INSTRUCTION_TEMPLATE.format(prem=prem, hypo=hypo)}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EGDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_len: int,
        split: str = "train",
        val_ratio: float = 0.05,
        seed: int = 42,
    ):
        with open(data_path) as f:
            raw = json.load(f)

        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(raw)).tolist()
        n_val = int(len(raw) * val_ratio)
        indices = indices[:n_val] if split == "val" else indices[n_val:]
        self.items = [raw[i] for i in indices]

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        prompt = make_chat_prompt(self.tokenizer, item["prem"], item["hypo"])
        response = "True" if item["label"] else "False"
        return {"prompt": prompt, "response": response, "label": item["label"]}


def make_collate_fn(tokenizer, max_seq_len: int):
    def collate_fn(batch):
        # Build full sequences: prompt + response (teacher-forced)
        full_texts = [b["prompt"] + b["response"] for b in batch]
        prompt_texts = [b["prompt"] for b in batch]

        full_enc = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        )
        prompt_enc = tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,          # right-pad prompts to find response start
            truncation=True,
            max_length=max_seq_len,
        )
        # prompt_len = length of the longest prompt in this batch (right-padded)
        prompt_len = prompt_enc["input_ids"].shape[1]

        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        return {
            "input_ids": full_enc["input_ids"],
            "attention_mask": full_enc["attention_mask"],
            "prompt_len": prompt_len,
            "labels": labels,
        }

    return collate_fn


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def compute_kd_loss(
    teacher_logits: torch.Tensor,   # (B, L, V)
    student_logits: torch.Tensor,   # (B, L, V)
    response_mask: torch.Tensor,    # (B, L)  1 on response tokens, 0 elsewhere
    temperature: float,
) -> torch.Tensor:
    """
    Forward KL: KL( p_teacher(T) ‖ p_student(T) ) averaged over response tokens.
    Scaled by T² so gradients are comparable across temperatures.
    """
    T = temperature
    # Soften distributions
    log_p_student = F.log_softmax(student_logits / T, dim=-1)   # (B, L, V)
    p_teacher     = F.softmax(teacher_logits.detach() / T, dim=-1)  # (B, L, V)  — no grad

    # KL per position: Σ_v  p_t * (log p_t − log p_s)
    # F.kl_div(input=log Q, target=P) = Σ P*(log P − log Q)
    kl = F.kl_div(log_p_student, p_teacher, reduction="none").sum(dim=-1)  # (B, L)

    # Mask to response positions only and average
    kl_loss = (kl * response_mask).sum() / response_mask.sum().clamp(min=1)
    return kl_loss * (T ** 2)


def compute_ce_loss(
    student_logits: torch.Tensor,   # (B, L, V)
    input_ids: torch.Tensor,        # (B, L)
    response_mask: torch.Tensor,    # (B, L)  1 on response tokens
) -> torch.Tensor:
    """Standard next-token CE loss on response tokens against gold labels."""
    # Shift: logits at position t predict token at t+1
    shift_logits = student_logits[:, :-1, :].contiguous()   # (B, L-1, V)
    shift_ids    = input_ids[:, 1:].contiguous()             # (B, L-1)
    shift_mask   = response_mask[:, 1:].contiguous()         # (B, L-1)

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_ids.view(-1),
        reduction="none",
    ).view(shift_ids.shape)  # (B, L-1)

    return (loss * shift_mask).sum() / shift_mask.sum().clamp(min=1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: DistillConfig):
    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # ---- Tokenizer (shared vocabulary — use student's) ----
    tokenizer = AutoTokenizer.from_pretrained(cfg.student_model)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Teacher (frozen, bf16) ----
    logger.info("Loading teacher model (frozen)...")
    teacher = AutoModelForCausalLM.from_pretrained(
        cfg.teacher_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # ---- Student (LoRA, bf16) ----
    logger.info("Loading student model + LoRA...")
    student_base = AutoModelForCausalLM.from_pretrained(
        cfg.student_model,
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
    student = get_peft_model(student_base, lora_cfg)
    student.print_trainable_parameters()

    # ---- Datasets ----
    collate = make_collate_fn(tokenizer, cfg.max_seq_len)
    train_ds = EGDataset(cfg.data_path, tokenizer, cfg.max_seq_len, "train", cfg.val_ratio, cfg.seed)
    val_ds   = EGDataset(cfg.data_path, tokenizer, cfg.max_seq_len, "val",   cfg.val_ratio, cfg.seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, collate_fn=collate)
    logger.info(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    # ---- Optimizer & scheduler ----
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=0.01,
    )
    total_steps = len(train_loader) * cfg.num_epochs // cfg.gradient_accumulation_steps
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ---- Training loop ----
    global_step = 0
    running = {"loss": 0.0, "kd": 0.0, "ce": 0.0}
    # Infer device from student
    device = next(student.parameters()).device

    for epoch in range(cfg.num_epochs):
        student.train()
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            prompt_len     = batch["prompt_len"]   # scalar int

            # Response mask: 1 on response positions (after prompt), 0 on prompt + padding
            response_mask = torch.zeros_like(input_ids, dtype=torch.float)
            response_mask[:, prompt_len:] = attention_mask[:, prompt_len:].float()

            # ----------------------------------------------------------
            # Teacher forward (no grad, full vocabulary logits)
            # ----------------------------------------------------------
            with torch.no_grad():
                teacher_out = teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
            teacher_logits = teacher_out.logits  # (B, L, V)

            # ----------------------------------------------------------
            # Student forward (with grad)
            # ----------------------------------------------------------
            student_out = student(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            student_logits = student_out.logits  # (B, L, V)

            # ----------------------------------------------------------
            # Losses
            # ----------------------------------------------------------
            kd_loss = compute_kd_loss(teacher_logits, student_logits, response_mask, cfg.temperature)
            ce_loss = compute_ce_loss(student_logits, input_ids, response_mask)

            loss = cfg.alpha * kd_loss + (1.0 - cfg.alpha) * ce_loss
            loss = loss / cfg.gradient_accumulation_steps
            loss.backward()

            # ----------------------------------------------------------
            # Optimizer step
            # ----------------------------------------------------------
            if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in student.parameters() if p.requires_grad],
                    cfg.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                running["loss"] += loss.item() * cfg.gradient_accumulation_steps
                running["kd"]   += kd_loss.item()
                running["ce"]   += ce_loss.item()

                if global_step % cfg.logging_steps == 0:
                    n = cfg.logging_steps
                    logger.info(
                        f"step {global_step} | "
                        f"loss={running['loss']/n:.4f} | "
                        f"kd={running['kd']/n:.4f} | "
                        f"ce={running['ce']/n:.4f}"
                    )
                    running = {k: 0.0 for k in running}

                if global_step % cfg.save_steps == 0:
                    save_path = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
                    student.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    logger.info(f"Saved checkpoint to {save_path}")

    # ---- Final save ----
    student.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    logger.info(f"Training complete. Model saved to {cfg.output_dir}")

    # ---- Validation ----
    evaluate(student, tokenizer, val_loader, device)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, tokenizer, loader, device):
    model.eval()
    correct = 0
    parse_fail = 0
    total = 0

    true_token_id  = tokenizer.encode("True",  add_special_tokens=False)[0]
    false_token_id = tokenizer.encode("False", add_special_tokens=False)[0]

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            prompt_len     = batch["prompt_len"]
            labels         = batch["labels"]   # (B,) 0/1

            # Use only the prompt part for generation evaluation
            prompt_ids   = input_ids[:, :prompt_len]
            prompt_mask  = attention_mask[:, :prompt_len]

            generated = model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            responses = tokenizer.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)

            for resp, label in zip(responses, labels.tolist()):
                text = resp.strip().lower()
                has_true  = "true"  in text
                has_false = "false" in text
                total += 1
                if has_true and not has_false:
                    correct += (label == 1)
                elif has_false and not has_true:
                    correct += (label == 0)
                else:
                    parse_fail += 1

    logger.info(
        f"\n=== Validation ===\n"
        f"  Accuracy    : {correct}/{total} = {correct/total:.4f}\n"
        f"  Parse fails : {parse_fail}/{total} = {parse_fail/total:.4f}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> DistillConfig:
    cfg = DistillConfig()
    parser = argparse.ArgumentParser(description="Knowledge Distillation for EG Entailment")
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
