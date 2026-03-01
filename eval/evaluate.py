"""
Evaluation script for EG Entailment post-training comparison.

Evaluates any combination of:
  - base   : Llama-3.2-1B-Instruct (no fine-tuning)
  - sft    : base + SFT LoRA  (finetune/saves/...)
  - grpo   : base + GRPO LoRA (rl/saves/...)
  - distill: base + Distill LoRA (distill/saves/...)

Test set: data/LevyHolt/lvholt_test_llama2.json
  fields: prem, hypo, label, neg_prem, neg_hypo

Metrics reported per model:
  - Accuracy, Precision, Recall, F1  (standard)
  - Parse-fail rate  (model didn't output "True"/"False")
  - Negation consistency (optional --negation flag):
      % of examples where the model flips its answer
      when prem/hypo are negated (expected behaviour varies by case)

Usage:
    # Evaluate all available models
    python evaluate.py --models base sft grpo distill

    # Single model with negation probing
    python evaluate.py --models sft --negation

    # Custom checkpoint path
    python evaluate.py --models sft --sft_path path/to/adapter
"""

import json
import re
import os
import argparse
import logging
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ── Defaults ────────────────────────────────────────────────────────────────

BASE_MODEL   = "/mnt/raid0hdd1/liang/models/Llama-3.2-1B-Instruct"
SFT_PATH     = "../finetune/saves/llama3.2-1b-lora-EG"
GRPO_PATH    = "../rl/saves/llama3.2-1b-grpo-EG"
DISTILL_PATH = "../distill/saves/llama3.2-1b-distill-EG"
TEST_DATA    = "../data/LevyHolt/lvholt_test_llama2.json"

INSTRUCTION_TEMPLATE = (
    'If "{prem}", then "{hypo}", is that true or false? '
    'Answer with only "True" or "False".'
)

# ── Prompt helpers ───────────────────────────────────────────────────────────

def make_prompt(tokenizer, prem: str, hypo: str) -> str:
    messages = [{"role": "user", "content": INSTRUCTION_TEMPLATE.format(prem=prem, hypo=hypo)}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parse_response(text: str) -> Optional[bool]:
    """Return True/False/None (None = parse failure)."""
    t = text.strip().lower()
    has_true  = bool(re.search(r"\btrue\b",  t))
    has_false = bool(re.search(r"\bfalse\b", t))
    if has_true and not has_false:
        return True
    if has_false and not has_true:
        return False
    return None


# ── Dataset ──────────────────────────────────────────────────────────────────

class LevyHoltDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, negation: bool = False):
        with open(data_path) as f:
            self.raw = json.load(f)
        self.tokenizer = tokenizer
        self.negation   = negation  # if True, also build negated prompts

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, idx):
        item = self.raw[idx]
        entry = {
            "prem"     : item["prem"],
            "hypo"     : item["hypo"],
            "label"    : item["label"],
            "prompt"   : make_prompt(self.tokenizer, item["prem"], item["hypo"]),
        }
        if self.negation:
            entry["neg_prompt"] = make_prompt(self.tokenizer, item["neg_prem"], item["neg_hypo"])
        return entry


def collate_fn(batch):
    out = {k: [b[k] for b in batch] for k in batch[0]}
    return out


# ── Model loading ────────────────────────────────────────────────────────────

def load_model(base_model_path: str, adapter_path: Optional[str], device: torch.device):
    logger.info(f"Loading base model from {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if adapter_path:
        if not os.path.isdir(adapter_path):
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")
        logger.info(f"Loading LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()   # merge LoRA for faster inference

    model.eval()
    return tokenizer, model


# ── Inference ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, tokenizer, prompts: list[str], batch_size: int, max_new_tokens: int, device) -> list[str]:
    responses = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(device)
        prompt_len = enc["input_ids"].shape[1]
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        decoded = tokenizer.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
        responses.extend(decoded)
    return responses


# ── Probability scores for AUC ───────────────────────────────────────────────

@torch.no_grad()
def get_true_false_scores(model, tokenizer, prompts: list[str], batch_size: int, device) -> list[float]:
    """
    For each prompt, return P("True") at the first response token position.

    We run a forward pass on the prompt only (no generation), read the logits
    at the last prompt position, then softmax over just the two tokens {"True",
    "False"} to get a calibrated probability.  This score is used for ROC-AUC.
    """
    true_id  = tokenizer.encode("True",  add_special_tokens=False)[0]
    false_id = tokenizer.encode("False", add_special_tokens=False)[0]

    scores = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=256,
        ).to(device)

        outputs   = model(**enc)
        # logits[:, -1, :] = distribution over next token (= first response token)
        last_logits = outputs.logits[:, -1, :]                           # (B, V)
        tf_logits   = last_logits[:, [true_id, false_id]]                # (B, 2)
        tf_probs    = torch.softmax(tf_logits.float(), dim=-1)           # (B, 2)
        scores.extend(tf_probs[:, 0].cpu().tolist())                     # P(True)
    return scores


@torch.no_grad()
def get_norm_scores(model, tokenizer, prompts: list[str], batch_size: int, device) -> list[float]:
    """
    Compute S_ent for each prompt using:

        S_ent = 0.5 + 0.5 * I[tok=True]  * S_tok
                    - 0.5 * I[tok=False] * S_tok

    where tok is the argmax of the full vocabulary distribution and
    S_tok = softmax(full_logits)[tok_id] is its raw probability.

    If the argmax token is neither "True" nor "False" (parse failure),
    S_ent = 0.5 (maximum uncertainty, contributes nothing to AUC).
    """
    true_id  = tokenizer.encode("True",  add_special_tokens=False)[0]
    false_id = tokenizer.encode("False", add_special_tokens=False)[0]

    s_ent_list = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=256,
        ).to(device)

        outputs     = model(**enc)
        last_logits = outputs.logits[:, -1, :].float()      # (B, V)
        full_probs  = torch.softmax(last_logits, dim=-1)    # (B, V)
        argmax_ids  = last_logits.argmax(dim=-1)            # (B,)

        for b in range(last_logits.shape[0]):
            tok_id = argmax_ids[b].item()
            s_tok  = full_probs[b, tok_id].item()
            if tok_id == true_id:
                s_ent = 0.5 + 0.5 * s_tok
            elif tok_id == false_id:
                s_ent = 0.5 - 0.5 * s_tok
            else:
                s_ent = 0.5   # parse failure → maximum uncertainty
            s_ent_list.append(s_ent)

    return s_ent_list


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(labels: list[bool], preds: list[Optional[bool]], scores: list[float] = None,
                    norm_scores: list[float] = None) -> dict:
    """Compute accuracy, P/R/F1, and parse-fail rate."""
    total      = len(labels)
    parse_fail = sum(1 for p in preds if p is None)
    valid_idx  = [i for i, p in enumerate(preds) if p is not None]

    if not valid_idx:
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0,
                "parse_fail": parse_fail, "parse_fail_rate": 1.0, "total": total}

    y_true = [int(labels[i]) for i in valid_idx]
    y_pred = [int(preds[i])  for i in valid_idx]

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

    int_labels = [int(l) for l in labels]

    # AUC: P(True) from softmax over {True, False} tokens
    auc = None
    if scores is not None:
        try:
            auc = round(roc_auc_score(int_labels, scores), 4)
        except Exception:
            auc = None

    # AUC_norm: S_ent from the full-vocabulary probability formula
    auc_norm = None
    if norm_scores is not None:
        try:
            auc_norm = round(roc_auc_score(int_labels, norm_scores), 4)
        except Exception:
            auc_norm = None

    return {
        "accuracy"       : round(acc,  4),
        "precision"      : round(float(prec), 4),
        "recall"         : round(float(rec),  4),
        "f1"             : round(float(f1),   4),
        "auc"            : auc,
        "auc_norm"       : auc_norm,
        "parse_fail"     : parse_fail,
        "parse_fail_rate": round(parse_fail / total, 4),
        "total"          : total,
        "evaluated"      : len(valid_idx),
    }


def compute_negation_consistency(
    orig_preds: list[Optional[bool]],
    neg_preds:  list[Optional[bool]],
    labels:     list[bool],
) -> dict:
    """
    Negation consistency:
      For a TRUE example:  original=True  and neg=False is CONSISTENT
                           (negating both prem and hypo should flip entailment)
      For a FALSE example: original=False and neg=True  is CONSISTENT
    Flip rate = fraction of examples where model changed its answer on negated input.
    """
    both_valid = [(i, o, n, l) for i, (o, n, l) in enumerate(zip(orig_preds, neg_preds, labels))
                  if o is not None and n is not None]

    if not both_valid:
        return {"negation_flip_rate": None, "negation_consistent_rate": None}

    flipped    = sum(1 for _, o, n, _ in both_valid if o != n)
    consistent = sum(1 for _, o, n, l in both_valid if (l and o == True and n == False) or
                                                         (not l and o == False and n == True))
    n = len(both_valid)
    return {
        "negation_flip_rate"      : round(flipped    / n, 4),
        "negation_consistent_rate": round(consistent / n, 4),
        "negation_evaluated"      : n,
    }


# ── Per-model evaluation ─────────────────────────────────────────────────────

def evaluate_model(
    name: str,
    base_model_path: str,
    adapter_path: Optional[str],
    data_path: str,
    batch_size: int,
    max_new_tokens: int,
    negation: bool,
    output_dir: str,
    device: torch.device,
) -> dict:
    tokenizer, model = load_model(base_model_path, adapter_path, device)

    dataset = LevyHoltDataset(data_path, tokenizer, negation=negation)
    loader  = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    all_labels, all_prompts, all_neg_prompts = [], [], []
    for batch in loader:
        all_labels.extend(batch["label"])
        all_prompts.extend(batch["prompt"])
        if negation:
            all_neg_prompts.extend(batch["neg_prompt"])

    logger.info(f"[{name}] Running inference on {len(all_prompts)} examples...")
    responses = run_inference(model, tokenizer, all_prompts, batch_size, max_new_tokens, device)
    preds     = [parse_response(r) for r in responses]

    logger.info(f"[{name}] Computing P(True) scores for AUC...")
    scores      = get_true_false_scores(model, tokenizer, all_prompts, batch_size, device)
    logger.info(f"[{name}] Computing S_ent scores for AUC_norm...")
    norm_scores = get_norm_scores(model, tokenizer, all_prompts, batch_size, device)
    metrics     = compute_metrics(all_labels, preds, scores, norm_scores)

    neg_metrics = {}
    if negation:
        logger.info(f"[{name}] Running negation inference...")
        neg_responses = run_inference(model, tokenizer, all_neg_prompts, batch_size, max_new_tokens, device)
        neg_preds     = [parse_response(r) for r in neg_responses]
        neg_metrics   = compute_negation_consistency(preds, neg_preds, all_labels)

    result = {"model": name, **metrics, **neg_metrics}

    # Save per-example predictions
    os.makedirs(output_dir, exist_ok=True)
    pred_path = os.path.join(output_dir, f"{name}_predictions.jsonl")
    with open(pred_path, "w") as f:
        for i, (label, resp, pred, score, ns) in enumerate(
                zip(all_labels, responses, preds, scores, norm_scores)):
            row = {"idx": i, "label": label, "response": resp.strip(), "pred": pred,
                   "p_true": round(score, 6), "s_ent": round(ns, 6)}
            if negation and i < len(neg_responses):
                row["neg_response"] = neg_responses[i].strip()
                row["neg_pred"]     = neg_preds[i]
            f.write(json.dumps(row) + "\n")
    logger.info(f"[{name}] Predictions saved to {pred_path}")

    # Free GPU memory before next model
    del model
    torch.cuda.empty_cache()

    return result


# ── Pretty print ─────────────────────────────────────────────────────────────

def print_results(results: list[dict], negation: bool):
    cols = ["model", "accuracy", "auc", "auc_norm", "f1", "precision", "recall", "parse_fail_rate"]
    if negation:
        cols += ["negation_flip_rate", "negation_consistent_rate"]

    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in results)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    sep    = "  ".join("-" * widths[c] for c in cols)

    print("\n" + "=" * len(header))
    print("LevyHolt Evaluation Results")
    print("=" * len(header))
    print(header)
    print(sep)
    for r in results:
        print("  ".join(str(r.get(c, "N/A")).ljust(widths[c]) for c in cols))
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Named groups for convenience
    MODEL_CHOICES = ["standard", "sft", "grpo", "distill", "all"]
    #   standard  → base Llama-3.2-1B-Instruct (no fine-tuning)
    #   sft       → SFT LoRA checkpoint
    #   grpo      → GRPO LoRA checkpoint
    #   distill   → Knowledge Distillation LoRA checkpoint
    #   all       → standard + sft + grpo + distill

    parser = argparse.ArgumentParser(
        description="Evaluate entailment models on LevyHolt",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--models", nargs="+", default=["standard"],
        choices=MODEL_CHOICES,
        metavar="MODEL",
        help=(
            "Which model(s) to evaluate. Choices:\n"
            "  standard  – Llama-3.2-1B-Instruct, no fine-tuning\n"
            "  sft       – SFT LoRA (finetune/saves/...)\n"
            "  grpo      – GRPO LoRA (rl/saves/...)\n"
            "  distill   – Distillation LoRA (distill/saves/...)\n"
            "  all       – run all four above\n"
            "Example: --models standard sft"
        ),
    )
    parser.add_argument("--base_model",     default=BASE_MODEL)
    parser.add_argument("--sft_path",       default=SFT_PATH)
    parser.add_argument("--grpo_path",      default=GRPO_PATH)
    parser.add_argument("--distill_path",   default=DISTILL_PATH)
    parser.add_argument("--test_data",      default=TEST_DATA)
    parser.add_argument("--batch_size",     type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--negation",       action="store_true",
                        help="Also evaluate on negated premise/hypothesis")
    parser.add_argument("--output_dir",     default="results")
    args = parser.parse_args()

    # Expand "all" shorthand
    requested = args.models
    if "all" in requested:
        requested = ["standard", "sft", "grpo", "distill"]
    # Deduplicate while preserving order
    seen, ordered = set(), []
    for m in requested:
        if m not in seen:
            seen.add(m)
            ordered.append(m)
    requested = ordered

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    script_dir = Path(__file__).parent
    def resolve(p):
        return str((script_dir / p).resolve()) if not os.path.isabs(p) else p

    adapter_paths = {
        "standard": None,
        "sft"     : resolve(args.sft_path),
        "grpo"    : resolve(args.grpo_path),
        "distill" : resolve(args.distill_path),
    }
    test_data  = resolve(args.test_data)
    output_dir = resolve(args.output_dir)

    all_results = []
    for name in requested:
        adapter = adapter_paths[name]
        if name != "standard" and not os.path.isdir(adapter):
            logger.warning(f"Skipping '{name}': checkpoint not found at {adapter}")
            continue

        logger.info(f"\n{'='*50}\nEvaluating: {name}\n{'='*50}")
        result = evaluate_model(
            name            = name,
            base_model_path = args.base_model,
            adapter_path    = adapter,
            data_path       = test_data,
            batch_size      = args.batch_size,
            max_new_tokens  = args.max_new_tokens,
            negation        = args.negation,
            output_dir      = output_dir,
            device          = device,
        )
        all_results.append(result)
        logger.info(f"[{name}] Accuracy={result['accuracy']:.4f}  AUC={result['auc']}  "
                    f"AUC_norm={result['auc_norm']}  F1={result['f1']:.4f}  "
                    f"Parse-fail={result['parse_fail_rate']:.4f}")

    if all_results:
        print_results(all_results, args.negation)

        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
