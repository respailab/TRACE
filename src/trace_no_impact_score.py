"""
trace_no_impact_score.py — TRACE ablation: uniform weights (no impact score).

Phase 2 (gold batch + gJ + impact score computation) is skipped entirely.
All INVERT and PUNISH samples receive impact_score = 1.0. RETAIN = 0.0.
This isolates the contribution of the alignment impact weighting.

Run:
    python src/trace_no_impact_score.py \
        --model_name_or_path <hf-id or /path/to/model> \
        --dataset_json_path  <path/to/triaged.json> \
        --output_dir         runs/trace_no_impact
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from accelerate import Accelerator

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.datamodule import (
    _extract_text, _tokenize_pair, _load_json_dataset,
    _trace_collator, _build_output_dir,
)
from utils.losses import seq_logp, seq_logp_nograd

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False


# =============================================================================
# TRACETrainer (same loss as trace — impact_score field drives weighting)
# =============================================================================

class TRACETrainer(Trainer):
    def __init__(self, ref_model, tokenizer, beta, alpha_kl, max_length, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model  = ref_model
        self.tokenizer  = tokenizer
        self.beta       = beta
        self.alpha_kl   = alpha_kl
        self.max_length = max_length

    def _logp(self, model, text, prompt, grad=True):
        device = next(model.parameters()).device
        batch  = _tokenize_pair(prompt, text, self.tokenizer, device, self.max_length)
        fn = seq_logp if grad else seq_logp_nograd
        return fn(model, batch["input_ids"], batch["attention_mask"], batch["labels"])

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        device = next(model.parameters()).device
        LI  = torch.tensor(0.0, device=device)
        LII = torch.tensor(0.0, device=device)
        LKL = torch.tensor(0.0, device=device)

        prompts    = inputs["prompt"]
        chosens    = inputs["chosen"]
        rejecteds  = inputs["rejected"]
        categories = inputs["category"]
        weights    = inputs["impact_score"]

        for i in range(len(prompts)):
            x        = prompts[i]
            chosen   = chosens[i]
            rejected = rejecteds[i]
            cat      = categories[i]
            w        = float(weights[i])

            if cat == "INVERT":
                logp_c = self._logp(model, chosen,   x)
                logp_r = self._logp(model, rejected, x)
                with torch.no_grad():
                    logp_c_ref = self._logp(self.ref_model, chosen,   x, grad=False)
                    logp_r_ref = self._logp(self.ref_model, rejected, x, grad=False)
                margin = (logp_c - logp_c_ref) - (logp_r - logp_r_ref)
                LI = LI + w * (-F.logsigmoid(self.beta * margin).squeeze())

            elif cat == "PUNISH":
                for resp in (chosen, rejected):
                    logp_t = self._logp(model, resp, x)
                    with torch.no_grad():
                        logp_r = self._logp(self.ref_model, resp, x, grad=False)
                    LII = LII + w * (-F.logsigmoid(-self.beta * (logp_t - logp_r)).squeeze())

            elif cat == "RETAIN":
                batch = _tokenize_pair(x, chosen, self.tokenizer, device, self.max_length)
                curr_logits = model(**{k: batch[k] for k in ("input_ids", "attention_mask")}).logits.float()
                with torch.no_grad():
                    ref_logits = self.ref_model(**{k: batch[k] for k in ("input_ids", "attention_mask")}).logits.float()
                p_ref = F.softmax(ref_logits,  dim=-1)
                p_cur = F.softmax(curr_logits, dim=-1)
                mask  = (batch["labels"] != -100).float()  # response tokens only
                kl    = (p_ref * (torch.log(p_ref + 1e-12) - torch.log(p_cur + 1e-12))).sum(-1)
                LKL   = LKL + (kl * mask).sum() / (mask.sum() + 1e-8)

        loss = LI + LII + self.alpha_kl * LKL
        return (loss, None) if return_outputs else loss


# =============================================================================
# Main
# =============================================================================

def main(args):
    output_dir = _build_output_dir(args.output_dir, args.model_name_or_path, args.dataset_json_path)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Output dir: {output_dir}")

    accelerator = Accelerator()
    if _WANDB and accelerator.is_main_process:
        wandb.init(project="trace-no-impact", name=os.path.basename(output_dir))

    model     = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, cache_dir="cache_dir/", torch_dtype=torch.bfloat16)
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, cache_dir="cache_dir/", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    triaged = _load_json_dataset(args.dataset_json_path)
    counts  = {}
    for s in triaged:
        counts[s["category"]] = counts.get(s["category"], 0) + 1
    print(f"[INFO] Category breakdown: {counts}")

    print("[INFO] Ablation mode: assigning uniform impact_score=1.0 to INVERT/PUNISH.")
    for s in triaged:
        s["impact_score"] = 1.0 if s["category"] in ("INVERT", "PUNISH") else 0.0

    for s in triaged:
        s["chosen"]   = _extract_text(s["chosen"])
        s["rejected"] = _extract_text(s["rejected"])
        s["prompt"]   = _extract_text(s["prompt"])

    dataset = Dataset.from_list(triaged)

    if args.use_lora:
        peft_config = LoraConfig(
            r=32, lora_alpha=16, lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
            bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    ref_model.requires_grad_(False)
    ref_model = ref_model.to(accelerator.device)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        remove_unused_columns=False,
        report_to="wandb" if _WANDB else "none",
    )
    trainer = TRACETrainer(
        ref_model=ref_model,
        tokenizer=tokenizer,
        beta=args.beta,
        alpha_kl=args.alpha_kl,
        max_length=args.max_length,
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=_trace_collator,
    )
    trainer.train()
    trainer.save_model(output_dir)
    print(f"[INFO] Model saved: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TRACE ablation: no impact score (uniform weights).")
    parser.add_argument("--model_name_or_path",          type=str, required=True)
    parser.add_argument("--dataset_json_path",           type=str, required=True)
    parser.add_argument("--output_dir",                  type=str, default="runs/trace_no_impact")
    parser.add_argument("--batch_size",                  type=int,   default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int,   default=4)
    parser.add_argument("--learning_rate",               type=float, default=1e-4)
    parser.add_argument("--num_train_epochs",            type=int,   default=1)
    parser.add_argument("--beta",                        type=float, default=0.3)
    parser.add_argument("--alpha_kl",                    type=float, default=0.1)
    parser.add_argument("--max_length",                  type=int,   default=256)
    parser.add_argument("--use_lora", action="store_true", default=True,
                        help="Wrap model with LoRA (PEFT) adapters.")
    args = parser.parse_args()
    main(args)
