"""
trace_notriage.py — Ablation: "Punish All" (NPO on every sample, no triage).

All samples receive the NPO loss regardless of INVERT/PUNISH/RETAIN category.
Impact scores are still computed via Algorithm 1+2 (or set uniform with --uniform_weights).
This isolates the contribution of triage.

Run:
    python src/trace_notriage.py \
        --model_name_or_path  Qwen/Qwen2.5-7B-Instruct \
        --dataset_json_path   splits/trace/train1.json \
        --output_dir          runs/trace_notriage

    # Without impact weighting (double ablation):
    ... --uniform_weights
"""

import argparse
import json
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
    get_gold_batch, _build_gold_dataloader,
)
from utils.losses import (
    seq_logp, seq_logp_nograd,
    compute_pa_grads, compute_impact_scores_all,
)

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False


# =============================================================================
# NoTriageTrainer — NPO on ALL samples, category ignored
# =============================================================================

class NoTriageTrainer(Trainer):
    """
    'Punish All' ablation: NPO loss applied to all samples uniformly.
    Loss: w[i] · [-log σ(-β (log πθ(rejected|x) - log πref(rejected|x)))]
    """

    def __init__(self, ref_model, tokenizer, beta, max_length, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model  = ref_model
        self.tokenizer  = tokenizer
        self.beta       = beta
        self.max_length = max_length

    def _logp(self, model, text, prompt, grad=True):
        device = next(model.parameters()).device
        batch  = _tokenize_pair(prompt, text, self.tokenizer, device, self.max_length)
        fn = seq_logp if grad else seq_logp_nograd
        return fn(model, batch["input_ids"], batch["attention_mask"], batch["labels"])

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        device  = next(model.parameters()).device
        L_total = torch.tensor(0.0, device=device)

        prompts   = inputs["prompt"]
        rejecteds = inputs["rejected"]
        weights   = inputs["impact_score"]

        for i in range(len(prompts)):
            x   = prompts[i]
            rej = rejecteds[i]
            w   = float(weights[i])
            logp_t = self._logp(model, rej, x)
            with torch.no_grad():
                logp_r = self._logp(self.ref_model, rej, x, grad=False)
            L_total = L_total + w * (-F.logsigmoid(-self.beta * (logp_t - logp_r)).squeeze())

        return (L_total, None) if return_outputs else L_total


# =============================================================================
# Main
# =============================================================================

def main(args):
    if args.resume_from_checkpoint:
        output_dir = os.path.dirname(os.path.abspath(args.resume_from_checkpoint))
    else:
        output_dir = _build_output_dir(
            args.output_dir, args.model_name_or_path, args.dataset_json_path, args.run_name
        )
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Output dir: {output_dir}")

    accelerator = Accelerator()
    if _WANDB and accelerator.is_main_process:
        wandb.init(project="trace-notriage", name=os.path.basename(output_dir))

    model     = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, cache_dir="cache_dir/", torch_dtype=torch.bfloat16)
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, cache_dir="cache_dir/", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    triaged = _load_json_dataset(args.dataset_json_path, require_category=False)
    counts  = {}
    for s in triaged:
        counts[s["category"]] = counts.get(s["category"], 0) + 1
    print(f"[INFO] Category breakdown (ignored in loss): {counts}")

    weighted_json_path = os.path.join(output_dir, "weighted_dataset.json")

    if args.uniform_weights:
        print("[INFO] Phase 2 SKIPPED — uniform_weights: impact_score=1.0 for all.")
        for s in triaged:
            s["impact_score"] = 1.0

    elif os.path.exists(weighted_json_path):
        print(f"[INFO] Phase 2 SKIPPED — loading cached {weighted_json_path}")
        with open(weighted_json_path) as f:
            triaged = json.load(f)

    else:
        print("[INFO] Phase 2: Computing impact scores for ALL samples (NPO gradient).")
        if args.use_lora:
            peft_config = LoraConfig(
                r=32, lora_alpha=16, lora_dropout=0.1,
                target_modules=["q_proj", "v_proj"],
                bias="none", task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        ref_model.requires_grad_(False)
        model, ref_model = accelerator.prepare(model, ref_model)

        di  = [s for s in triaged if s["category"] == "INVERT"]
        dii = [s for s in triaged if s["category"] == "PUNISH"]
        dr  = [s for s in triaged if s["category"] == "RETAIN"]
        dgold, _ = get_gold_batch(di, dii, dr, args.gold_batch_size)
        gold_dl  = _build_gold_dataloader(dgold, tokenizer, args.pa_batch_size)
        gold_dl  = accelerator.prepare(gold_dl)
        gJ = compute_pa_grads(model, ref_model, gold_dl, accelerator, beta=args.beta)

        triaged = compute_impact_scores_all(triaged, model, ref_model, tokenizer, gJ, args.beta, gamma=args.lamda)

        with open(weighted_json_path, "w", encoding="utf-8") as f:
            json.dump(triaged, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Weighted dataset saved: {weighted_json_path}")

    for s in triaged:
        s["chosen"]   = _extract_text(s["chosen"])
        s["rejected"] = _extract_text(s["rejected"])
        s["prompt"]   = _extract_text(s["prompt"])

    dataset = Dataset.from_list(triaged)

    print("[INFO] Phase 3: Training (no-triage NPO).")
    accelerator.free_memory()
    model     = model.module     if hasattr(model,     "module") else model
    ref_model = ref_model.module if hasattr(ref_model, "module") else ref_model
    ref_model.requires_grad_(False)
    ref_model = ref_model.to(accelerator.device)
    ref_model.eval()

    if args.uniform_weights and args.use_lora:
        peft_config = LoraConfig(
            r=32, lora_alpha=16, lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
            bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=args.num_train_epochs,
        remove_unused_columns=False,
        report_to="wandb" if _WANDB else "none",
    )
    trainer = NoTriageTrainer(
        ref_model=ref_model,
        tokenizer=tokenizer,
        beta=args.beta,
        max_length=args.max_length,
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=_trace_collator,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    trainer.save_model(output_dir)
    print(f"[INFO] Model saved: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TRACE ablation: no triage (NPO on all samples).")
    parser.add_argument("--model_name_or_path",          type=str, required=True)
    parser.add_argument("--dataset_json_path",           type=str, required=True)
    parser.add_argument("--output_dir",                  type=str, default="runs/trace_notriage")
    parser.add_argument("--batch_size",                  type=int,   default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int,   default=4)
    parser.add_argument("--learning_rate",               type=float, default=1e-4)
    parser.add_argument("--num_train_epochs",            type=int,   default=1)
    parser.add_argument("--beta",                        type=float, default=0.3)
    parser.add_argument("--gold_batch_size",             type=int,   default=64)
    parser.add_argument("--pa_batch_size",               type=int,   default=2)
    parser.add_argument("--max_length",                  type=int,   default=256)
    parser.add_argument("--uniform_weights",  action="store_true", default=False)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--run_name",    type=str,   default=None)
    parser.add_argument("--use_lora", action="store_true", default=True,
                        help="Wrap model with LoRA (PEFT) adapters.")
    parser.add_argument("--lamda",    type=float, default=1.0,
                        help="Hessian damping factor γ: scales impact scores by 1/γ before clamping (Algorithm 1 line 28).")
    args = parser.parse_args()
    main(args)
