"""
trace.py — Full TRACE Algorithm (Algorithm 1 + Algorithm 2).

Run:
    python src/trace.py \
        --model_name_or_path <hf-id or /path/to/model> \
        --dataset_json_path  <path/to/triaged.json> \
        --output_dir         runs/trace
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

_HERE = os.path.dirname(os.path.abspath(__file__))   # src/
_ROOT = os.path.dirname(_HERE)                        # repo root
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
    compute_pa_grads, compute_impact_scores,
)

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False


# =============================================================================
# TRACETrainer
# =============================================================================

class TRACETrainer(Trainer):
    """
    Algorithm 1 lines 34-46:
      LI    = Σ w[i] · [-log σ(β Δ_θ(xi, chosen, rejected))]   (INVERT)
      LII   = Σ w[j] · dual-NPO(xj, chosen, rejected)           (PUNISH)
      LKL   = Σ KL(πref(·|xk) ∥ πθ(·|xk))                     (RETAIN)
      LTRACE = LI + LII + αKL·LKL
    """

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
        wandb.init(project="trace-full", name=os.path.basename(output_dir))

    model     = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, cache_dir="cache_dir/", torch_dtype=torch.bfloat16)
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, cache_dir="cache_dir/", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    triaged = _load_json_dataset(args.dataset_json_path)
    di  = [s for s in triaged if s["category"] == "INVERT"]
    dii = [s for s in triaged if s["category"] == "PUNISH"]
    dr  = [s for s in triaged if s["category"] == "RETAIN"]
    print(f"[INFO] INVERT={len(di)}  PUNISH={len(dii)}  RETAIN={len(dr)}")

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

    print("[INFO] Building gold batch (Algorithm 2).")
    dgold, _ = get_gold_batch(di, dii, dr, args.gold_batch_size)
    print(f"[INFO] Gold batch size: {len(dgold)}")

    gold_dl = _build_gold_dataloader(dgold, tokenizer, args.pa_batch_size)
    gold_dl = accelerator.prepare(gold_dl)

    print("[INFO] Computing PA gradient gJ.")
    gJ = compute_pa_grads(model, ref_model, gold_dl, accelerator, beta=args.beta)

    print("[INFO] Computing impact scores.")
    triaged = compute_impact_scores(triaged, di, dii, model, ref_model, tokenizer, gJ, args.beta, gamma=args.lamda)

    weighted_path = os.path.join(output_dir, "weighted_dataset.json")
    with open(weighted_path, "w", encoding="utf-8") as f:
        json.dump(triaged, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Weighted dataset saved: {weighted_path}")

    for s in triaged:
        s["chosen"]   = _extract_text(s["chosen"])
        s["rejected"] = _extract_text(s["rejected"])
        s["prompt"]   = _extract_text(s["prompt"])

    dataset = Dataset.from_list(triaged)

    print("[INFO] Phase 3: Training TRACE.")
    accelerator.free_memory()
    model     = model.module     if hasattr(model,     "module") else model
    ref_model = ref_model.module if hasattr(ref_model, "module") else ref_model
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
    parser = argparse.ArgumentParser(description="TRACE full algorithm (Alg. 1 + 2).")
    parser.add_argument("--model_name_or_path",          type=str, required=True)
    parser.add_argument("--dataset_json_path",           type=str, required=True)
    parser.add_argument("--output_dir",                  type=str, default="runs/trace")
    parser.add_argument("--batch_size",                  type=int,   default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int,   default=4)
    parser.add_argument("--learning_rate",               type=float, default=1e-4)
    parser.add_argument("--num_train_epochs",            type=int,   default=1)
    parser.add_argument("--beta",                        type=float, default=0.3)
    parser.add_argument("--alpha_kl",                    type=float, default=0.1)
    parser.add_argument("--gold_batch_size",             type=int,   default=64)
    parser.add_argument("--pa_batch_size",               type=int,   default=2)
    parser.add_argument("--max_length",                  type=int,   default=256)
    parser.add_argument("--use_lora", action="store_true", default=True,
                        help="Wrap model with LoRA (PEFT) adapters.")
    parser.add_argument("--lamda",    type=float, default=1.0,
                        help="Hessian damping factor γ: scales impact scores by 1/γ before clamping (Algorithm 1 line 28).")
    args = parser.parse_args()
    main(args)
