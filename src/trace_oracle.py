"""
trace_oracle.py — TRACE Algorithm (Algorithm 1 + Algorithm 2) with
oracle-guided DPO correction for the PUNISH (D_II) branch.

Paper (Algorithm 1, lines 20-23 and 36-40):
  If oracle O is available for a PUNISH sample (xj, yw, yl):
    yc ← O(xj)
    Impact: gLj = ∇θ[-log σ(β Δθref(xj, yc, yw))]   (DPO gradient)
    Train:  LII += w[j] · [-log σ(β Δθ(xj, yc, yw))]  (DPO loss, prefer yc over yw)

Run:
    python src/trace_oracle.py \\
        --model_name_or_path <hf-id or /path/to/model> \\
        --dataset_json_path  <path/to/triaged.json> \\
        --output_dir         runs/trace_oracle \\
        --oracle_max_tokens  512
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
from tqdm import tqdm

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
    logp_grad_flat,
    compute_pa_grads, compute_impact_scores,
)
from utils.oracle import oracle_guided_punish_preference_pair

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False


# =============================================================================
# Oracle response generation
# =============================================================================

def _generate_oracle_responses(dii, max_tokens=512):
    """
    Pre-generate oracle corrections yc for every PUNISH sample.
    Stores result in sample["oracle_response"] in-place.
    Returns the count of successful generations.
    """
    ok = 0
    for sample in tqdm(dii, desc="Oracle corrections (PUNISH)"):
        pair = {
            "prompt": _extract_text(sample["prompt"]),
            "chosen": _extract_text(sample["chosen"]),
        }
        try:
            sample["oracle_response"] = oracle_guided_punish_preference_pair(pair, max_tokens=max_tokens)
            ok += 1
        except Exception as e:
            print(f"[WARN] Oracle call failed ({e}); sample will fall back to dual-NPO.")
            sample["oracle_response"] = ""
    print(f"[INFO] Oracle corrections generated: {ok}/{len(dii)}")
    return ok


# =============================================================================
# Impact score — oracle-guided PUNISH (Algorithm 1 lines 20-23)
# =============================================================================

def _impact_punish_oracle(sample, model, ref_model, tokenizer, gJ, beta, device):
    """
    PUNISH impact score with oracle correction yc (Algorithm 1 line 22-23):
      gLj = ∇θ[-log σ(β Δθref(xj, yc, yw))]
    where yc is the oracle correction and yw is the original harmful chosen.
    """
    prompt = _extract_text(sample["prompt"])
    yw     = _extract_text(sample["chosen"])   # original harmful chosen
    yc     = sample["oracle_response"]

    # DPO gradient: prefer yc over yw
    batch_c = _tokenize_pair(prompt, yc, tokenizer, device)
    batch_w = _tokenize_pair(prompt, yw, tokenizer, device)

    with torch.no_grad():
        logp_yc_t = seq_logp_nograd(model,     batch_c["input_ids"], batch_c["attention_mask"], batch_c["labels"])
        logp_yw_t = seq_logp_nograd(model,     batch_w["input_ids"], batch_w["attention_mask"], batch_w["labels"])
        logp_yc_r = seq_logp_nograd(ref_model, batch_c["input_ids"], batch_c["attention_mask"], batch_c["labels"])
        logp_yw_r = seq_logp_nograd(ref_model, batch_w["input_ids"], batch_w["attention_mask"], batch_w["labels"])

    delta = (logp_yc_t - logp_yc_r) - (logp_yw_t - logp_yw_r)
    c     = (-torch.sigmoid(-beta * delta) * beta).squeeze()
    del logp_yc_t, logp_yw_t, logp_yc_r, logp_yw_r, delta
    torch.cuda.empty_cache()

    gL_yc = logp_grad_flat(model, prompt, yc, tokenizer, device)
    gL_yw = logp_grad_flat(model, prompt, yw, tokenizer, device)
    gL    = c * (gL_yc - gL_yw)
    del gL_yc, gL_yw

    score = torch.dot(gJ, gL).item()
    del gL
    torch.cuda.empty_cache()
    return score


def compute_impact_scores_oracle(triaged_list, di, dii, model, ref_model, tokenizer, gJ, beta, gamma=1.0):
    """
    Like compute_impact_scores but uses oracle DPO gradient for PUNISH samples.
    Applies gamma damping, clamp >= 0, L1-normalize (Algorithm 1 lines 28-31).
    Writes impact_score in-place on each sample dict.
    """
    device = next(model.parameters()).device
    scores = {}

    for sample in tqdm(di,  desc="Impact scores (INVERT)"):
        from utils.losses import _impact_invert
        scores[id(sample)] = _impact_invert(sample, model, ref_model, tokenizer, gJ, beta, device)
    for sample in tqdm(dii, desc="Impact scores (PUNISH, oracle)"):
        scores[id(sample)] = _impact_punish_oracle(sample, model, ref_model, tokenizer, gJ, beta, device)

    if gamma == 0.0:
        raise ValueError("gamma (--lamda) must be non-zero; 1/γ scaling is undefined at γ=0.")
    conflict_keys = [id(s) for s in di + dii]
    for k in conflict_keys:
        scores[k] /= gamma                   # Algorithm 1 line 28: w[k] ← (1/γ) w[k]
    for k in conflict_keys:
        scores[k] = max(0.0, scores[k])
    Z = sum(scores[k] for k in conflict_keys) or 1.0
    for k in conflict_keys:
        scores[k] /= Z

    for sample in triaged_list:
        if sample["category"] in ("INVERT", "PUNISH"):
            sample["impact_score"] = scores.get(id(sample), 0.0)
        else:
            sample["impact_score"] = 0.0

    return triaged_list


# =============================================================================
# OracleTRACETrainer
# =============================================================================

class OracleTRACETrainer(Trainer):
    """
    Algorithm 1 lines 34-46 with oracle-guided DPO for PUNISH:
      LI    = Σ w[i] · [-log σ(β Δθ(xi, yl, yw))]              (INVERT: standard DPO)
      LII   = Σ w[j] · [-log σ(β Δθ(xj, yc, yw))]   if yc      (PUNISH: oracle DPO)
            = Σ w[j] · dual-NPO(xj, yw, yl)           otherwise  (fallback)
      LKL   = Σ KL(πref(·|xk) ∥ πθ(·|xk))                      (RETAIN)
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

        prompts          = inputs["prompt"]
        chosens          = inputs["chosen"]
        rejecteds        = inputs["rejected"]
        categories       = inputs["category"]
        weights          = inputs["impact_score"]
        oracle_responses = inputs["oracle_response"]

        for i in range(len(prompts)):
            x        = prompts[i]
            chosen   = chosens[i]    # INVERT: yl (compliant); PUNISH: yw (harmful)
            rejected = rejecteds[i]
            cat      = categories[i]
            w        = float(weights[i])
            yc       = oracle_responses[i]

            if cat == "INVERT":
                # Standard DPO: prefer chosen (yl, compliant) over rejected (yw, harmful)
                logp_c = self._logp(model, chosen,   x)
                logp_r = self._logp(model, rejected, x)
                with torch.no_grad():
                    logp_c_ref = self._logp(self.ref_model, chosen,   x, grad=False)
                    logp_r_ref = self._logp(self.ref_model, rejected, x, grad=False)
                margin = (logp_c - logp_c_ref) - (logp_r - logp_r_ref)
                LI = LI + w * (-F.logsigmoid(self.beta * margin).squeeze())

            elif cat == "PUNISH":
                # Oracle DPO: prefer yc (oracle correction) over chosen (yw, harmful)
                logp_c = self._logp(model, yc,     x)
                logp_w = self._logp(model, chosen, x)
                with torch.no_grad():
                    logp_c_ref = self._logp(self.ref_model, yc,     x, grad=False)
                    logp_w_ref = self._logp(self.ref_model, chosen, x, grad=False)
                margin = (logp_c - logp_c_ref) - (logp_w - logp_w_ref)
                LII = LII + w * (-F.logsigmoid(self.beta * margin).squeeze())

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
        wandb.init(project="trace-full-oracle", name=os.path.basename(output_dir))

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

    # Pre-generate oracle corrections for all PUNISH samples (main process only).
    # oracle_response is stored in each dii sample dict in-place.
    if accelerator.is_main_process:
        print("[INFO] Generating oracle corrections for PUNISH samples.")
        _generate_oracle_responses(dii, max_tokens=args.oracle_max_tokens)
    accelerator.wait_for_everyone()

    # Ensure RETAIN and INVERT samples also have oracle_response field (empty).
    for s in di + dr:
        s.setdefault("oracle_response", "")

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

    print("[INFO] Computing impact scores (oracle-guided PUNISH).")
    triaged = compute_impact_scores_oracle(triaged, di, dii, model, ref_model, tokenizer, gJ, args.beta, gamma=args.lamda)

    weighted_path = os.path.join(output_dir, "weighted_dataset.json")
    with open(weighted_path, "w", encoding="utf-8") as f:
        json.dump(triaged, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Weighted dataset (with oracle responses) saved: {weighted_path}")

    for s in triaged:
        s["chosen"]   = _extract_text(s["chosen"])
        s["rejected"] = _extract_text(s["rejected"])
        s["prompt"]   = _extract_text(s["prompt"])

    dataset = Dataset.from_list(triaged)

    print("[INFO] Phase 3: Training TRACE (oracle-guided PUNISH branch).")
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
    trainer = OracleTRACETrainer(
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
    parser = argparse.ArgumentParser(description="TRACE full algorithm with oracle-guided DPO for PUNISH (D_II).")
    parser.add_argument("--model_name_or_path",          type=str, required=True)
    parser.add_argument("--dataset_json_path",           type=str, required=True)
    parser.add_argument("--output_dir",                  type=str, default="runs/trace_oracle")
    parser.add_argument("--batch_size",                  type=int,   default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int,   default=4)
    parser.add_argument("--learning_rate",               type=float, default=1e-4)
    parser.add_argument("--num_train_epochs",            type=int,   default=1)
    parser.add_argument("--beta",                        type=float, default=0.3)
    parser.add_argument("--alpha_kl",                    type=float, default=0.1)
    parser.add_argument("--gold_batch_size",             type=int,   default=64)
    parser.add_argument("--pa_batch_size",               type=int,   default=2)
    parser.add_argument("--max_length",                  type=int,   default=256)
    parser.add_argument("--oracle_max_tokens", type=int,   default=512,
                        help="Max tokens for oracle GPT-4o correction responses.")
    parser.add_argument("--use_lora",          action="store_true", default=True,
                        help="Wrap model with LoRA (PEFT) adapters.")
    parser.add_argument("--lamda",             type=float, default=1.0,
                        help="Hessian damping factor γ: scales impact scores by 1/γ before clamping (Algorithm 1 line 28).")
    args = parser.parse_args()
    main(args)
