import os
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))          # trace/utils/
_TRACE = os.path.dirname(_HERE)                              # trace/
_ROOT = os.path.dirname(_TRACE)                              # repo root
for _p in (_TRACE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.common import get_trainable_params
from utils.reward import _get_batch_logps, concatenated_forward, preference_loss
from utils.datamodule import _tokenize_pair, _extract_text


# ---------------------------------------------------------------------------
# Sequence log-prob
# ---------------------------------------------------------------------------

def seq_logp(model, input_ids, attention_mask, labels):
    """Sum log-prob of response tokens (with grad)."""
    with torch.set_grad_enabled(True):
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
    return _get_batch_logps(logits, labels, average_log_prob=False)


def seq_logp_nograd(model, input_ids, attention_mask, labels):
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
    return _get_batch_logps(logits, labels, average_log_prob=False)


# ---------------------------------------------------------------------------
# Gradient helpers
# ---------------------------------------------------------------------------

def logp_grad_flat(model, prompt, response, tokenizer, device):
    """∂ log πθ(response|prompt) / ∂θ — flattened, graph freed."""
    params = get_trainable_params(model)
    model.zero_grad()
    batch = _tokenize_pair(prompt, response, tokenizer, device)
    logp  = seq_logp(model, batch["input_ids"], batch["attention_mask"], batch["labels"])
    torch.cuda.empty_cache()
    gL = torch.cat([g.view(-1) for g in torch.autograd.grad(logp, params, retain_graph=False)])
    model.zero_grad()
    torch.cuda.empty_cache()
    return gL


def npo_grad_flat(model, ref_model, prompt, response, tokenizer, beta, device):
    """∇[-log σ(-β log πθ(y)/πref(y))] — flattened, graph freed."""
    params = get_trainable_params(model)
    model.zero_grad()
    batch  = _tokenize_pair(prompt, response, tokenizer, device)
    logp_t = seq_logp(model, batch["input_ids"], batch["attention_mask"], batch["labels"])
    with torch.no_grad():
        logp_r = seq_logp_nograd(ref_model, batch["input_ids"], batch["attention_mask"], batch["labels"])
    loss = -F.logsigmoid(-beta * (logp_t - logp_r)).mean()
    torch.cuda.empty_cache()
    gL = torch.cat([g.view(-1) for g in torch.autograd.grad(loss, params, retain_graph=False)])
    model.zero_grad()
    torch.cuda.empty_cache()
    return gL


# ---------------------------------------------------------------------------
# PA gradient (gJ)
# ---------------------------------------------------------------------------

def compute_pa_grads(model, ref_model, gold_dataloader, accelerator, beta=0.3):
    """Compute ∇J on the gold batch at θ=θref."""
    model.zero_grad()
    for batch in tqdm(gold_dataloader, desc="Computing gJ on gold batch"):
        policy_chosen_logps, policy_rejected_logps = concatenated_forward(model, batch)
        with torch.no_grad():
            ref_chosen_logps, ref_rejected_logps = concatenated_forward(ref_model, batch)
        losses = preference_loss(
            policy_chosen_logps, policy_rejected_logps,
            ref_chosen_logps, ref_rejected_logps,
            beta=beta, label_smoothing=0, ipo=False, reference_free=False,
        )
        accelerator.backward(losses.mean())

    trainable = get_trainable_params(model)
    gJ = torch.cat([p.grad.view(-1) for p in trainable if p.grad is not None])
    model.zero_grad()
    torch.cuda.empty_cache()
    return gJ


# ---------------------------------------------------------------------------
# Per-sample impact scores
# ---------------------------------------------------------------------------

def _impact_invert(sample, model, ref_model, tokenizer, gJ, beta, device):
    """
    INVERT impact score (Algorithm 1 line 16).
    INVERT pairs are pre-swapped: chosen=yl (compliant), rejected=yw (violating).
    """
    prompt = _extract_text(sample["prompt"])
    yl     = _extract_text(sample["chosen"])    # compliant
    yw     = _extract_text(sample["rejected"])  # violating

    batch_l = _tokenize_pair(prompt, yl, tokenizer, device)
    batch_w = _tokenize_pair(prompt, yw, tokenizer, device)

    with torch.no_grad():
        logp_yl_t = seq_logp_nograd(model,     batch_l["input_ids"], batch_l["attention_mask"], batch_l["labels"])
        logp_yw_t = seq_logp_nograd(model,     batch_w["input_ids"], batch_w["attention_mask"], batch_w["labels"])
        logp_yl_r = seq_logp_nograd(ref_model, batch_l["input_ids"], batch_l["attention_mask"], batch_l["labels"])
        logp_yw_r = seq_logp_nograd(ref_model, batch_w["input_ids"], batch_w["attention_mask"], batch_w["labels"])
    delta = (logp_yl_t - logp_yl_r) - (logp_yw_t - logp_yw_r)
    c     = (-torch.sigmoid(-beta * delta) * beta).squeeze()
    del logp_yl_t, logp_yw_t, logp_yl_r, logp_yw_r, delta
    torch.cuda.empty_cache()

    gL_yl = logp_grad_flat(model, prompt, yl, tokenizer, device)
    gL_yw = logp_grad_flat(model, prompt, yw, tokenizer, device)
    gL    = c * (gL_yl - gL_yw)
    del gL_yl, gL_yw

    score = torch.dot(gJ, gL).item()
    del gL
    torch.cuda.empty_cache()
    return score


def _impact_punish(sample, model, ref_model, tokenizer, gJ, beta, device):
    """
    PUNISH impact score — dual-NPO gradient (Algorithm 1 line 24).
    PUNISH convention: chosen=yw (violating), rejected=yl (also violating).
    """
    prompt = _extract_text(sample["prompt"])
    yw     = _extract_text(sample["chosen"])
    yl     = _extract_text(sample["rejected"])

    gL_yw = npo_grad_flat(model, ref_model, prompt, yw, tokenizer, beta, device)
    gL_yl = npo_grad_flat(model, ref_model, prompt, yl, tokenizer, beta, device)
    gL    = gL_yw + gL_yl
    del gL_yw, gL_yl

    score = torch.dot(gJ, gL).item()
    del gL
    torch.cuda.empty_cache()
    return score


def compute_impact_scores(triaged_list, di, dii, model, ref_model, tokenizer, gJ, beta, gamma=1.0):
    """
    Compute INVERT+PUNISH impact scores (Alg. 1 lines 28-31):
      line 28: w[k] ← (1/γ) w[k]   — damping by gamma
      line 29: clamp to >= 0
      lines 30-31: L1-normalize
    Writes impact_score in-place on each sample dict.
    """
    if gamma == 0.0:
        raise ValueError("gamma (--lamda) must be non-zero; 1/γ scaling is undefined at γ=0.")
    device = next(model.parameters()).device
    scores = {}

    for sample in tqdm(di,  desc="Impact scores (INVERT)"):
        scores[id(sample)] = _impact_invert(sample, model, ref_model, tokenizer, gJ, beta, device)
    for sample in tqdm(dii, desc="Impact scores (PUNISH)"):
        scores[id(sample)] = _impact_punish(sample, model, ref_model, tokenizer, gJ, beta, device)

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


def compute_impact_scores_all(triaged_list, model, ref_model, tokenizer, gJ, beta, gamma=1.0):
    """
    NPO impact scores for ALL samples (no-triage ablation).
    Applies gamma damping, clamp >= 0, L1-normalize. Writes impact_score in-place.
    """
    if gamma == 0.0:
        raise ValueError("gamma (--lamda) must be non-zero; 1/γ scaling is undefined at γ=0.")
    device = next(model.parameters()).device
    scores = {}

    for s in tqdm(triaged_list, desc="Impact scores (all, NPO)"):
        prompt = _extract_text(s["prompt"])
        resp   = _extract_text(s["rejected"])
        gL     = npo_grad_flat(model, ref_model, prompt, resp, tokenizer, beta, device)
        scores[id(s)] = torch.dot(gJ, gL).item()
        del gL
        torch.cuda.empty_cache()

    all_keys = [id(s) for s in triaged_list]
    for k in all_keys:
        scores[k] /= gamma                   # Algorithm 1 line 28: w[k] ← (1/γ) w[k]
    for k in all_keys:
        scores[k] = max(0.0, scores[k])
    Z = sum(scores[k] for k in all_keys) or 1.0
    for k in all_keys:
        scores[k] /= Z

    for s in triaged_list:
        s["impact_score"] = scores.get(id(s), 0.0)
    return triaged_list
