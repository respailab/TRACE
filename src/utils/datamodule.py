import json
import os
import random
from datetime import datetime

import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


def _extract_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        for turn in reversed(value):
            if isinstance(turn, dict) and turn.get("role") == "assistant":
                return turn["content"]
        return " ".join(t.get("content", "") for t in value if isinstance(t, dict))
    raise TypeError(f"Cannot extract text from {type(value)}: {value!r}")


def _tokenize_pair(prompt: str, response: str, tokenizer, device, max_length: int = 256):
    """Tokenize (prompt, response) → {input_ids, attention_mask, labels, start_locs, weights}."""
    messages = [
        {"role": "user",      "content": prompt},
        {"role": "assistant", "content": response},
    ]
    if tokenizer.chat_template:
        qa_text = tokenizer.apply_chat_template(messages, tokenize=False)
        q_text  = tokenizer.apply_chat_template([messages[0]], tokenize=False)
    else:
        qa_text = f"### Question: {prompt}\n ### Answer: {response}"
        q_text  = f"### Question: {prompt}\n ### Answer: "

    tok_qa = tokenizer(qa_text, truncation=True, padding="max_length", max_length=max_length)
    tok_q  = tokenizer(q_text)
    start  = len(tok_q["input_ids"]) - 1

    ids    = tok_qa["input_ids"]
    mask   = tok_qa["attention_mask"]
    labels = [-100] * start + ids[start:]
    labels = [l if mask[i] else -100 for i, l in enumerate(labels)]

    return {
        "input_ids":      torch.tensor([ids],    dtype=torch.long).to(device),
        "attention_mask": torch.tensor([mask],   dtype=torch.long).to(device),
        "labels":         torch.tensor([labels], dtype=torch.long).to(device),
        "start_locs":     torch.tensor([start],  dtype=torch.long).to(device),
        "weights":        torch.tensor([1.0]).to(device),
    }


def _tokenize_for_dpo(prompt, chosen, rejected, tokenizer, max_prompt=128, max_length=512):
    """Tokenize a (prompt, chosen, rejected) triple for concatenated_forward."""
    prompt_ids = tokenizer(
        prompt, add_special_tokens=False, truncation=True, max_length=max_prompt
    )["input_ids"]

    records = {}
    for response, prefix in ((chosen, "chosen"), (rejected, "rejected")):
        resp_ids = (
            tokenizer(
                response, add_special_tokens=False, truncation=True,
                max_length=max_length - len(prompt_ids),
            )["input_ids"]
            + [tokenizer.eos_token_id]
        )
        full   = prompt_ids + resp_ids
        attn   = [1] * len(full)
        labels = [-100] * len(prompt_ids) + resp_ids
        records[f"{prefix}_input_ids"]      = torch.tensor(full,   dtype=torch.long)
        records[f"{prefix}_attention_mask"] = torch.tensor(attn,   dtype=torch.long)
        records[f"{prefix}_labels"]         = torch.tensor(labels, dtype=torch.long)
    return records


def get_gold_batch(di, dii, dr, B):
    """
    Algorithm 2: build Dgold from triaged lists.

    di  = INVERT samples  (pre-swapped: chosen=yl compliant, rejected=yw violating)
    dii = PUNISH samples  (both violate)
    dr  = RETAIN samples  (chosen complies)
    B   = target batch size

    Returns (dgold, y_compliant).
    """
    y_compliant = (
        [_extract_text(s["chosen"]) for s in dr] +
        [_extract_text(s["chosen"]) for s in di]
    )

    BR = min(len(dr), B // 3)
    BI = min(len(di), B // 3)
    sr = random.sample(dr, BR) if BR > 0 else []
    si = random.sample(di, BI) if BI > 0 else []

    dgold = []
    for s in sr:
        dgold.append({
            "prompt":   _extract_text(s["prompt"]),
            "chosen":   _extract_text(s["chosen"]),
            "rejected": _extract_text(s["rejected"]),
        })
    for s in si:
        dgold.append({
            "prompt":   _extract_text(s["prompt"]),
            "chosen":   _extract_text(s["chosen"]),
            "rejected": _extract_text(s["rejected"]),
        })
    if y_compliant and dii:
        BP = B - len(dgold)
        sp = random.sample(dii, min(len(dii), BP)) if BP > 0 else []
        for s in sp:
            dgold.append({
                "prompt":   _extract_text(s["prompt"]),
                "chosen":   random.choice(y_compliant),
                "rejected": _extract_text(s["chosen"]),  # yw (violating)
            })

    return dgold, y_compliant


def _build_gold_dataloader(dgold, tokenizer, batch_size, max_length=512, max_prompt=128):
    records = {k: [] for k in [
        "chosen_input_ids", "chosen_attention_mask", "chosen_labels",
        "rejected_input_ids", "rejected_attention_mask", "rejected_labels",
    ]}
    for item in tqdm(dgold, desc="Building gold dataloader"):
        row = _tokenize_for_dpo(
            item["prompt"], item["chosen"], item["rejected"],
            tokenizer, max_prompt=max_prompt, max_length=max_length,
        )
        for k in records:
            records[k].append(row[k])

    def collate(batch):
        out = {}
        for key in batch[0]:
            pad_val = -100 if "labels" in key else 0
            out[key] = pad_sequence([b[key] for b in batch], batch_first=True, padding_value=pad_val)
        return out

    dataset = [{k: records[k][i] for k in records} for i in range(len(dgold))]
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate, shuffle=False)


def _load_json_dataset(path, require_category=True):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    required = {"prompt", "chosen", "rejected", "category"}
    for i, item in enumerate(data):
        missing = required - set(item.keys())
        if missing:
            raise ValueError(f"Item {i} missing fields: {missing}")
        if require_category and item["category"] not in ("INVERT", "PUNISH", "RETAIN"):
            raise ValueError(f"Item {i} unknown category '{item['category']}'")
    print(f"[INFO] Loaded {len(data)} samples from {path}")
    return data


def _trace_collator(features):
    """Collate mixed text+numeric batch."""
    batch = {}
    for key in features[0]:
        vals = [f[key] for f in features]
        if isinstance(vals[0], str):
            batch[key] = vals
        elif isinstance(vals[0], (int, float)):
            batch[key] = torch.tensor(vals, dtype=torch.float32)
        else:
            batch[key] = vals
    return batch


def _build_output_dir(base, model_path, dataset_path, run_name=None):
    if run_name:
        return os.path.join(base, run_name)
    mb = os.path.basename(model_path.rstrip("/\\")) or model_path
    db = os.path.splitext(os.path.basename(dataset_path))[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base, f"{mb}__{db}__{ts}")
