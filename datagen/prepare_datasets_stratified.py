"""
prepare_datasets_stratified.py — Split DPO-Gold checkpoint with stratified sampling.

Ensures RETAIN/INVERT/PUNISH distribution is consistent across train1, train2, test.

Splits:
  train1 = --train1_size samples (stratified by category)
  train2 = --train2_size samples (stratified subset of train1)
  test   = remaining (held out, same distribution)

Usage:
    python prepare_datasets_stratified.py --ckpt_json pku_dpo_gold_13k.json
"""

import argparse
import json
import os
import random


def load_json(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: list, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(data):>6} samples → {path}")


def dist_str(samples):
    counts = {}
    for s in samples:
        c = s.get("category", "N/A")
        counts[c] = counts.get(c, 0) + 1
    total = len(samples)
    return "  ".join(f"{k}: {v} ({v/total*100:.1f}%)" for k, v in sorted(counts.items()))


def main(args):
    print(f"[INFO] Loading: {args.ckpt_json}")
    data = load_json(args.ckpt_json)
    total = len(data)
    print(f"[INFO] Total samples: {total}")

    if args.train1_size > total:
        raise ValueError(f"train1_size ({args.train1_size}) > dataset size ({total})")

    rng = random.Random(args.seed)

    # ── Group by category ─────────────────────────────────────────────────────
    bins = {}
    for item in data:
        cat = item.get("category", "UNKNOWN")
        bins.setdefault(cat, []).append(item)

    for cat in bins:
        rng.shuffle(bins[cat])

    print(f"\n[INFO] Full dataset distribution:")
    for cat, items in sorted(bins.items()):
        print(f"  {cat:8s}: {len(items):>6}  ({len(items)/total*100:.1f}%)")

    # ── Compute proportional allocation for train1 ────────────────────────────
    train1_per_bin = {
        cat: round(args.train1_size * len(items) / total)
        for cat, items in bins.items()
    }
    # Fix rounding drift
    diff = args.train1_size - sum(train1_per_bin.values())
    for cat in list(bins.keys())[:abs(diff)]:
        train1_per_bin[cat] += 1 if diff > 0 else -1

    print(f"\n[INFO] Stratified allocation — train1 ({args.train1_size}):")
    for cat in sorted(bins):
        available = len(bins[cat])
        needed = train1_per_bin[cat]
        status = "OK" if available >= needed else f"!! INSUFFICIENT (need {needed}, have {available})"
        print(f"  {cat:8s}: {needed:>5} / {available:>6}  {status}")
        if available < needed:
            raise ValueError(f"Not enough {cat} samples: need {needed}, have {available}")

    # ── Compute proportional allocation for train2 ────────────────────────────
    train2_per_bin = {
        cat: round(args.train2_size * train1_per_bin[cat] / args.train1_size)
        for cat in bins
    }
    diff2 = args.train2_size - sum(train2_per_bin.values())
    for cat in list(bins.keys())[:abs(diff2)]:
        train2_per_bin[cat] += 1 if diff2 > 0 else -1

    print(f"\n[INFO] Stratified allocation — train2 ({args.train2_size}, subset of train1):")
    for cat in sorted(bins):
        print(f"  {cat:8s}: {train2_per_bin[cat]:>5} / {train1_per_bin[cat]:>5}")

    # ── Split ─────────────────────────────────────────────────────────────────
    train1, test = [], []
    train1_by_cat = {}
    for cat, items in bins.items():
        n = train1_per_bin[cat]
        train1.extend(items[:n])
        test.extend(items[n:])
        train1_by_cat[cat] = items[:n]

    train2 = []
    for cat, n in train2_per_bin.items():
        pool = train1_by_cat.get(cat, [])
        train2.extend(rng.sample(pool, min(n, len(pool))))

    rng.shuffle(train1)
    rng.shuffle(train2)
    rng.shuffle(test)

    # ── Distribution check ────────────────────────────────────────────────────
    print(f"\n[INFO] Final split distributions:")
    print(f"  train1 ({len(train1):>6}): {dist_str(train1)}")
    print(f"  train2 ({len(train2):>6}): {dist_str(train2)}")
    print(f"  test   ({len(test):>6}): {dist_str(test)}")

    # ── Save TRACE format ─────────────────────────────────────────────────────
    trace_dir = os.path.join(args.output_dir, "trace")
    os.makedirs(trace_dir, exist_ok=True)
    print(f"\n[INFO] Saving TRACE splits → {trace_dir}/")
    save_json(train1, os.path.join(trace_dir, "train1.json"))
    save_json(train2, os.path.join(trace_dir, "train2.json"))
    save_json(test,   os.path.join(trace_dir, "test.json"))

    # ── Save DPO format (prompt/chosen/rejected only) ─────────────────────────
    def to_dpo(samples):
        return [{"prompt": s["prompt"], "chosen": s["chosen"], "rejected": s["rejected"]}
                for s in samples]

    dpo_dir = os.path.join(args.output_dir, "dpo")
    os.makedirs(dpo_dir, exist_ok=True)
    print(f"\n[INFO] Saving DPO splits → {dpo_dir}/")
    save_json(to_dpo(train1), os.path.join(dpo_dir, "train1.json"))
    save_json(to_dpo(train2), os.path.join(dpo_dir, "train2.json"))
    save_json(to_dpo(test),   os.path.join(dpo_dir, "test.json"))

    print(f"\n[DONE] All splits saved to {args.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_json",   type=str, required=True)
    parser.add_argument("--output_dir",  type=str, default="splits")
    parser.add_argument("--train1_size", type=int, default=10000)
    parser.add_argument("--train2_size", type=int, default=5000)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()
    main(args)
