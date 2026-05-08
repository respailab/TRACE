"""
prepare_datasets.py — Split a DPO-Gold checkpoint into train/test sets.

Produces ready-to-run datasets for:
  1. TRACE  (trace.py) — JSON with prompt/chosen/rejected/category
  2. DPO    (dpo_accelerate.py) — JSON with prompt/chosen/rejected

Splits:
  train1 = 10,000 samples (random)
  train2 = 5,000 randomly sampled FROM train1
  test   = remaining samples (not in train1)

Usage:
    python prepare_datasets.py --ckpt_json dpo_gold_checkpoints/dpo_gold_checkpoint_0070000.json
    python prepare_datasets.py --ckpt_json dpo_gold_checkpoints/dpo_gold_checkpoint_0070000.json \\
        --train1_size 10000 --train2_size 5000 --output_dir splits/ --seed 42
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


def main(args):
    print(f"[INFO] Loading checkpoint: {args.ckpt_json}")
    data = load_json(args.ckpt_json)
    print(f"[INFO] Total samples: {len(data)}")

    rng = random.Random(args.seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)

    # ── Split ─────────────────────────────────────────────────────────────────
    if args.train1_size > len(data):
        raise ValueError(f"train1_size ({args.train1_size}) > dataset size ({len(data)})")

    train1_idx = indices[:args.train1_size]
    test_idx   = indices[args.train1_size:]
    train2_idx = rng.sample(train1_idx, args.train2_size)

    train1 = [data[i] for i in train1_idx]
    train2 = [data[i] for i in train2_idx]
    test   = [data[i] for i in test_idx]

    print(f"\n[INFO] Split sizes:")
    print(f"  train1 : {len(train1)}")
    print(f"  train2 : {len(train2)} (subset of train1)")
    print(f"  test   : {len(test)}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── TRACE format (JSON with category) ─────────────────────────────────────
    # trace.py needs: prompt, chosen, rejected, category
    # dpo_gold JSON already has all of these — save as-is
    trace_dir = os.path.join(args.output_dir, "trace")
    os.makedirs(trace_dir, exist_ok=True)
    print(f"\n[INFO] Saving TRACE datasets → {trace_dir}/")
    save_json(train1, os.path.join(trace_dir, "train1.json"))
    save_json(train2, os.path.join(trace_dir, "train2.json"))
    save_json(test,   os.path.join(trace_dir, "test.json"))

    # ── DPO format (JSON with only prompt/chosen/rejected) ────────────────────
    # dpo_accelerate.py only needs: prompt, chosen, rejected
    def to_dpo(samples):
        return [{"prompt": s["prompt"], "chosen": s["chosen"], "rejected": s["rejected"]}
                for s in samples]

    dpo_dir = os.path.join(args.output_dir, "dpo")
    os.makedirs(dpo_dir, exist_ok=True)
    print(f"\n[INFO] Saving DPO datasets → {dpo_dir}/")
    save_json(to_dpo(train1), os.path.join(dpo_dir, "train1.json"))
    save_json(to_dpo(train2), os.path.join(dpo_dir, "train2.json"))
    save_json(to_dpo(test),   os.path.join(dpo_dir, "test.json"))

    # ── Category distribution ─────────────────────────────────────────────────
    def cat_counts(samples):
        counts = {}
        for s in samples:
            c = s.get("category", "N/A")
            counts[c] = counts.get(c, 0) + 1
        return counts

    print(f"\n[INFO] Category distribution:")
    print(f"  train1 : {cat_counts(train1)}")
    print(f"  train2 : {cat_counts(train2)}")
    print(f"  test   : {cat_counts(test)}")

    print(f"\n[DONE] All splits saved to {args.output_dir}/")
    print(f"\nRun TRACE (train1):")
    print(f"  python trace.py --model_name_or_path <model> --dataset_json_path {trace_dir}/train1.json --output_dir runs/trace")
    print(f"\nRun DPO (train1):")
    print(f"  python dpo_accelerate.py --model_id <model> --dataset_source json --json_path {dpo_dir}/train1.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_json",    type=str, required=True,
                        help="Path to dpo_gold checkpoint JSON.")
    parser.add_argument("--output_dir",   type=str, default="splits")
    parser.add_argument("--train1_size",  type=int, default=10000)
    parser.add_argument("--train2_size",  type=int, default=5000)
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()
    main(args)

