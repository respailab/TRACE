"""
create_noisy_splits.py — Create noisy versions of a triaged DPO dataset.

Two types of noise are applied randomly to selected samples:
  1. Category flip  — category reassigned to a different random category
  2. Chosen/rejected swap — chosen and rejected responses are swapped

Usage:
    python create_noisy_splits.py \
        --input_json  pku_dpo_gold_20k_fresh.json \
        --output_dir  noisy_splits \
        --noise_rates 0.1 0.2 \
        --seed        42
"""

import argparse
import json
import os
import random


CATEGORIES = ["INVERT", "PUNISH", "RETAIN"]


def inject_noise(data: list, noise_rate: float, rng: random.Random) -> list:
    n_noisy = round(len(data) * noise_rate)
    noisy_indices = set(rng.sample(range(len(data)), n_noisy))

    result = []
    noise_log = {"category_flip": 0, "swap": 0, "clean": 0}

    for i, sample in enumerate(data):
        s = dict(sample)
        if i in noisy_indices:
            # 50/50: category flip or chosen/rejected swap
            if rng.random() < 0.5:
                other_cats = [c for c in CATEGORIES if c != s["category"]]
                s["category"] = rng.choice(other_cats)
                noise_log["category_flip"] += 1
            else:
                s["chosen"], s["rejected"] = s["rejected"], s["chosen"]
                noise_log["swap"] += 1
        else:
            noise_log["clean"] += 1
        result.append(s)

    print(f"  Noise {noise_rate*100:.0f}%: {n_noisy} samples corrupted "
          f"(category_flip={noise_log['category_flip']}, swap={noise_log['swap']})")

    # Distribution after noise
    counts = {}
    for s in result:
        counts[s["category"]] = counts.get(s["category"], 0) + 1
    print(f"  Category distribution after noise: {counts}")

    return result


def main(args):
    with open(args.input_json) as f:
        data = json.load(f)
    print(f"[INFO] Loaded {len(data)} samples from {args.input_json}")

    counts = {}
    for s in data:
        counts[s["category"]] = counts.get(s["category"], 0) + 1
    print(f"[INFO] Original distribution: {counts}")

    os.makedirs(args.output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(args.input_json))[0]

    for rate in args.noise_rates:
        rng = random.Random(args.seed)
        noisy = inject_noise(data, rate, rng)

        pct = int(rate * 100)
        out_path = os.path.join(args.output_dir, f"{base}_noisy{pct}pct.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(noisy, f, indent=2, ensure_ascii=False)
        print(f"  Saved -> {out_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json",  type=str, required=True)
    parser.add_argument("--output_dir",  type=str, default="noisy_splits")
    parser.add_argument("--noise_rates", type=float, nargs="+", default=[0.1, 0.2])
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()
    main(args)
