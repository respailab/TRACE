"""
create_pku_preference_dataset.py — Build chosen/rejected dataset from raw PKU-SafeRLHF.

Assignment logic (priority order):
  1. One safe, one unsafe      → safe=chosen,   unsafe=rejected          [USED]
  2. Both safe                 → better_response_id breaks tie            [USED]
  3. Both unsafe               → skipped (no good chosen available)       [UNUSED]
  4. better_response_id == -1  → no preference label, skipped            [UNUSED]

Prints a summary table at the end.

Usage:
    python create_pku_preference_dataset.py --output_json pku_chosen_rejected.json
    python create_pku_preference_dataset.py --output_json pku_chosen_rejected.json --num_samples 100
"""

import argparse
import json
from datasets import load_dataset


def main(args):
    print("[INFO] Loading PKU-SafeRLHF ...")
    dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split=args.split, cache_dir="cache_dir")
    if args.num_samples:
        dataset = dataset.select(range(args.num_samples))
    print(f"[INFO] Total samples loaded: {len(dataset)}")

    used = []
    unused = []

    counts = {
        "one_safe_one_unsafe": 0,
        "both_safe":           0,
        "both_unsafe":         0,
        "no_preference_label": 0,
    }

    for item in dataset:
        prompt     = item["prompt"]
        resp0      = item["response_0"]
        resp1      = item["response_1"]
        safe0      = item["is_response_0_safe"]
        safe1      = item["is_response_1_safe"]
        safer_id   = item["safer_response_id"]   # primary: safety-focused
        better_id  = item["better_response_id"]  # fallback: quality-focused

        # ── Case: no safer_response label ────────────────────────────────────
        if safer_id not in (0, 1):
            counts["no_preference_label"] += 1
            unused.append({"prompt": prompt, "response_0": resp0, "response_1": resp1,
                            "reason": "no_preference_label"})
            continue

        # ── Case 1: one safe, one unsafe ─────────────────────────────────────
        if safe0 and not safe1:
            chosen, rejected = resp0, resp1
            counts["one_safe_one_unsafe"] += 1
            used.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

        elif safe1 and not safe0:
            chosen, rejected = resp1, resp0
            counts["one_safe_one_unsafe"] += 1
            used.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

        # ── Case 2: both safe — use safer_response_id (fallback: better_response_id)
        elif safe0 and safe1:
            rank_id  = safer_id if safer_id in (0, 1) else better_id
            chosen   = resp0 if rank_id == 0 else resp1
            rejected = resp1 if rank_id == 0 else resp0
            counts["both_safe"] += 1
            used.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

        # ── Case 3: both unsafe — use safer_response_id (less harmful as chosen)
        else:
            chosen   = resp0 if safer_id == 0 else resp1
            rejected = resp1 if safer_id == 0 else resp0
            counts["both_unsafe"] += 1
            used.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    # ── Save used dataset ─────────────────────────────────────────────────────
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(used, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved {len(used)} usable pairs to {args.output_json}")

    if args.save_unused:
        unused_path = args.output_json.replace(".json", "_unused.json")
        with open(unused_path, "w", encoding="utf-8") as f:
            json.dump(unused, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Saved {len(unused)} unused pairs to {unused_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    total = len(dataset)
    print("\n" + "=" * 52)
    print(f"{'Category':<30} {'Count':>8} {'%':>8}")
    print("-" * 52)
    print(f"{'One safe, one unsafe (USED)':<30} {counts['one_safe_one_unsafe']:>8} {counts['one_safe_one_unsafe']/total*100:>7.1f}%")
    print(f"{'Both safe (USED)':<30} {counts['both_safe']:>8} {counts['both_safe']/total*100:>7.1f}%")
    print(f"{'Both unsafe, safer_id used (USED)':<30} {counts['both_unsafe']:>8} {counts['both_unsafe']/total*100:>7.1f}%")
    print(f"{'No safer_response_id (UNUSED)':<30} {counts['no_preference_label']:>8} {counts['no_preference_label']/total*100:>7.1f}%")
    print("-" * 52)
    print(f"{'TOTAL USED':<30} {len(used):>8} {len(used)/total*100:>7.1f}%")
    print(f"{'TOTAL UNUSED':<30} {len(unused):>8} {len(unused)/total*100:>7.1f}%")
    print(f"{'TOTAL':<30} {total:>8} {'100.0%':>8}")
    print("=" * 52)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_json",  type=str, default="pku_chosen_rejected.json")
    parser.add_argument("--split",        type=str, default="train")
    parser.add_argument("--num_samples",  type=int, default=None,
                        help="Subset size for testing (default: full dataset)")
    parser.add_argument("--save_unused",  action="store_true",
                        help="Also save skipped pairs to <output>_unused.json")
    args = parser.parse_args()
    main(args)

