"""
oracle_evaluate_general.py — Evaluate saved general benchmark responses using an API oracle.

Loads output from generate_general.py, calls oracle API for each response,
computes accuracy.

Saves results alongside the input file with _oracle suffix.

Usage:
    python oracle_evaluate_general.py \
        --responses_json benchmark_general/mmlu_trace10k_500.json \
        --oracle_model gpt-4o-mini

    # Resume from where it left off (safe to re-run)
    python oracle_evaluate_general.py \
        --responses_json benchmark_general/mmlu_trace10k_500.json \
        --resume
"""

import argparse
import json
import os
from datetime import datetime

from openai import OpenAI
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Oracle prompts — one for multiple-choice, one for numeric (GSM8K)
# ---------------------------------------------------------------------------

MC_ORACLE_PROMPT = """The correct answer is: {correct_answer}

Question: {prompt}

Model response: {response}

Does the model's response select or clearly indicate the correct answer '{correct_answer}'?
Answer with only 'Yes' or 'No'."""

NUMERIC_ORACLE_PROMPT = """The correct numeric answer is: {correct_answer}

Question: {prompt}

Model response: {response}

Does the model's response arrive at the correct numeric answer {correct_answer}?
Answer with only 'Yes' or 'No'."""


def judge_correct(client, model: str, item: dict) -> bool:
    """Returns True if the model response is correct according to the oracle."""
    benchmark = item.get("benchmark", "")

    if benchmark == "gsm8k":
        oracle_input = NUMERIC_ORACLE_PROMPT.format(
            correct_answer=item["correct_answer"],
            prompt=item["prompt"],
            response=item["model_response"],
        )
    else:
        oracle_input = MC_ORACLE_PROMPT.format(
            correct_answer=item["correct_answer"],
            prompt=item["prompt"],
            response=item["model_response"],
        )

    result = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": oracle_input},
        ],
        max_tokens=5,
    )
    answer = result.choices[0].message.content.strip().lower()

    if answer.startswith("yes"):
        return True
    if answer.startswith("no"):
        return False
    print(f"  [Oracle] Ambiguous: '{answer}' — treating as incorrect")
    return False


def main(args):
    # Load responses
    with open(args.responses_json, "r") as f:
        data = json.load(f)

    metadata = data["metadata"]
    results = data["results"]
    benchmark = metadata["benchmark"]
    print(f"[INFO] Loaded {len(results)} responses from {args.responses_json}")
    print(f"[INFO] Model: {metadata['model_path']}")
    print(f"[INFO] Benchmark: {benchmark}")

    # Check for existing partial results (resume support)
    output_path = args.responses_json.replace(".json", "_oracle.json")
    already_judged = {}
    if args.resume and os.path.exists(output_path):
        with open(output_path, "r") as f:
            existing = json.load(f)
        for r in existing.get("results", []):
            if "correct" in r:
                key = r["prompt"][:200]  # use truncated prompt as key
                already_judged[key] = r["correct"]
        print(f"[INFO] Resuming: {len(already_judged)} already judged")

    # Oracle API
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Judge each response
    for item in tqdm(results, desc="Oracle judging"):
        key = item["prompt"][:200]
        if key in already_judged:
            item["correct"] = already_judged[key]
            continue

        item["correct"] = judge_correct(client, args.oracle_model, item)

    # Compute metrics
    total = len(results)
    correct_count = sum(1 for r in results if r["correct"])
    accuracy = correct_count / max(total, 1) * 100

    # Per-subject breakdown for MMLU
    subject_metrics = {}
    if benchmark == "mmlu":
        subject_totals = {}
        subject_correct = {}
        for r in results:
            subj = r.get("subject", "unknown")
            subject_totals[subj] = subject_totals.get(subj, 0) + 1
            if r["correct"]:
                subject_correct[subj] = subject_correct.get(subj, 0) + 1
        for subj in sorted(subject_totals):
            sc = subject_correct.get(subj, 0)
            st = subject_totals[subj]
            subject_metrics[subj] = {
                "correct": sc, "total": st,
                "accuracy_pct": round(sc / st * 100, 2),
            }

    metrics = {
        "accuracy_pct": round(accuracy, 2),
        "correct_count": correct_count,
        "total_count": total,
        "oracle_model": args.oracle_model,
        "timestamp": datetime.now().isoformat(),
    }
    if subject_metrics:
        metrics["subject_breakdown"] = subject_metrics

    # Save
    output_data = {
        "metadata": {**metadata, **metrics},
        "results": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 60)
    print(f"Model: {metadata['model_path']}")
    print(f"Benchmark: {benchmark}")
    print(f"Oracle: {args.oracle_model}")
    print("=" * 60)
    print(f"  Accuracy: {accuracy:.1f}%  ({correct_count}/{total})")
    if subject_metrics:
        print(f"  Subjects: {len(subject_metrics)}")
        # Show top/bottom 3
        sorted_subj = sorted(subject_metrics.items(), key=lambda x: x[1]["accuracy_pct"])
        if len(sorted_subj) > 6:
            print("  Bottom 3:")
            for s, m in sorted_subj[:3]:
                print(f"    {s}: {m['accuracy_pct']:.1f}%")
            print("  Top 3:")
            for s, m in sorted_subj[-3:]:
                print(f"    {s}: {m['accuracy_pct']:.1f}%")
    print("=" * 60)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses_json", type=str, required=True,
                        help="Output from generate_general.py.")
    parser.add_argument("--oracle_model", type=str, default="gpt-4o-mini",
                        help="OpenAI model for correctness judging.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from partial results if output file exists.")
    args = parser.parse_args()
    main(args)
