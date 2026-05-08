"""
generate_general.py — Generate model responses for general capability benchmarks.

Supports: GPQA, MMLU, HellaSwag, GSM8K (loaded from HuggingFace datasets).

Saves: question/prompt, correct answer, choices (if applicable), model_response.

Output file: benchmark_general/{benchmark}_{model}_{samplesize}.json

Usage:
    python generate_general.py \
        --model_path path/to/merged_model \
        --benchmark mmlu \
        --sample_size 500 \
        --seed 42 \
        --output_dir benchmark_general

    # Run all benchmarks at once
    python generate_general.py \
        --model_path path/to/merged_model \
        --benchmark all \
        --sample_size 500
"""

import argparse
import json
import os
import random
from datetime import datetime

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

SUPPORTED_BENCHMARKS = ["gpqa", "mmlu", "hellaswag", "gsm8k"]


def load_model(model_path: str):
    print(f"[INFO] Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.bfloat16,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    return model, tokenizer


def prepare_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    return prompt


def generate_batch(model, tokenizer, prompts: list[str], max_new_tokens: int = 256) -> list[str]:
    texts = [prepare_prompt(tokenizer, p) for p in prompts]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(out[input_len:], skip_special_tokens=True)
        for out in outputs
    ]


# ---------------------------------------------------------------------------
# Benchmark loaders — each returns list of {"prompt": str, "correct_answer": str, ...}
# ---------------------------------------------------------------------------

def load_gpqa(sample_size, seed):
    """GPQA Diamond — graduate-level science questions."""
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    items = []
    for row in ds:
        choices = [row["Correct Answer"], row["Incorrect Answer 1"],
                   row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
        rng = random.Random(seed)
        rng.shuffle(choices)
        correct_idx = choices.index(row["Correct Answer"])
        labels = ["A", "B", "C", "D"]

        prompt = row["Question"] + "\n\n"
        for i, c in enumerate(choices):
            prompt += f"{labels[i]}. {c}\n"
        prompt += "\nAnswer with the letter of the correct choice."

        items.append({
            "prompt": prompt,
            "correct_answer": labels[correct_idx],
            "correct_text": row["Correct Answer"],
            "choices": {labels[i]: choices[i] for i in range(4)},
            "benchmark": "gpqa",
        })
    return _sample(items, sample_size, seed)


def load_mmlu(sample_size, seed):
    """MMLU — multitask language understanding."""
    ds = load_dataset("cais/mmlu", "all", split="test")
    items = []
    labels = ["A", "B", "C", "D"]
    for row in ds:
        choices = row["choices"]
        correct_idx = row["answer"]

        prompt = row["question"] + "\n\n"
        for i, c in enumerate(choices):
            prompt += f"{labels[i]}. {c}\n"
        prompt += "\nAnswer with the letter of the correct choice."

        items.append({
            "prompt": prompt,
            "correct_answer": labels[correct_idx],
            "correct_text": choices[correct_idx],
            "subject": row["subject"],
            "benchmark": "mmlu",
        })
    return _sample(items, sample_size, seed)


def load_hellaswag(sample_size, seed):
    """HellaSwag — commonsense reasoning / sentence completion."""
    ds = load_dataset("Rowan/hellaswag", split="validation")
    items = []
    labels = ["A", "B", "C", "D"]
    for row in ds:
        endings = row["endings"]
        correct_idx = int(row["label"])

        prompt = row["ctx"] + "\n\n"
        for i, e in enumerate(endings):
            prompt += f"{labels[i]}. {e}\n"
        prompt += "\nWhich ending best completes the text? Answer with the letter."

        items.append({
            "prompt": prompt,
            "correct_answer": labels[correct_idx],
            "correct_text": endings[correct_idx],
            "benchmark": "hellaswag",
        })
    return _sample(items, sample_size, seed)


def load_gsm8k(sample_size, seed):
    """GSM8K — grade school math."""
    ds = load_dataset("openai/gsm8k", "main", split="test")
    items = []
    for row in ds:
        # Extract numeric answer from "#### <number>" format
        answer_text = row["answer"]
        final_answer = answer_text.split("####")[-1].strip()

        prompt = row["question"] + "\n\nSolve step by step and give the final numeric answer."

        items.append({
            "prompt": prompt,
            "correct_answer": final_answer,
            "full_solution": answer_text,
            "benchmark": "gsm8k",
        })
    return _sample(items, sample_size, seed)


def _sample(items, sample_size, seed):
    if sample_size and sample_size < len(items):
        rng = random.Random(seed)
        items = rng.sample(items, sample_size)
    return items


LOADERS = {
    "gpqa": load_gpqa,
    "mmlu": load_mmlu,
    "hellaswag": load_hellaswag,
    "gsm8k": load_gsm8k,
}


def run_benchmark(model, tokenizer, benchmark_name, sample_size, seed,
                  max_new_tokens, batch_size, output_dir, model_path):
    print(f"\n{'='*60}")
    print(f"[INFO] Loading benchmark: {benchmark_name}")
    items = LOADERS[benchmark_name](sample_size, seed)
    actual_size = len(items)
    print(f"[INFO] {actual_size} samples loaded")

    # Generate responses in batches
    for i in tqdm(range(0, len(items), batch_size), desc=f"Generating ({benchmark_name})"):
        batch = items[i : i + batch_size]
        prompts = [item["prompt"] for item in batch]
        responses = generate_batch(model, tokenizer, prompts, max_new_tokens)
        for item, resp in zip(batch, responses):
            item["model_response"] = resp

    # Save
    os.makedirs(output_dir, exist_ok=True)
    model_name = os.path.basename(model_path.rstrip("/\\"))
    filename = f"{benchmark_name}_{model_name}_{actual_size}.json"
    output_path = os.path.join(output_dir, filename)

    output_data = {
        "metadata": {
            "model_path": model_path,
            "benchmark": benchmark_name,
            "sample_size": actual_size,
            "seed": seed,
            "max_new_tokens": max_new_tokens,
            "timestamp": datetime.now().isoformat(),
        },
        "results": items,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Saved {actual_size} responses to: {output_path}")
    return output_path


def main(args):
    model, tokenizer = load_model(args.model_path)

    benchmarks = SUPPORTED_BENCHMARKS if args.benchmark == "all" else [args.benchmark]

    for bm in benchmarks:
        run_benchmark(
            model, tokenizer, bm, args.sample_size, args.seed,
            args.max_new_tokens, args.batch_size, args.output_dir, args.model_path,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to merged model.")
    parser.add_argument("--benchmark", type=str, required=True,
                        choices=SUPPORTED_BENCHMARKS + ["all"],
                        help="Which benchmark to run (or 'all').")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Number of samples to evaluate (default: all).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for generation.")
    parser.add_argument("--output_dir", type=str, default="benchmark_general")
    args = parser.parse_args()
    main(args)
