"""
generate_tpa.py — Generate model responses for Target Policy Agreement evaluation.

Saves: prompt, chosen, rejected, category, model_response for every sample.

Output file: benchmark_TPA/{dataset}_{model}_{samplesize}.json

Usage:
    python generate_tpa.py \
        --model_path path/to/merged_model \
        --test_json splits/trace/test.json \
        --sample_size 1000 \
        --seed 42 \
        --output_dir benchmark_TPA
"""

import argparse
import json
import os
import random
from datetime import datetime

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path: str):
    print(f"[INFO] Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", cache_dir="cache_dir",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    return model, tokenizer


def extract_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        for turn in reversed(value):
            if isinstance(turn, dict) and turn.get("role") == "assistant":
                return turn["content"]
        return " ".join(t.get("content", "") for t in value if isinstance(t, dict))
    return str(value)


def extract_prompt(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        for turn in value:
            if isinstance(turn, dict) and turn.get("role") == "user":
                return turn["content"]
        return " ".join(t.get("content", "") for t in value if isinstance(t, dict))
    return str(value)


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
    # Decode only the generated part (after input)
    input_len = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(out[input_len:], skip_special_tokens=True)
        for out in outputs
    ]


def build_output_path(output_dir, test_json, model_path, sample_size):
    dataset_name = os.path.splitext(os.path.basename(test_json))[0]
    model_name = os.path.basename(model_path.rstrip("/\\"))
    filename = f"{dataset_name}_{model_name}_{sample_size}.json"
    return os.path.join(output_dir, filename)


def main(args):
    # Load test data
    with open(args.test_json, "r") as f:
        test_data = json.load(f)
    print(f"[INFO] Loaded {len(test_data)} samples from {args.test_json}")

    # Sample if requested
    if args.sample_size and args.sample_size < len(test_data):
        rng = random.Random(args.seed)
        test_data = rng.sample(test_data, args.sample_size)
        print(f"[INFO] Sampled {args.sample_size} with seed={args.seed}")
    sample_size = len(test_data)

    # Bin counts
    bin_counts = {}
    for item in test_data:
        cat = item["category"]
        bin_counts[cat] = bin_counts.get(cat, 0) + 1
    print(f"[INFO] Bin counts: {bin_counts}")

    # Load model
    model, tokenizer = load_model(args.model_path)

    # Generate responses in batches
    results = []
    for i in tqdm(range(0, len(test_data), args.batch_size), desc="Generating responses"):
        batch = test_data[i : i + args.batch_size]
        prompts = [extract_prompt(item["prompt"]) for item in batch]
        responses = generate_batch(model, tokenizer, prompts, args.max_new_tokens)

        for item, prompt_text, response in zip(batch, prompts, responses):
            results.append({
                "prompt": prompt_text,
                "chosen": extract_text(item["chosen"]),
                "rejected": extract_text(item["rejected"]),
                "category": item["category"],
                "model_response": response,
            })

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = build_output_path(args.output_dir, args.test_json, args.model_path, sample_size)

    output_data = {
        "metadata": {
            "model_path": args.model_path,
            "test_json": args.test_json,
            "sample_size": sample_size,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "bin_counts": bin_counts,
            "timestamp": datetime.now().isoformat(),
        },
        "results": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Saved {len(results)} responses to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to merged model.")
    parser.add_argument("--test_json", type=str, required=True,
                        help="Path to test.json with prompt/chosen/rejected/category.")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Number of samples to evaluate (default: all).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for generation.")
    parser.add_argument("--output_dir", type=str, default="benchmark_TPA")
    args = parser.parse_args()
    main(args)
