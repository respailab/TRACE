"""
generate_tpa_vllm.py — Generate model responses for TPA evaluation using vLLM.

Drop-in replacement for generate_tpa.py with vLLM backend.
Same input/output format — run each model on a specific GPU via CUDA_VISIBLE_DEVICES.

Usage (two models on two separate GPUs):
    CUDA_VISIBLE_DEVICES=0 python generate_tpa_vllm.py \
        --model_path .runs/trace-10k/Llama-3.1-8B-TRACE-10k \
        --test_json  pku_test_2k.json \
        --output_dir benchmark_TPA

    CUDA_VISIBLE_DEVICES=1 python generate_tpa_vllm.py \
        --model_path .runs/dpo-10k/Llama-3.1-8B-DPO-Gold-10k \
        --test_json  pku_test_2k.json \
        --output_dir benchmark_TPA
"""

import argparse
import json
import os
import random
from datetime import datetime

from tqdm import tqdm
from vllm import LLM, SamplingParams


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


def build_output_path(output_dir, test_json, model_path, sample_size):
    dataset_name = os.path.splitext(os.path.basename(test_json))[0]
    model_name = os.path.basename(model_path.rstrip("/\\"))
    filename = f"{dataset_name}_{model_name}_{sample_size}.json"
    return os.path.join(output_dir, filename)


def main(args):
    # Load test data
    with open(args.test_json) as f:
        test_data = json.load(f)
    print(f"[INFO] Loaded {len(test_data)} samples from {args.test_json}")

    if args.sample_size and args.sample_size < len(test_data):
        rng = random.Random(args.seed)
        test_data = rng.sample(test_data, args.sample_size)
        print(f"[INFO] Sampled {args.sample_size} with seed={args.seed}")
    sample_size = len(test_data)

    bin_counts = {}
    for item in test_data:
        cat = item["category"]
        bin_counts[cat] = bin_counts.get(cat, 0) + 1
    print(f"[INFO] Bin counts: {bin_counts}")

    # Load vLLM model — uses whatever GPUs are visible via CUDA_VISIBLE_DEVICES
    print(f"[INFO] Loading model with vLLM: {args.model_path}")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
    )

    # Build prompts using chat template if available
    tokenizer = llm.get_tokenizer()
    prompts_text = []
    for item in test_data:
        prompt = extract_prompt(item["prompt"])
        messages = [{"role": "user", "content": prompt}]
        if tokenizer.chat_template:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt
        prompts_text.append(text)

    # Generate all at once — vLLM handles batching internally
    print(f"[INFO] Generating {sample_size} responses...")
    outputs = llm.generate(prompts_text, sampling_params)

    results = []
    for item, output in tqdm(zip(test_data, outputs), total=sample_size, desc="Collecting"):
        results.append({
            "prompt":         extract_prompt(item["prompt"]),
            "chosen":         extract_text(item["chosen"]),
            "rejected":       extract_text(item["rejected"]),
            "category":       item["category"],
            "model_response": output.outputs[0].text,
        })

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = build_output_path(args.output_dir, args.test_json, args.model_path, sample_size)

    output_data = {
        "metadata": {
            "model_path":     args.model_path,
            "test_json":      args.test_json,
            "sample_size":    sample_size,
            "seed":           args.seed,
            "max_new_tokens": args.max_new_tokens,
            "bin_counts":     bin_counts,
            "timestamp":      datetime.now().isoformat(),
        },
        "results": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Saved {len(results)} responses to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",   type=str, required=True)
    parser.add_argument("--test_json",    type=str, required=True)
    parser.add_argument("--sample_size",  type=int, default=None)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--output_dir",   type=str, default="benchmark_TPA")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism (1 = single GPU).")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="Fraction of GPU memory vLLM can use.")
    parser.add_argument("--max_model_len", type=int, default=4096,
                        help="Max sequence length (reduce if OOM).")
    args = parser.parse_args()
    main(args)

