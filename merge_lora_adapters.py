#!/usr/bin/env python3
"""
Simple script to load a LoRA model, merge the adapter, and save the merged model.

Usage:
    python merge_lora.py --model_path ./outputs/dpo/Qwen2.5-7B-Instruct_dpo_lora_10000_bs1_ep1_lr5e-05
"""

import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_and_save_lora(base_model_path: str, lora_path: str, output_path: str = None):
    """
    Load a base model, apply LoRA adapter, merge, and save the merged model.
    
    Args:
        base_model_path: Path to the base model (e.g., "Qwen/Qwen2.5-7B-Instruct")
        lora_path: Path to the LoRA adapter checkpoint
        output_path: Path to save the merged model (default: lora_path + "_merged")
    """
    # Set default output path
    if output_path is None:
        output_path = f"{lora_path}_merged"
    
    print("="*70)
    print("LoRA Merge and Save Script")
    print("="*70)
    print(f"\n📂 Base model: {base_model_path}")
    print(f"� LoRA adapter: {lora_path}")
    print(f"�💾 Output merged model: {output_path}")
    print()
    
    # Check if LoRA path exists
    if not os.path.exists(lora_path):
        raise ValueError(f"LoRA path does not exist: {lora_path}")
    
    # 1. Load tokenizer from the LoRA checkpoint (MUST MATCH vocab size)
    print("[1/5] Loading tokenizer from LoRA checkpoint...")
    tokenizer = AutoTokenizer.from_pretrained(lora_path)
    print(f"✓ Tokenizer loaded (vocab size: {len(tokenizer)})")
    
    # 2. Load base model normally (NO tokenizer argument)
    print(f"\n[2/5] Loading base model from {base_model_path}...")
    print("⚠️  This may take a few minutes for large models...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",  # Automatically distribute across GPUs if available
        torch_dtype="auto",
        trust_remote_code=True,

    )
    print("✓ Base model loaded")
    
    # 3. Resize base model vocab to match LoRA vocab size (critical)
    print(f"\n[3/5] Resizing base model embeddings to match tokenizer...")
    original_vocab_size = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tokenizer))
    new_vocab_size = model.get_input_embeddings().weight.shape[0]
    print(f"✓ Resized from {original_vocab_size} to {new_vocab_size} tokens")
    
    # 4. Load LoRA adapter
    print(f"\n[4/5] Loading LoRA adapter from {lora_path}...")
    model = PeftModel.from_pretrained(model, lora_path)
    print("✓ LoRA adapter loaded")
    
    # 5. Merge LoRA adapter with base model
    print("\n[5/5] Merging LoRA adapter with base model...")
    print("⚠️  This will create a full model (not LoRA anymore)")
    merged_model = model.merge_and_unload()
    print("✓ LoRA adapter merged successfully")
    
    # Save the merged model
    print(f"\n[6/6] Saving merged model to: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("✓ Merged model and tokenizer saved")
    
    # Print summary
    print("\n" + "="*70)
    print("✅ MERGE COMPLETE!")
    print("="*70)
    print(f"\n📊 Summary:")
    print(f"   Base Model:    {base_model_path}")
    print(f"   LoRA Adapter:  {lora_path}")
    print(f"   Merged Model:  {output_path}")
    print(f"   Vocab Size:    {len(tokenizer)}")
    print(f"\n💡 You can now load this merged model like any regular model:")
    print(f"   from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"   model = AutoModelForCausalLM.from_pretrained('{output_path}')")
    print(f"   tokenizer = AutoTokenizer.from_pretrained('{output_path}')")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter with base model and save"
    )
    
    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        help="Path to the base model (e.g., 'Qwen/Qwen2.5-7B-Instruct' or local path)"
    )
    
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to the LoRA adapter checkpoint directory"
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the merged model (default: lora_path + '_merged')"
    )
    
    args = parser.parse_args()
    
    merge_and_save_lora(args.base_model_path, args.lora_path, args.output_path)

