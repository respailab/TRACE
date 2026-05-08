import torch
from copy import deepcopy
import json

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
from trl import DPOTrainer , DPOConfig

import argparse


def get_model_and_tokenizer(
    model_id: str = "Qwen/Qwen2.5-0.5b",
    cache_dir: str = "cache_dir"
):
    """
    Load model and tokenizer with proper configuration.
    
    Args:
        model_id: HuggingFace model ID
        cache_dir: Directory to cache the model
        
    Returns:
        model, tokenizer
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    
    # Configure tokenizer padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"  # DPO typically uses left padding
    
    # Load model with multi-GPU support
    # Using "auto" allows DPOTrainer to properly manage device placement across GPUs
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        # device_map="auto",  # Automatic device mapping for multi-GPU
        # torch_dtype=torch.bfloat16
    )
    
    # Enable gradient checkpointing to reduce memory usage
    model.gradient_checkpointing_enable()
    
    # Resize embeddings to exact tokenizer vocab size
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    
    return model, tokenizer


def apply_lora(
    model: AutoModelForCausalLM,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_layers: str = "both"
):
    """
    Apply LoRA to the model.
    
    Args:
        model: Base model
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        lora_layers: Which layers to apply LoRA to ('mlp', 'attn', or 'both')
        
    Returns:
        PEFT model with LoRA
    """
    # Determine target modules based on lora_layers flag
    if lora_layers == "mlp":
        target_modules = ["up_proj", "down_proj", "gate_proj"]
    elif lora_layers == "attn":
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    else:  # both
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    
    print(f"Applying LoRA to {lora_layers} layers: {target_modules}")
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none"
    )
    
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    
    return peft_model


def load_dpo_dataset_from_jsonl(
    jsonl_path: str = "hh_rlhf_train.jsonl",
    sample_size: int = 1000
):
    """
    Load DPO dataset from JSONL file.
    Each conversation will be formatted with chosen/rejected pairs.
    
    Args:
        jsonl_path: Path to JSONL file
        sample_size: Number of samples to use
        
    Returns:
        Dataset with prompt, chosen, rejected fields
    """
    print(f"Loading dataset from {jsonl_path}...")
    conversations = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            conversation = json.loads(line.strip())
            conversations.append(conversation)
    
    print(f"Loaded {len(conversations)} conversations")
    
    # Sample if needed
    if sample_size and sample_size < len(conversations):
        conversations = conversations[:sample_size]
        print(f"Sampled {sample_size} conversations")
    
    # For DPO, we need to create prompt, chosen, rejected format
    # Since we only have chosen responses in the JSONL, we'll extract the prompt
    # and use the full conversation as "chosen"
    # Note: For real DPO, you'd need actual rejected responses
    
    dpo_data = []
    for conv in conversations:
        if len(conv) < 2:
            continue
            
        # Extract prompt (all user messages up to last assistant response)
        prompt_messages = []
        chosen_messages = []
        
        for i, msg in enumerate(conv):
            if msg["role"] == "assistant":
                # This is the chosen response, everything before is the prompt
                prompt_messages = conv[:i]
                chosen_messages = conv[:i+1]  # Include this assistant response
                break
        
        if not prompt_messages or not chosen_messages:
            continue
            
        dpo_data.append({
            "prompt": prompt_messages,
            "chosen": chosen_messages,
            "rejected": chosen_messages,  # Placeholder - in real DPO you'd have actual rejected
        })
    
    print(f"Created {len(dpo_data)} DPO training examples")
    
    # Convert to Dataset
    dataset = Dataset.from_list(dpo_data)
    return dataset


def load_dpo_dataset_from_json(json_path: str):
    """Load DPO dataset from JSON file produced by prepare_datasets.py."""
    print(f"Loading dataset from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples")
    dataset = Dataset.from_list([
        {"prompt": s["prompt"], "chosen": s["chosen"], "rejected": s["rejected"]}
        for s in data
    ])
    return dataset


def load_dpo_dataset_from_hf(
    dataset_id: str = "Anthropic/hh-rlhf",
    split: str = "train",
    sample_size: int = 1000
):
    """
    Load DPO dataset from HuggingFace.
    
    Args:
        dataset_id: HuggingFace dataset ID
        split: Dataset split
        sample_size: Number of samples to use
        
    Returns:
        Dataset with prompt, chosen, rejected fields
    """
    from datasets import load_dataset
    
    print(f"Loading dataset from HuggingFace: {dataset_id}")
    dataset = load_dataset(dataset_id, split=split)
    
    if sample_size:
        dataset = dataset.select(range(min(sample_size, len(dataset))))
        print(f"Sampled {len(dataset)} examples")
    
    # The HH-RLHF dataset already has 'chosen' and 'rejected' fields
    # We need to parse them into conversation format
    
    def parse_conversation(text):
        """Parse HH format into conversation list."""
        conversation = []
        parts = text.split("\n\n")
        
        current_role = None
        current_content = ""
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
                
            if part.startswith("Human:"):
                if current_role and current_content:
                    conversation.append({"role": current_role, "content": current_content.strip()})
                current_role = "user"
                current_content = part[6:].strip()
            elif part.startswith("Assistant:"):
                if current_role and current_content:
                    conversation.append({"role": current_role, "content": current_content.strip()})
                current_role = "assistant"
                current_content = part[10:].strip()
            else:
                # Continuation of current message
                if current_content:
                    current_content += "\n" + part
                else:
                    current_content = part
        
        # Add the last message
        if current_role and current_content:
            conversation.append({"role": current_role, "content": current_content.strip()})
        
        return conversation
    
    def preprocess_for_dpo(example):
        """Convert to DPO format with prompt, chosen, rejected."""
        chosen_conv = parse_conversation(example["chosen"])
        rejected_conv = parse_conversation(example["rejected"])
        
        # Extract prompt (everything except the last assistant response)
        prompt = [msg for msg in chosen_conv if msg["role"] == "user"]
        
        return {
            "prompt": prompt,
            "chosen": chosen_conv,
            "rejected": rejected_conv
        }
    
    dataset = dataset.map(preprocess_for_dpo, remove_columns=dataset.column_names)
    print(f"Preprocessed {len(dataset)} examples for DPO")
    
    return dataset


def dpo_training(
    model_id: str = "Qwen/Qwen2.5-0.5b",
    dataset_source: str = "hf",  # "hf" or "jsonl"
    jsonl_path: str = "hh_rlhf_train.jsonl",
    sample_size: int = 1000,
    output_dir: str = "./outputs/dpo",
    epochs: int = 1,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 5e-5,
    beta: float = 0.1,
    use_lora: bool = True,
    lora_layers: str = "both",
    save_steps: int = 100,
    logging_steps: int = 10
):
    """
    Main DPO training function.
    
    Args:
        model_id: HuggingFace model ID
        dataset_source: "hf" for HuggingFace or "jsonl" for local file
        jsonl_path: Path to JSONL file (if using dataset_source="jsonl")
        sample_size: Number of samples to use
        output_dir: Output directory for checkpoints
        epochs: Number of training epochs
        batch_size: Per-device batch size
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Learning rate
        beta: DPO beta parameter (controls strength of KL penalty)
        use_lora: Whether to use LoRA
        save_steps: Save checkpoint every N steps
        logging_steps: Log every N steps
    """
    print("="*60)
    print("DPO Training Configuration")
    print("="*60)
    print(f"Model: {model_id}")
    print(f"Dataset source: {dataset_source}")
    print(f"Sample size: {sample_size}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Beta: {beta}")
    print(f"Use LoRA: {use_lora}")
    print("="*60)
    
    # Load model and tokenizer
    print("\n[1/5] Loading model and tokenizer...")
    model, tokenizer = get_model_and_tokenizer(model_id)
    print(f"✓ Model loaded: {model_id}")
    
    # Prepare PEFT config (don't apply LoRA yet - DPOTrainer will handle it)
    peft_config = None
    if use_lora:
        print("\n[2/5] Preparing LoRA configuration...")
        
        # Determine target modules based on lora_layers flag
        if lora_layers == "mlp":
            target_modules = ["up_proj", "down_proj", "gate_proj"]
        elif lora_layers == "attn":
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        else:  # both
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        
        print(f"Applying LoRA to {lora_layers} layers: {target_modules}")
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=target_modules,
            bias="none"
        )
        print("✓ LoRA config prepared (DPOTrainer will apply it)")
        print("✓ This approach uses the base model as reference and trains LoRA adapters")
    else:
        print("\n[2/5] Skipping LoRA (full fine-tuning)")
        print("⚠  Warning: Full fine-tuning with deepcopy reference may cause GPU memory issues")
    
    # Load dataset
    print("\n[3/5] Loading dataset...")
    if dataset_source == "json":
        dataset = load_dpo_dataset_from_json(jsonl_path)
    elif dataset_source == "jsonl":
        dataset = load_dpo_dataset_from_jsonl(jsonl_path, sample_size)
    else:
        dataset = load_dpo_dataset_from_hf(sample_size=sample_size)
    
    print(f"✓ Dataset loaded: {len(dataset)} examples")
    print("\nSample example:")
    sample = dataset[0]
    print(f"Prompt: {sample['prompt']}")
    print(f"Chosen: {sample['chosen']}")
    print(f"Rejected: {sample['rejected']}")
    
    # Training arguments
    print("\n[4/5] Setting up trainer...")
    training_args = DPOConfig(
        loss_type="robust",
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=2,
        bf16=False,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,  # Reduces memory usage
        # Multi-GPU fixes for tensor device errors
        ddp_find_unused_parameters=False,  # Critical: prevents DDP errors with PEFT
        dataloader_pin_memory=False,  # Prevents device mismatch in multi-GPU setups
    )
    
    # Initialize DPO trainer
    # When peft_config is provided, DPOTrainer uses the base model as reference
    # and only trains the LoRA adapters - this is memory efficient and multi-GPU friendly
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,  # Pass PEFT config instead of pre-applying LoRA
    )
    
    print("✓ Trainer initialized")
    
    # Train
    print("\n" + "="*60)
    print("Starting DPO Training...")
    print("="*60 + "\n")
    
    trainer.train()
    
    # Save final model
    print("\n" + "="*60)
    print("Saving final model...")
    #final_path = f"{output_dir}/{model_id.split('/')[-1]}_{sample_size}_bs{batch_size}_ep{epochs}_lr{learning_rate}_beta{beta}"
    ft_type = "lora" if args.use_lora else "full"
    final_path = f"{args.output_dir}/{args.model_id.split('/')[-1]}_dpo_{ft_type}_{args.lora_layers}_{args.sample_size}_bs{args.batch_size}_ep{args.epochs}_lr{args.learning_rate}"
    
    trainer.save_model(final_path)
    print(f"✓ Model saved to: {final_path}")
    print("="*60)
    print("Training complete!")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPO Training with Anthropic HH-RLHF")
    
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen2.5-0.5b",
        help="HuggingFace model ID"
    )
    
    parser.add_argument(
        "--dataset_source",
        type=str,
        default="hf",
        choices=["hf", "jsonl", "json"],
        help="Dataset source: 'hf' for HuggingFace, 'jsonl' for local JSONL, 'json' for DPO-Gold JSON"
    )
    
    parser.add_argument(
        "--jsonl_path",
        type=str,
        default="hh_rlhf_train.jsonl",
        help="Path to JSONL file (only used if dataset_source='jsonl')"
    )
    
    parser.add_argument(
        "--sample_size",
        type=int,
        default=1000,
        help="Number of samples to use from dataset"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/dpo",
        help="Output directory for checkpoints"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Per-device batch size"
    )
    
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO beta parameter (KL penalty coefficient)"
    )
    
    parser.add_argument(
        "--use_lora",
        action="store_true",
        default=True,
        help="Use LoRA for parameter-efficient fine-tuning"
    )
    
    parser.add_argument(
        "--no_lora",
        action="store_true",
        help="Disable LoRA (full fine-tuning)"
    )
    
    parser.add_argument(
        "--lora_layers",
        type=str,
        choices=["mlp", "attn", "both"],
        default="both",
        help="Which layers to apply LoRA to: 'mlp' (MLP layers only), 'attn' (attention layers only), or 'both' (all layers)"
    )
    
    args = parser.parse_args()
    
    # Handle --no_lora flag
    if args.no_lora:
        args.use_lora = False
    
    # Run training
    dpo_training(
        model_id=args.model_id,
        dataset_source=args.dataset_source,
        jsonl_path=args.jsonl_path,
        sample_size=args.sample_size,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        beta=args.beta,
        use_lora=args.use_lora,
        lora_layers=args.lora_layers
    )


