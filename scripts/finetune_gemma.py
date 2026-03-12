"""
Gemma Fine-Tuning Script (run on GCP)
======================================
Fine-tunes Gemma 3 270M on function-calling data using LoRA.
Designed to run on a single T4/L4 GPU.

Usage:
    python finetune_gemma.py --data data/gemma_270m_finetune.jsonl
"""

import json
import argparse
from pathlib import Path

# NOTE: These imports require the optional dependencies.
# Install with: pip install transformers peft datasets accelerate bitsandbytes
try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    print("Fine-tuning dependencies not installed.")
    print("Install with: pip install transformers peft datasets accelerate bitsandbytes")


# ─── Config ─────────────────────────────────────────────────────────

BASE_MODEL = "google/gemma-3-1b-it"  # Smallest Gemma 3 for prototyping
# For production: "google/gemma-3-270m" when released

LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "task_type": "CAUSAL_LM",
}

TRAINING_CONFIG = {
    "output_dir": "models/gemma-270m-finetuned",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "warmup_steps": 50,
    "logging_steps": 10,
    "save_steps": 100,
    "eval_steps": 50,
    "eval_strategy": "steps",
    "fp16": True,
    "optim": "adamw_torch",
    "report_to": "none",
}


def load_data(data_path: str) -> Dataset:
    """Load JSONL fine-tuning data."""
    examples = []
    with open(data_path) as f:
        for line in f:
            item = json.loads(line)
            # Format as chat template
            messages = item["messages"]
            text = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    text += f"<start_of_turn>system\n{content}<end_of_turn>\n"
                elif role == "user":
                    text += f"<start_of_turn>user\n{content}<end_of_turn>\n"
                elif role == "assistant":
                    text += f"<start_of_turn>model\n{content}<end_of_turn>\n"
            examples.append({"text": text})

    return Dataset.from_list(examples)


def finetune(data_path: str, output_dir: str = None):
    """Run fine-tuning."""
    if not HAS_DEPS:
        print("Cannot run fine-tuning without dependencies. Exiting.")
        return

    if output_dir:
        TRAINING_CONFIG["output_dir"] = output_dir

    print(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Apply LoRA
    lora_config = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and tokenize data
    dataset = load_data(data_path)
    print(f"Loaded {len(dataset)} training examples")

    # Split
    split = dataset.train_test_split(test_size=0.1, seed=42)

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
        )

    train_ds = split["train"].map(tokenize, batched=True, remove_columns=["text"])
    eval_ds = split["test"].map(tokenize, batched=True, remove_columns=["text"])

    # Train
    training_args = TrainingArguments(**TRAINING_CONFIG)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("Starting fine-tuning...")
    trainer.train()

    # Save
    model.save_pretrained(TRAINING_CONFIG["output_dir"])
    tokenizer.save_pretrained(TRAINING_CONFIG["output_dir"])
    print(f"Model saved to {TRAINING_CONFIG['output_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Gemma for ElderWatch")
    parser.add_argument("--data", default="data/gemma_270m_finetune.jsonl")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    finetune(args.data, args.output_dir)
