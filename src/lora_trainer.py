"""LoRA fine-tuning pipeline.

Uses HuggingFace transformers + PEFT when available.
Falls back to a mock trainer if heavy deps are missing.
"""
import os
import json
import random
from typing import Dict, Any


def _try_import_training_stack():
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
        return {
            "torch": torch,
            "AutoModelForCausalLM": AutoModelForCausalLM,
            "AutoTokenizer": AutoTokenizer,
            "TrainingArguments": TrainingArguments,
            "Trainer": Trainer,
            "DataCollatorForLanguageModeling": DataCollatorForLanguageModeling,
            "LoraConfig": LoraConfig,
            "get_peft_model": get_peft_model,
            "TaskType": TaskType,
            "Dataset": Dataset,
        }
    except Exception as e:
        print(f"[WARN] Training stack unavailable ({e.__class__.__name__}). Using mock trainer.")
        return None


def _format_prompt(sample: Dict[str, str]) -> str:
    return (
        f"### Instruction:\n{sample['instruction']}\n\n"
        f"### Response:\n{sample['output']}"
    )


def _mock_train(dataset_path: str, output_dir: str = "adapters") -> Dict[str, Any]:
    """Mock training - simulates LoRA training and saves metadata."""
    os.makedirs(output_dir, exist_ok=True)

    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    # Simulate training
    metrics = {
        "base_model": os.getenv("BASE_MODEL", "tinyllama/tinyllama-1.1B-Chat-v1.0"),
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"],
        "train_samples": len(data),
        "epochs": 3,
        "final_loss": round(random.uniform(0.3, 0.6), 4),
        "mode": "mock",
    }

    with open(f"{output_dir}/training_metadata.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[OK] Mock LoRA training complete. Metrics:")
    for k, v in metrics.items():
        print(f"    {k}: {v}")

    return metrics


def train_lora(dataset_path: str = "data/qa_dataset.json",
               output_dir: str = "adapters",
               base_model: str = None,
               epochs: int = 3,
               batch_size: int = 4,
               learning_rate: float = 2e-4,
               lora_r: int = 8,
               lora_alpha: int = 16) -> Dict[str, Any]:
    """Run LoRA fine-tuning (real or mock)."""
    base_model = base_model or os.getenv("BASE_MODEL", "tinyllama/tinyllama-1.1B-Chat-v1.0")

    stack = _try_import_training_stack()
    if stack is None:
        return _mock_train(dataset_path, output_dir)

    torch = stack["torch"]
    print(f"[OK] Training stack loaded. Base model: {base_model}")

    with open(dataset_path, encoding="utf-8") as f:
        raw = json.load(f)

    tokenizer = stack["AutoTokenizer"].from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = stack["AutoModelForCausalLM"].from_pretrained(
        base_model, trust_remote_code=True, device_map="auto"
    )

    # LoRA config
    lora_config = stack["LoraConfig"](
        task_type=stack["TaskType"].CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    model = stack["get_peft_model"](model, lora_config)
    model.print_trainable_parameters()

    # Prepare dataset
    texts = [_format_prompt(s) for s in raw]
    max_len = int(os.getenv("MAX_LENGTH", "256"))

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_len,
        )

    ds = stack["Dataset"].from_dict({"text": texts})
    ds = ds.map(tokenize, batched=True)
    ds = ds.remove_columns(["text"])

    collator = stack["DataCollatorForLanguageModeling"](tokenizer, mlm=False)

    args = stack["TrainingArguments"](
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        fp16=torch.cuda.is_available(),
    )

    trainer = stack["Trainer"](
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collator,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = {
        "base_model": base_model,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "train_samples": len(raw),
        "epochs": epochs,
        "mode": "real",
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/training_metadata.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[OK] Real LoRA training complete. Adapter saved to {output_dir}")
    return metrics


if __name__ == "__main__":
    train_lora()
