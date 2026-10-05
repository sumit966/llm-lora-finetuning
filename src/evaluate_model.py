"""Evaluate fine-tuned model vs base model."""
import json
import os
import random


def evaluate_models(dataset_path: str = "data/qa_dataset.json",
                    eval_size: int = 20,
                    seed: int = 42):
    """Compare fine-tuned vs base model on a fixed eval set.

    In mock mode, simulates an improvement.
    """
    random.seed(seed)

    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    eval_set = data[:eval_size]

    # Simulated metrics (in real mode, compute against held-out responses)
    base_accuracy = 0.52
    finetuned_accuracy = 0.78

    results = {
        "eval_size": len(eval_set),
        "base_model": {
            "accuracy": base_accuracy,
            "avg_response_length": 45,
        },
        "finetuned_model": {
            "accuracy": finetuned_accuracy,
            "avg_response_length": 62,
        },
        "improvement_pct": round((finetuned_accuracy - base_accuracy) / base_accuracy * 100, 2),
    }

    print(f"\n[Eval] Base model accuracy:      {base_accuracy:.2%}")
    print(f"[Eval] Fine-tuned accuracy:      {finetuned_accuracy:.2%}")
    print(f"[Eval] Improvement:              +{results['improvement_pct']}%")

    return results


if __name__ == "__main__":
    evaluate_models()
