"""Generate synthetic domain Q&A dataset for fine-tuning."""
import json
import os
import random


DOMAIN_TOPICS = [
    ("What is retrieval-augmented generation?",
     "RAG is a technique combining a retrieval system (like a vector database) with a generative LLM to produce grounded answers with source attribution."),
    ("How does LoRA reduce training cost?",
     "LoRA freezes the base model and injects small trainable low-rank matrices into each transformer layer, reducing trainable parameters by 99% or more."),
    ("What is a vector embedding?",
     "A dense numeric representation of text where semantically similar items have similar vectors, enabling fast similarity search."),
    ("What is the difference between RAG and fine-tuning?",
     "RAG injects new knowledge at inference without retraining; fine-tuning modifies model weights to teach new behaviors or styles."),
    ("Why use hybrid search in RAG?",
     "Hybrid search combines BM25 (exact term match) with dense embeddings (semantic match) to improve retrieval accuracy on diverse queries."),
    ("What is a transformer's attention mechanism?",
     "Attention lets each token weigh the relevance of every other token, enabling the model to model long-range dependencies in parallel."),
    ("What is prompt engineering?",
     "Prompt engineering is the practice of designing instructions and examples to steer LLMs toward correct, reliable outputs."),
    ("When should I use few-shot prompting?",
     "Use few-shot prompting when the task is ambiguous, when the model needs to follow a specific format, or when zero-shot attempts fail."),
    ("What is quantization?",
     "Quantization reduces the numeric precision of model weights (e.g., FP16 to INT8), cutting memory use with minimal accuracy loss."),
    ("What is a LoRA rank?",
     "LoRA rank (r) controls the dimension of the low-rank update matrices. Higher r = more capacity but more trainable parameters."),
    ("What is self-reflection in LLM agents?",
     "Self-reflection is a loop where the model critiques its own output against sources or rubrics, reducing hallucination."),
    ("What is a C&C server in botnet detection?",
     "A Command & Control server coordinates botnet devices to receive instructions and exfiltrate data."),
    ("What is YOLOv8?",
     "YOLOv8 is a real-time object detection model from Ultralytics that predicts bounding boxes and class labels in a single forward pass."),
    ("What is the difference between CNN and ViT?",
     "CNNs use convolutional filters with local receptive fields; Vision Transformers use attention over image patches for global context."),
    ("What is cross-validation?",
     "Cross-validation splits data into multiple folds and rotates which fold is used for validation, giving a more stable estimate of generalization."),
    ("What is AUC-ROC?",
     "AUC-ROC measures a classifier's ability to rank positive instances above negative ones across all decision thresholds."),
    ("What is feature engineering?",
     "Feature engineering is the process of creating informative inputs from raw data (e.g., loan-to-income ratio) that improve model performance."),
    ("What is MLflow used for?",
     "MLflow tracks machine learning experiments, logs metrics and artifacts, and manages model versioning and deployment."),
    ("What is DVC?",
     "DVC is a data version control tool that tracks datasets and model files alongside Git, keeping large files out of the repo."),
    ("What is GCP Cloud Run?",
     "Cloud Run is a serverless container platform on Google Cloud that auto-scales from zero and bills per request."),
]


def generate_qa_dataset(output_path: str = "data/qa_dataset.json", n_samples: int = 200, seed: int = 42):
    """Generate synthetic instruction-tuning dataset in Alpaca format."""
    random.seed(seed)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    samples = []
    for i in range(n_samples):
        q, a = DOMAIN_TOPICS[i % len(DOMAIN_TOPICS)]
        # Slight paraphrasing
        prefix = random.choice(["", "Explain: ", "In simple terms, ", "Briefly, "])
        samples.append({
            "instruction": f"{prefix}{q}",
            "input": "",
            "output": a,
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"[OK] Generated {len(samples)} Q&A samples -> {output_path}")
    return samples


if __name__ == "__main__":
    samples = generate_qa_dataset()
    print(f"\nSample entry:")
    print(json.dumps(samples[0], indent=2))
