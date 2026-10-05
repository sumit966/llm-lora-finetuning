# Domain-Specific LLM Fine-Tuning with LoRA

Fine-tune an open-source LLM on a domain-specific Q&A dataset using **LoRA (Low-Rank Adaptation)**, then compare the fine-tuned model against the base model on a fixed evaluation set. Includes a FastAPI service for training, evaluation, and inference.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-1C3C3C?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

## Overview

Instead of retraining an entire LLM (which is expensive and slow), this project uses **LoRA** to inject small trainable low-rank matrices into the base model's attention layers. The base model stays frozen, and only ~1% of parameters are trained — but the model learns domain-specific behaviors.

The pipeline:

1. Generates (or loads) a domain-specific instruction-tuning dataset
2. Wraps the base model with LoRA adapters via HuggingFace PEFT
3. Fine-tunes on the dataset
4. Saves only the adapter weights (a few MB instead of GB)
5. Evaluates fine-tuned vs base model on a fixed eval set
6. Serves inference via a FastAPI endpoint

If the heavy training stack (torch, transformers, peft) is unavailable, the project falls back to a mock trainer that keeps the interface intact — so tests always pass, even in CI without GPU.

## Problem Statement

General-purpose LLMs fail on domain-specific tasks:

- **Wrong terminology** — they mix up jargon from different domains
- **Verbose answers** — instead of concise, on-point responses
- **Hallucinated facts** — they invent plausible-sounding but false details
- **No format control** — outputs don't follow your required structure

Full fine-tuning solves this but costs:

- $$$ (multiple GPUs for days)
- Storage (each fine-tuned model = 10+ GB)
- Risk of catastrophic forgetting

## Solution

LoRA fine-tuning delivers 90% of the benefit at 1% of the cost:

- **Frozen base model** — no catastrophic forgetting
- **Trainable low-rank adapters** — millions of params instead of billions
- **Tiny artifacts** — adapters are a few MB, easy to swap and version
- **Fast training** — hours instead of days on a single GPU

## Architecture

Synthetic Q&A Dataset (data/qa_dataset.json)
         |
         v
Tokenize + Format Prompts
         |
         v
Base LLM (frozen)  +  LoRA Adapters (trainable)
         |
         v
Training Loop (HuggingFace Trainer)
         |
         v
Save Adapter Weights (few MB)
         |
         v
Evaluate: Fine-tuned vs Base
         |
         v
FastAPI Endpoints: /train  /evaluate  /infer
         |
         v
Docker + GitHub Actions CI

## Tech Stack

| Category | Technologies |
|----------|-------------|
| Base Model | TinyLlama 1.1B (or any HF causal LM) |
| Fine-Tuning | LoRA via PEFT |
| Training | HuggingFace Transformers + Trainer |
| Datasets | HuggingFace Datasets |
| API | FastAPI, Uvicorn, Pydantic |
| Testing | pytest, httpx |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Language | Python 3.11+ |

## Project Structure

llm-lora-finetuning/
├── src/
│   ├── __init__.py
│   ├── data_generator.py       # Generate synthetic Q&A dataset
│   ├── lora_trainer.py         # LoRA fine-tuning (real or mock)
│   └── evaluate_model.py       # Base vs fine-tuned comparison
├── api/
│   ├── __init__.py
│   └── main.py                 # FastAPI service
├── tests/
│   ├── __init__.py
│   └── test_api.py             # pytest tests
├── data/                        # Generated dataset (git-ignored)
├── adapters/                    # LoRA adapter weights (git-ignored)
├── .github/workflows/ci.yml
├── Dockerfile
├── requirements.txt
├── requirements-optional.txt
├── .env.example
├── .gitignore
├── LICENSE
└── README.md

## Quick Start

### 1. Clone

git clone https://github.com/sumit966/llm-lora-finetuning.git
cd llm-lora-finetuning

### 2. Virtual environment

Windows:
python -m venv venv
venv\Scripts\activate

macOS / Linux:
python3 -m venv venv
source venv/bin/activate

### 3. Install base dependencies

pip install -r requirements.txt

### 4. (Optional) Install real LoRA training stack

pip install -r requirements-optional.txt

If this fails on Python 3.14 (torch/transformers wheels missing), skip it. The project automatically uses a mock trainer and still works fully.

### 5. (Optional) Configure environment

Copy .env.example to .env and edit:

HF_TOKEN=your-hf-token-if-needed
BASE_MODEL=tinyllama/tinyllama-1.1B-Chat-v1.0
MAX_LENGTH=256

### 6. Start the API

uvicorn api.main:app --reload

API runs at http://localhost:8000

### 7. Open Swagger UI

http://localhost:8000/docs

## API Usage

### POST /generate-dataset

Generate the synthetic Q&A dataset.

curl -X POST "http://localhost:8000/generate-dataset?n_samples=200"

Response:
{
  "message": "Generated 200 Q&A samples",
  "metrics": {"n_samples": 200}
}

### POST /train

Fine-tune with LoRA.

Request:
{
  "n_samples": 200,
  "epochs": 3,
  "lora_r": 8
}

Response:
{
  "message": "Training complete",
  "metrics": {
    "base_model": "tinyllama/tinyllama-1.1B-Chat-v1.0",
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj"],
    "train_samples": 200,
    "epochs": 3,
    "final_loss": 0.4,
    "mode": "mock"
  }
}

### POST /evaluate

Compare base vs fine-tuned on a fixed eval set.

Response:
{
  "base_accuracy": 0.52,
  "finetuned_accuracy": 0.78,
  "improvement_pct": 50.0
}

### POST /infer

Generate a response using the fine-tuned model (or mock).

Request:
{
  "instruction": "What is LoRA?",
  "max_new_tokens": 100
}

Response:
{
  "instruction": "What is LoRA?",
  "response": "...",
  "mode": "mock"
}

### Other Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| / | GET | API info |
| /health | GET | Health check |
| /docs | GET | Swagger UI |
| /redoc | GET | Alternative docs |

## LoRA Configuration

| Parameter | Value | Meaning |
|-----------|-------|---------|
| r (rank) | 8 | Dimension of low-rank matrices |
| alpha | 16 | Scaling factor (typically 2x rank) |
| dropout | 0.05 | Dropout on adapter output |
| target_modules | q_proj, v_proj | Which attention layers get adapters |

Why these values:

- r=8 is a common starting point — enough capacity for domain tuning without overfitting
- alpha=16 (2×r) is standard practice
- Adapting only q_proj and v_proj covers the attention mechanism cheaply

## Offline / Mock Mode

If torch, transformers, peft, or datasets are not installed:

- The training step prints realistic metrics and saves a metadata JSON
- The inference endpoint returns a template response
- Tests still pass because the API interface is unchanged

This makes the repo:

- Runnable in GitHub Actions without a GPU
- Testable on any machine
- Easy to swap in the real training stack later

## Testing

pytest tests/ -v

Tests cover:
- Root endpoint responds
- Health check returns expected status
- /generate-dataset creates N samples
- /train returns metrics
- /evaluate returns base vs fine-tuned accuracies
- /infer returns a response
- Invalid instruction returns 422

## Docker

Build:
docker build -t llm-lora-finetuning .

Run:
docker run -p 8000:8000 llm-lora-finetuning

## CI/CD Pipeline

Every push to main triggers GitHub Actions:

1. Install Python 3.11 + base dependencies
2. Run pytest test suite (runs in mock mode, no GPU required)

See .github/workflows/ci.yml.

## Key Learnings

- LoRA trains ~1% of parameters while matching full fine-tuning quality on narrow domains
- Adapter weights are ~10 MB vs ~5 GB for a full fine-tune — massive storage savings
- Freezing the base model prevents catastrophic forgetting
- Target module choice (q_proj, v_proj) covers attention cheaply
- Rank and alpha are the two main hyperparameters to tune
- Graceful fallback to mock mode keeps CI green without GPUs
- FastAPI + Pydantic gives free validation and Swagger docs

## Future Improvements

- Add QLoRA (4-bit quantization) for training on consumer GPUs
- Multi-adapter serving (swap LoRA per request)
- Evaluation with BLEU, ROUGE, and semantic similarity
- Real domain dataset (medical, legal, or finance)
- Deploy adapter to HuggingFace Hub
- Add streaming inference via Server-Sent Events
- Track experiments with Weights & Biases
- Serve via vLLM for high-throughput inference

## License

MIT License - see LICENSE file.

## Author

Sumit Raj
- M.Tech Applied AI & ML @ VNIT Nagpur
- Ex-Software Engineer Intern @ Salesforce
- GitHub: https://github.com/sumit966
- LinkedIn: https://www.linkedin.com/in/er-sumit-raj-/
- Portfolio: https://sumit966-github-io.vercel.app
- Email: info.sr0909@gmail.com

If you found this project useful, please consider giving it a star!
