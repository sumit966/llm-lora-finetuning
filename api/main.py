"""FastAPI service for LoRA fine-tuning + inference simulation."""
import sys
import os
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from data_generator import generate_qa_dataset
from lora_trainer import train_lora
from evaluate_model import evaluate_models


app = FastAPI(
    title="LLM LoRA Fine-Tuning API",
    description="Domain-specific LLM fine-tuning with LoRA, plus evaluation",
    version="1.0.0",
)


class TrainRequest(BaseModel):
    n_samples: int = Field(200, ge=10, le=5000)
    epochs: int = Field(3, ge=1, le=20)
    lora_r: int = Field(8, ge=1, le=64)


class TrainResponse(BaseModel):
    message: str
    metrics: dict


class EvalResponse(BaseModel):
    base_accuracy: float
    finetuned_accuracy: float
    improvement_pct: float


class InferRequest(BaseModel):
    instruction: str = Field(..., min_length=3, max_length=500)
    max_new_tokens: int = Field(100, ge=10, le=500)


class InferResponse(BaseModel):
    instruction: str
    response: str
    mode: str


@app.get("/")
def root():
    return {"message": "LLM LoRA Fine-Tuning API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/generate-dataset", response_model=TrainResponse)
def generate_dataset(n_samples: int = 200):
    samples = generate_qa_dataset(n_samples=n_samples)
    return TrainResponse(
        message=f"Generated {len(samples)} Q&A samples",
        metrics={"n_samples": len(samples)},
    )


@app.post("/train", response_model=TrainResponse)
def train(req: TrainRequest):
    try:
        # Generate dataset if needed
        if not os.path.exists("data/qa_dataset.json"):
            generate_qa_dataset(n_samples=req.n_samples)

        metrics = train_lora(
            dataset_path="data/qa_dataset.json",
            output_dir="adapters",
            epochs=req.epochs,
            lora_r=req.lora_r,
        )
        return TrainResponse(message="Training complete", metrics=metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate", response_model=EvalResponse)
def evaluate():
    try:
        if not os.path.exists("data/qa_dataset.json"):
            generate_qa_dataset()
        results = evaluate_models()
        return EvalResponse(
            base_accuracy=results["base_model"]["accuracy"],
            finetuned_accuracy=results["finetuned_model"]["accuracy"],
            improvement_pct=results["improvement_pct"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest):
    """Inference endpoint. In mock mode, returns a template response.

    In real mode, loads the fine-tuned adapter and generates.
    """
    mode = "mock"

    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        adapter_dir = "adapters"
        if not os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
            raise FileNotFoundError("Adapter not found")

        base_model = os.getenv("BASE_MODEL", "tinyllama/tinyllama-1.1B-Chat-v1.0")
        tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", trust_remote_code=True)
        model = PeftModel.from_pretrained(base, adapter_dir)
        model.eval()

        prompt = f"### Instruction:\n{req.instruction}\n\n### Response:\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=req.max_new_tokens, do_sample=False)
        response = tokenizer.decode(out[0], skip_special_tokens=True).split("### Response:")[-1].strip()
        mode = "real"

    except Exception:
        # Mock response
        response = (
            f"[Fine-tuned LoRA response]\n\n"
            f"{req.instruction.strip()}\n\n"
            f"LoRA adapters tune the base model for this domain. "
            f"Low-rank updates keep trainable parameters small while "
            f"preserving pretrained knowledge."
        )

    return InferResponse(instruction=req.instruction, response=response, mode=mode)
