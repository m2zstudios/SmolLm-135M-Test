from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_ID = "HuggingFaceTB/SmolLM-135M"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32
)

app = FastAPI()

@app.get("/")
def health():
    return {"status": "ok", "model": "SmolLM-135M"}

@app.post("/generate")
def generate(prompt: str, max_new_tokens: int = 100):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7
        )
    return {
        "response": tokenizer.decode(output[0], skip_special_tokens=True)
    }
