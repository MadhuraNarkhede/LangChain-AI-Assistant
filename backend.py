from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Initialize FastAPI app
app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input model for API
class QueryInput(BaseModel):
    query: str

# Hugging Face model and tokenizer setup
model_name = "meta-llama/Llama-2-7b-chat-hf"
try:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, token="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"
    )
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cpu")
except Exception as e:
    raise RuntimeError(f"Error loading model/tokenizer: {e}")

# Define prompt template
template = """
You are an AI assistant. Respond clearly and concisely.

User: {query}

AI:
"""

# Generate response
def generate_response(query: str) -> str:
    try:
        inputs = tokenizer(
            template.format(query=query),
            return_tensors="pt",
            truncation=True,
            max_length=256,
        ).to("cpu")

        outputs = model.generate(
            inputs["input_ids"],
            max_length=150,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            temperature=0.7,
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("AI:", 1)[-1].strip()
        return response
    except Exception as e:
        raise RuntimeError(f"Error generating response: {e}")

# Define API endpoints
@app.get("/")
def root():
    return {"message": "Welcome to the Llama-2-7b-powered API!"}

@app.post("/query/")
def get_response(input_data: QueryInput):
    try:
        response = generate_response(input_data.query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {e}")
