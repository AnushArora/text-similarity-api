from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import torch

# Load the model
model = SentenceTransformer('semantic_model')  # previously saved model

# Define request format
class TextPair(BaseModel):
    text1: str
    text2: str

# Create FastAPI app
app = FastAPI()

@app.post("/")
def compute_similarity(pair: TextPair):
    try:
        # Encode texts
        emb1 = model.encode(pair.text1, convert_to_tensor=True)
        emb2 = model.encode(pair.text2, convert_to_tensor=True)

        # Compute cosine similarity
        similarity = util.cos_sim(emb1, emb2).item()

        # Return formatted response
        return {"similarity score": round(similarity, 4)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
