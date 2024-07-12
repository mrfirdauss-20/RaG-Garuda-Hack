from transformers import AutoTokenizer
from fastapi import FastAPI
import faiss



app = FastAPI(title="Endorse")
@app.get("/")
def home():
    return {"message": "Hello World"}

