from fastapi import FastAPI
import faiss
from transformers import AutoTokenizer, AutoModel
import requests
import os
import csv
import torch
from pydantic import BaseModel
import re

os.environ['TRANSFORMERS_CACHE'] = '/transformers_cache'

import os
API_KEY = os.getenv("API_KEY")
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"
class FormAIMaker(BaseModel):
    namaIklan: str
    namaProduk: str
    kategoriIklan: str

class Brief(BaseModel):
    prompt: str

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
def embed_text(texts):
    tokens = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**tokens)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

def load_csv_to_dict(file_path):
    dic = dict()
    with open(file_path, mode='r', newline='\n', encoding='utf-8') as csvfile:
        # Create a CSV DictReader
        csvs = csv.DictReader(csvfile)
        data = [row for row in csvs]
        # Convert the CSV data to a list of dictionaries
    if data[0].get('ID '):
        for dantum in data:
            dic[dantum['ID ']] = dantum
    return data, dic

file_path = 'dataset.csv'
data, dc = load_csv_to_dict(file_path)
idxs = [row['ID '] for row in data]
documents = [row['Deskripsi'] for row in data]
document_embeddings = embed_text(documents)
document_embeddings_np = document_embeddings.numpy()
index = faiss.IndexFlatL2(document_embeddings_np.shape[1])
index.add(document_embeddings_np)

file_path = 'dataset_search.csv'
contentsDoc, condc = load_csv_to_dict(file_path)
print(contentsDoc[0])
contents = [row['Deskripsi'] for row in contentsDoc]
contents_embeddings = embed_text(contents)
contents_embeddings_np = contents_embeddings.numpy()
index_contents = faiss.IndexFlatL2(contents_embeddings_np.shape[1])
index_contents.add(contents_embeddings_np)

app = FastAPI(title="Endorse")
@app.get("/")
def home():
    return {"message": "Hello World"}

@app.post("/brief")
def brief(form: FormAIMaker):
    global url
    headers = {
        'Content-Type': 'application/json'
    }
    prompt = f"""
        ### Context:
            Here is data for the advertisement brief:
            - Nama Iklan: {form.namaIklan}
            - Nama Produk: {form.namaProduk}
            - Kategori Iklan: {form.kategoriIklan}

        ### Question:
            Please generate a content brief that consist demographic, target audience, and content idea for the advertisement.
    """
    data = {
        "contents": [
            {
                'role': 'user',
                "parts": [
                    {
                    "text": prompt
                    }
                ]
            }
        ]
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return {"status_code": response.status_code, "message": response.json()["candidates"][0]["content"]["parts"][0]["text"]}
    else:
        #return with error code 
        return response.json()

@app.post("/similarity")
def similarity(payload: Brief):
    global index
    prompt = f"""
        ### Brief:
            {payload.prompt}
        ### Question:
            Please summarize the topic from brief above, make user it is compact such as "Young, Culinary" or "Sport, Old, Health and wellbeing" or "Beauty, Wellness, Woman" and etc        
    """
    global url
    headers = {
        'Content-Type': 'application/json'
    }
    payload = {
        "contents": [
            {
                'role': 'user',
                "parts": [
                    {
                    "text": prompt
                    }
                ]
            }
        ]
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        keyword = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        print(keyword)
        D, I = index.search(embed_text(keyword).numpy(), 3)
        retrieved_idx = [idxs[i] for i in I[0]]
        influencers = [data[i] for i in I[0]]
        return {"message": "Success", "data": influencers, "influencer_id": retrieved_idx}
    else:
        #return with error code 
        return response.json()
    

@app.get("/influencer/{influencer_id}")
def get_influencer(influencer_id: str):
    global dc
    if dc.get(influencer_id):
        return {"status": 200, "message": "Success", "influencer": dc[influencer_id]}
    return {"status": 400,"message": "Influencer not found"}

@app.get("/search")
def search(query: str):
    global index_contents, condc, contentsDoc
    D, I = index_contents.search(embed_text(query).numpy(), 4)
    contensResponse = [contentsDoc[i] for i in I[0]]
    return {"message": "Success", "data": contensResponse}
