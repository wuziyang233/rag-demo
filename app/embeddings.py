# python
import os
import json
import pickle
import asyncio
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv
import httpx

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_DIR = "faiss_index"
INDEX_FILE = os.path.join(INDEX_DIR, "index.faiss")
META_FILE = os.path.join(INDEX_DIR, "meta.pkl")

_model = None

def _load_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def _ensure_dir():
    os.makedirs(INDEX_DIR, exist_ok=True)

def _load_index():
    _ensure_dir()
    if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(META_FILE, "rb") as f:
            meta = pickle.load(f)
        return index, meta
    else:
        return None, []

def _save_index(index, meta):
    _ensure_dir()
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump(meta, f)

def _create_index(dim: int):
    index = faiss.IndexFlatL2(dim)
    return index

async def ingest_texts(texts: List[str]) -> List[int]:
    def sync_work(texts):
        model = _load_model()
        embeds = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        index, meta = _load_index()
        if index is None:
            index = _create_index(embeds.shape[1])
            meta = []
        index.add(embeds)
        start_id = len(meta)
        ids = list(range(start_id, start_id + len(texts)))
        meta.extend(texts)
        _save_index(index, meta)
        return ids
    return await asyncio.get_running_loop().run_in_executor(None, sync_work, texts)

async def query_index(question: str, top_k: int = 3) -> List[Dict]:
    def sync_work(question, top_k):
        model = _load_model()
        q_emb = model.encode([question], convert_to_numpy=True)
        index, meta = _load_index()
        if index is None or len(meta) == 0:
            return []
        D, I = index.search(q_emb, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(meta):
                continue
            results.append({"text": meta[idx], "score": float(score), "id": int(idx)})
        return results
    return await asyncio.get_running_loop().run_in_executor(None, sync_work, question, top_k)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini-realtime-preview-2024-12-17")
OPENAI_CHAT_COMPLETIONS_URL = os.getenv("OPENAI_CHAT_COMPLETIONS_URL")

async def generate_answer(question: str, results: List[Dict]) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")

    if not OPENAI_BASE_URL and not OPENAI_CHAT_COMPLETIONS_URL:
        raise RuntimeError("OPENAI_BASE_URL or OPENAI_CHAT_COMPLETIONS_URL is not set")

    context = "\n\n".join(
        f"[Doc {i + 1}] {item['text']}" for i, item in enumerate(results)
    )

    system_prompt = (
        "You are a helpful assistant. Answer the question using only the provided context. "
        "If the context does not contain the answer, say you do not know."
    )

    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    url = OPENAI_CHAT_COMPLETIONS_URL
    if not url:
        url = f"{OPENAI_BASE_URL.rstrip('/')}/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    return data["choices"][0]["message"]["content"].strip()
