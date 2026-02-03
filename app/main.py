from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from .schemas import IngestRequest, QueryRequest
from .embeddings import ingest_texts, query_index

app = FastAPI(title="RAG Demo API")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/ingest")
async def ingest(req: IngestRequest):
    try:
        ids = await ingest_texts(req.texts)
        return {"inserted": len(ids), "ids": ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query(req: QueryRequest):
    try:
        results = await query_index(req.question, top_k=req.top_k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
