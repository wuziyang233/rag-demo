from pydantic import BaseModel
from typing import List, Dict

class IngestRequest(BaseModel):
    texts: List[str]

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3
    generate: bool = False

class QueryResponse(BaseModel):
    results: List[Dict]  # 每项包含 \`text\` 和 \`score\`
