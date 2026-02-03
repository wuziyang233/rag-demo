# python
from pydantic import BaseModel
from typing import List

class IngestRequest(BaseModel):
    texts: List[str]

# python
from pydantic import BaseModel

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

# python
from pydantic import BaseModel
from typing import List, Dict

class QueryResponse(BaseModel):
    results: List[Dict]  # 每项包含 \`text\` 和 \`score\`

