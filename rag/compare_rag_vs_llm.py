import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "faiss_index"

def build_llm():
    return ChatOpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
    )

def llm_only(llm, question: str) -> str:
    return llm.invoke(question).content

def rag_answer(llm, question: str) -> tuple[str, list[str]]:
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.load_local(str(INDEX_DIR), emb, allow_dangerous_deserialization=True)
    docs = vs.similarity_search(question, k=4)

    context = "\n\n".join(d.page_content for d in docs)
    prompt = f"""你是一个严谨的助手，只能根据【上下文】回答【问题】。
如果上下文不足以回答，就说“上下文不足”。

【上下文】
{context}

【问题】
{question}
"""
    answer = llm.invoke(prompt).content
    chunks = [d.page_content for d in docs]
    return answer, chunks

def main():
    llm = build_llm()
    question = "这个项目的核心思想是什么？"

    a1 = llm_only(llm, question)
    a2, chunks = rag_answer(llm, question)

    print("=== 1) LLM Only（无检索） ===\n")
    print(a1)

    print("\n=== 2) RAG（先检索再回答）: 检索到的文本块 ===\n")
    for i, c in enumerate(chunks, 1):
        print(f"[{i}] {c}\n")

    print("=== 2) RAG 回答 ===\n")
    print(a2)

if __name__ == "__main__":
    main()
