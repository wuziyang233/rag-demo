import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "faiss_index"

def main():
    # 1) LLM：DeepSeek（走 OpenAI 兼容接口）
    llm = ChatOpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
    )

    # 2) Embedding：要和你建索引时用的一模一样
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 3) 载入本地 FAISS 索引
    vs = FAISS.load_local(str(INDEX_DIR), emb, allow_dangerous_deserialization=True)

    # 4) 检索：从向量库里拿最相关的 4 段文本
    question = "这个项目的核心思想是什么？"
    docs = vs.similarity_search(question, k=4)

    print("=== 检索到的文本块（将作为上下文）===\n")
    for i, d in enumerate(docs, 1):
        print(f"[{i}] {d.page_content}\n")

    # 5) 生成：把“上下文 + 问题”交给 DeepSeek 回答
    context = "\n\n".join(d.page_content for d in docs)
    prompt = f"""你是一个严谨的助手，只能根据【上下文】回答【问题】。
如果上下文不足以回答，就说“上下文不足”。

【上下文】
{context}

【问题】
{question}
"""
    answer = llm.invoke(prompt).content
    print("=== 模型回答 ===\n")
    print(answer)

if __name__ == "__main__":
    main()
