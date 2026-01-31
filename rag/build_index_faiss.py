from pathlib import Path
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "faiss_index"

def load_docs():
    docs = []
    pdf_path = DATA_DIR / "demo.pdf"
    txt_path = DATA_DIR / "demo.txt"

    if pdf_path.exists():
        docs += PyPDFLoader(str(pdf_path)).load()
    if txt_path.exists():
        docs += TextLoader(str(txt_path), encoding="utf-8").load()

    return docs

def main():
    docs = load_docs()
    if not docs:
        raise RuntimeError(f"No docs found in {DATA_DIR}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vs = FAISS.from_documents(chunks, emb)
    INDEX_DIR.mkdir(exist_ok=True)
    vs.save_local(str(INDEX_DIR))

    print(f"docs={len(docs)}, chunks={len(chunks)}, index={INDEX_DIR}")

if __name__ == "__main__":
    main()
