import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
)

print(llm.invoke("用一句话解释 RAG 是什么").content)
