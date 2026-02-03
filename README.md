安装依赖： pip install -r requirements.txt
运行服务： uvicorn app.main:app --reload
访问： http://localhost:8000/docs 查看自动生成的 Swagger UI

前端页面： http://localhost:8000/ （简单表单，可调用 /ingest 和 /query）

接口说明：
- GET /health: 健康检查
- POST /ingest: 写入文本到向量索引
- POST /query: 检索相关文本

示例请求：
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"texts":["Hello world","FastAPI is great","We build RAG demos"]}'

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is FastAPI?","top_k":3}'
```
