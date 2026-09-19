 # 瑞幸咖啡知识库 — RAG 智能问答系统

 基于 LangChain + ChromaDB + DeepSeek 的企业知识问答系统。

 ## 项目结构

 ```
 luckin-rag-knowledge-base/
 ├── documents/              # 知识库文档（Markdown 格式）
 │   ├── 01-公司概述.md
 │   ├── 02-商业模式.md
 │   ├── 03-产品体系.md
 │   ├── 04-财务数据.md
 │   └── 05-行业竞争.md
 ├── rag_app/                # RAG 应用核心代码
 │   ├── config/             # 配置管理
 │   ├── loader/             # 文档加载与文本切分
 │   ├── embedder/           # Embedding 模型
 │   ├── vectorstore/        # ChromaDB 向量存储
 │   ├── llm/                # DeepSeek LLM 接口
 │   ├── retriever/          # RAG 检索链
 │   ├── api/                # FastAPI 服务
 │   └── preprocess.py       # 数据预处理脚本
 ├── frontend/               # 前端页面
 │   └── index.html
 ├── .env                    # 环境变量配置
 ├── requirements.txt        # Python 依赖
 └── start_server.py         # 一键启动脚本
 ```

 ## 快速开始

 ### 1. 配置 API Key

 编辑 `.env` 文件，填入你的 DeepSeek API Key：

 ```
 DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
 ```

### 2. 安装依赖

 ```bash
pip install -r requirements.txt
```

首次运行时，如未随项目保留本地模型，程序会自动下载 `BAAI/bge-small-zh-v1.5`
作为中文向量模型；下载完成后会缓存到本机。原项目目录中若存在本地 `bge-m3`
模型，则会优先使用该模型。

 ### 3. 一键启动

 ```bash
 python start_server.py
 ```

 启动后会依次执行：
 1. 预处理文档（加载 → 切分 → 向量化 → 存入 ChromaDB）
 2. 启动 API 服务（http://localhost:8000）
 3. 启动前端页面（http://localhost:8001）

 ### 4. 访问前端

 打开浏览器访问 http://localhost:8001/index.html 即可开始问答。

 ## 手动运行

 如果想分步操作：

 ```bash
 # 预处理数据
 python rag_app/preprocess.py

 # 启动 API 服务
 uvicorn rag_app.api.app:app --host 0.0.0.0 --port 8000 --reload

 # 单独查看前端
 cd frontend && python -m http.server 8001
 ```

 ## API 接口

 - `GET /` — 健康检查
 - `POST /ask` — 问答接口
   - 请求体：`{"question": "瑞幸咖啡成立于哪一年？"}`
   - 返回：`{"answer": "...", "sources": [{"content": "...", "source": "xxx.md"}]}`
 - `GET /docs` — Swagger API 文档

 ## 添加新文档

 将新的 Markdown 文件放入 `documents/` 目录，然后重新运行预处理：

 ```bash
 python rag_app/preprocess.py
 ```

 ## 技术栈

 - **框架：** LangChain
 - **向量数据库：** ChromaDB
 - **LLM：** DeepSeek（通过 OpenAI 兼容接口）
 - **API 服务：** FastAPI + Uvicorn
 - **前端：** 原生 HTML/CSS/JavaScript
