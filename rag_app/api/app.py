from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag_app.retriever.rag_chain import RAGChain

app = FastAPI(title="Luckin QA", version="1.0.0")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = PROJECT_ROOT / "frontend"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_rag_chain():
    return RAGChain(top_k=4)


class QuestionRequest(BaseModel):
    question: str


class SourceInfo(BaseModel):
    content: str
    source: str


class AnswerResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "message": "Luckin QA running"}


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")

    try:
        rag = get_rag_chain()
        result = rag.invoke_with_sources(request.question)
        return AnswerResponse(
            answer=result["answer"],
            sources=[SourceInfo(**source) for source in result["sources"]],
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/health")
async def check_vector_db():
    try:
        rag = get_rag_chain()
        count = rag.vector_store._collection.count()
        return {"status": "ok", "document_count": count}
    except Exception as error:
        return {"status": "error", "detail": str(error)}


# Railway exposes one public port. This serves the web page and API together.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")