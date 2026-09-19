FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

# Build the ChromaDB index from the committed knowledge-base documents, then
# start FastAPI on Railway's dynamically assigned port.
CMD ["sh", "-c", "python rag_app/preprocess.py && uvicorn rag_app.api.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
