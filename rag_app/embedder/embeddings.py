from dotenv import load_dotenv
load_dotenv()

import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_MODEL_PATH = PROJECT_ROOT / 'models' / 'models--BAAI--bge-m3' / 'snapshots' / '5617a9f61b028005a4858fdac845db406aefb181'
# The smaller default keeps the source package portable.  When the original
# local bge-m3 model is present, it is still used automatically.
DEFAULT_PORTABLE_MODEL = 'BAAI/bge-small-zh-v1.5'

_instance = None


class LocalEmbeddingWrapper:
    def __init__(self):
        configured_model = os.getenv('EMBEDDING_MODEL', '').strip()
        if LOCAL_MODEL_PATH.exists() and not configured_model:
            model_source = str(LOCAL_MODEL_PATH)
            print(f'Using local embedding model: {model_source}')
        else:
            model_source = configured_model or DEFAULT_PORTABLE_MODEL
            print(f'Using portable embedding model: {model_source}')
        self._model = SentenceTransformer(model_source)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        vec = self._model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return vec.tolist()


def get_embedding_model():
    global _instance
    if _instance is None:
        _instance = LocalEmbeddingWrapper()
    return _instance
