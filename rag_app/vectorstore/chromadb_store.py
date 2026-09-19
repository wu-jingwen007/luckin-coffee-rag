import os
from typing import List

from langchain_core.documents import Document
from langchain_chroma import Chroma

from rag_app.embedder.embeddings import get_embedding_model
from rag_app.config.settings import CHROMA_PERSIST_DIR


class VectorStoreManager:
    def __init__(self, collection_name: str = 'luckin_knowledge'):
        self.collection_name = collection_name
        self.embedding_func = get_embedding_model()
        self.persist_dir = CHROMA_PERSIST_DIR
        self.vector_store = None
    
    def add_documents(self, documents: List[Document]):
        print(f'Writing {len(documents)} chunks to vector database...')
        
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_func,
            collection_name=self.collection_name,
            persist_directory=self.persist_dir,
        )
        
        count = self.vector_store._collection.count()
        print(f'Vector database ready: {count} records')
    
    def load_existing(self):
        if not os.path.exists(self.persist_dir):
            raise FileNotFoundError(
                f'Vector DB directory not found: {self.persist_dir}\n'
                f'Please run the preprocessing script first.'
            )
        
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_func,
            persist_directory=self.persist_dir,
        )
        
        count = self.vector_store._collection.count()
        print(f'Loaded vector database: {count} records')
        return self.vector_store
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        if self.vector_store is None:
            self.load_existing()
        return self.vector_store.similarity_search(query, k=k)
    
    def get_vector_store(self):
        if self.vector_store is None:
            self.load_existing()
        return self.vector_store
