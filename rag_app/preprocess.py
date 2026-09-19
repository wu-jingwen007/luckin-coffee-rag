# RAG Application Preprocessing Script
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag_app.loader.document_loader import load_documents
from rag_app.loader.splitters import split_by_headings
from rag_app.vectorstore.chromadb_store import VectorStoreManager
from rag_app.config.settings import DOCUMENTS_DIR

def main():
    print('=' * 60)
    print('Luckin Coffee Knowledge Base - Data Preprocessing')
    print('=' * 60)
    
    print('\n[1/3] Loading documents...')
    documents = load_documents(str(DOCUMENTS_DIR))
    
    if not documents:
        print('Error: No documents found')
        sys.exit(1)
    
    print('\n[2/3] Splitting documents...')
    chunks = split_by_headings(
        documents,
        chunk_size=500,
        chunk_overlap=50,
    )
    
    if not chunks:
        print('Error: No chunks generated')
        sys.exit(1)
    
    print('\n[3/3] Embedding and storing in ChromaDB...')
    vector_manager = VectorStoreManager()
    vector_manager.add_documents(chunks)
    
    print('\n' + '=' * 60)
    print('Preprocessing complete! Vector database is ready.')
    print(f'Storage location: {vector_manager.persist_dir}')
    print('=' * 60)

if __name__ == '__main__':
    main()
