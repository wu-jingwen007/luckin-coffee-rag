import re
from pathlib import Path
from typing import List

from langchain_core.documents import Document


def _clean_markdown(text):
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_documents(documents_dir):
    doc_path = Path(documents_dir)
    if not doc_path.exists():
        raise FileNotFoundError(f"Document directory not found: {documents_dir}")

    documents = []
    md_files = sorted(doc_path.glob("*.md"))

    if not md_files:
        raise ValueError(f"No .md files found in {documents_dir}")

    for file_path in md_files:
        text = file_path.read_text(encoding="utf-8")
        cleaned_text = _clean_markdown(text)

        doc = Document(
            page_content=cleaned_text,
            metadata={
                "source": file_path.name,
                "file_path": str(file_path),
            },
        )
        documents.append(doc)

    print(f"Loaded {len(documents)} documents")
    for doc in documents:
        src = doc.metadata["source"]
        clen = len(doc.page_content)
        print(f"  - {src} ({clen} chars)")

    return documents
