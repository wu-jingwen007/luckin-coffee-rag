from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


def split_by_headings(documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    all_splits = []
    
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ('#', 'heading_1'),
            ('##', 'heading_2'),
            ('###', 'heading_3'),
        ]
    )
    
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    for doc in documents:
        content = doc.page_content
        has_headers = bool(content.strip().startswith('#'))
        
        if has_headers:
            try:
                splits = header_splitter.split_text(content)
                for split in splits:
                    split.metadata.update(doc.metadata)
                    headings = []
                    for key in ['heading_1', 'heading_2', 'heading_3']:
                        if key in split.metadata and split.metadata[key]:
                            headings.append(split.metadata[key])
                    
                    if headings:
                        heading_info = ' > '.join(headings)
                        split.page_content = f'[{heading_info}]\n{split.page_content}'
                    
                    all_splits.extend(split.split_documents if hasattr(split, 'split_documents') else [split])
            except Exception:
                splits = char_splitter.split_documents([doc])
                all_splits.extend(splits)
        else:
            splits = char_splitter.split_documents([doc])
            all_splits.extend(splits)
    
    print(f'Split complete: {len(all_splits)} chunks')
    return all_splits
