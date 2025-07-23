import json
from typing import List
from langchain_core.documents import Document

def convert_to_documents(chunks: List[dict]) -> List[Document]:
    documents = []

    for idx, chunk in enumerate(chunks):
        content = chunk.get("text") or chunk.get("content")
        if not content or not content.strip():
            continue  # Skip empty or whitespace-only chunks

        metadata = {
            "source": chunk.get("source_url") or chunk.get("url", ""),
            "title": chunk.get("title") or chunk.get("page_title", ""),
            "chunk_index": chunk.get("chunk_index", idx),
            "total_chunks": chunk.get("total_chunks", len(chunks)),
            "section": chunk.get("section_title", "")
        }

        doc = Document(page_content=content.strip(), metadata=metadata)
        documents.append(doc)

    return documents
