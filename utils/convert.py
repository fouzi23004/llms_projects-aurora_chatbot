from langchain_core.documents import Document

def convert_to_documents(chunks: list[dict]) -> list[Document]:
    documents = []
    for chunk in chunks:
        content = chunk.get("text") or chunk.get("content")
        if not content:
            continue  # Skip empty chunks

        metadata = {
            "source": chunk.get("source_url") or chunk.get("url"),
            "title": chunk.get("title") or chunk.get("page_title"),
            "chunk_index": chunk.get("chunk_index"),
            "total_chunks": chunk.get("total_chunks"),
            "section": chunk.get("section_title", "")
        }

        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)

    return documents
