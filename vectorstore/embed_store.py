from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from opensearchpy import OpenSearch, RequestsHttpConnection, exceptions as opensearch_exceptions
from typing import List
from dotenv import load_dotenv
import os
from settings import OpenSearchSettings


load_dotenv()




opensearch_config = OpenSearchSettings()

INDEX_NAME = opensearch_config.index_name
VECTOR_DIM = 768  # Should match the embedding model output
OPENSEARCH_URL = f"http://{opensearch_config.host}:{opensearch_config.port}"


def connect_to_opensearch() -> OpenSearch | None:
    """Connect to OpenSearch instance using environment variables."""
    try:
        client = OpenSearch(
            hosts=[{
                "host": opensearch_config.host,
                "port": opensearch_config.port,
            }],
            http_auth=(
                opensearch_config.user,
                opensearch_config.password.get_secret_value()
            ),
            use_ssl=opensearch_config.use_ssl,
            verify_certs=opensearch_config.verify_certs,
            connection_class=RequestsHttpConnection,

        )
        print("[INFO] Connected to OpenSearch")
        return client
    except Exception as e:
        print(f"[ERROR] Failed to connect to OpenSearch: {e}")
        return None


def store_documents(docs: List[Document]):
    """Store documents in OpenSearch with embeddings."""
    if not docs:
        print("[WARN] No documents to store.")
        return

    client = connect_to_opensearch()
    if client is None:
        return

    ensure_index_exists(client, INDEX_NAME)
    vectorstore = create_vectorstore(client)

    try:
        vectorstore.add_documents(docs)
        print(f"[INFO] Stored {len(docs)} documents with embeddings.")
    except Exception as e:
        print(f"[ERROR] Failed to add documents to OpenSearch: {e}")


def ensure_index_exists(client: OpenSearch, index_name: str):
    """Create index with proper mapping if it doesn't exist."""
    if client.indices.exists(index=index_name):
        print(f"[INFO] Index '{index_name}' already exists.")
        return

    mapping = {
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "knn": True
            }
        },
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "metadata": {"type": "object"},
                "vector_field": {
                    "type": "knn_vector",
                    "dimension": VECTOR_DIM
                }
            }
        }
    }

    try:
        client.indices.create(index=index_name, body=mapping)
        print(f"[INFO] Created index '{index_name}' with vector mapping.")
    except opensearch_exceptions.RequestError as e:
        print(f"[ERROR] Failed to create index: {e.info}")
    except Exception as e:
        print(f"[ERROR] Unexpected error creating index: {e}")


def create_vectorstore(client: OpenSearch) -> OpenSearchVectorSearch:
    """Create the OpenSearch vector store with HuggingFace embeddings."""
    embedding_model = HuggingFaceEmbeddings(
        model_name=opensearch_config.embedding_model_name,
    )
    opensearch_vector_search = OpenSearchVectorSearch(
        index_name=INDEX_NAME,
        opensearch_client=client,
        opensearch_url=OPENSEARCH_URL,
        embedding_function=embedding_model,
        bulk_size=2000,
        vector_field="vector_field",  # Ensure this matches your index mapping
        embedding_dim=VECTOR_DIM,
    )
    print(f"[INFO] Created OpenSearch vector store with index '{INDEX_NAME}'.")

    return opensearch_vector_search

