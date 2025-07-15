from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def save_to_vectorstore(docs, path="faiss_index"):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(path)
