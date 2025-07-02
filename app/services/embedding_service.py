# ✅ app/services/embedding_service.py
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import os

class EmbeddingService:
    def __init__(self):
        self.vectorstore = None

    async def initialize(self):
        os.makedirs("chroma_db", exist_ok=True)
        self.vectorstore = Chroma(
            collection_name="warehouse_collection",
            embedding_function=HuggingFaceEmbeddings(),
            persist_directory="chroma_db"
        )

    def get_retriever(self):
        return self.vectorstore.as_retriever(search_kwargs={"k": 5})

    async def add_document(self, text: str, metadata: dict):
        self.vectorstore.add_texts([text], metadatas=[metadata])

    async def reset_collection(self):
        if self.vectorstore is not None:
            self.vectorstore.delete_collection()
            await self.initialize()

    async def get_all_documents(self):
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized")
        return self.vectorstore._collection.get()["documents"]

    async def delete_document_by_metadata(self, key: str, value: str):
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized")
        self.vectorstore._collection.delete(
            where={key: value}
        )
