# app/services/embedding_service.py
def get_retriever(self):
    """Return retriever for vector search"""
    return self.vectorstore.as_retriever(search_kwargs={"k": 5})
