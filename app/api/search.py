# ✅ app/api/search.py
from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict
from app.services.embedding_service import EmbeddingService

router = APIRouter()
embedding_service = EmbeddingService()

@router.on_event("startup")
async def startup():
    await embedding_service.initialize()

@router.get("/documents")
async def get_documents():
    try:
        docs = await embedding_service.get_all_documents()
        return {"documents": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/add")
async def add_document(text: str = Query(...), key: str = Query(...), value: str = Query(...)):
    try:
        await embedding_service.add_document(text=text, metadata={key: value})
        return {"status": "added"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reset")
async def reset_collection():
    try:
        await embedding_service.reset_collection()
        return {"status": "reset"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/delete")
async def delete_by_metadata(key: str = Query(...), value: str = Query(...)):
    try:
        await embedding_service.delete_document_by_metadata(key, value)
        return {"status": "deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
