# ✅ tests/test_api_search.py
import pytest
from httpx import AsyncClient
from fastapi import FastAPI
from app.api import search

app = FastAPI()
app.include_router(search.router, prefix="/api/v1/search")

@pytest.mark.asyncio
async def test_add_and_get_documents():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        res_add = await ac.post("/api/v1/search/add", params={
            "text": "Test product A",
            "key": "code",
            "value": "A123"
        })
        assert res_add.status_code == 200
        assert res_add.json()["status"] == "added"

        res_docs = await ac.get("/api/v1/search/documents")
        assert res_docs.status_code == 200
        assert any("Test product A" in d for d in res_docs.json().get("documents", []))

@pytest.mark.asyncio
async def test_delete_document():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        await ac.post("/api/v1/search/add", params={
            "text": "Delete me",
            "key": "code",
            "value": "DEL1"
        })

        res_delete = await ac.delete("/api/v1/search/delete", params={
            "key": "code",
            "value": "DEL1"
        })
        assert res_delete.status_code == 200
        assert res_delete.json()["status"] == "deleted"

@pytest.mark.asyncio
async def test_reset_collection():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        res = await ac.post("/api/v1/search/reset")
        assert res.status_code == 200
        assert res.json()["status"] == "reset"
