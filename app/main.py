from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.api import chat, inventory, search
from app.services.ai_service import AIService
from app.services.embedding_service import EmbeddingService
import asyncio

app = FastAPI(title="Warehouse Chatbot API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global services
ai_service = None
embedding_service = None

@app.on_event("startup")
async def startup_event():
    global ai_service, embedding_service
    
    # Initialize embedding service
    embedding_service = EmbeddingService()
    await embedding_service.initialize()
    
    # Initialize AI service
    ai_service = AIService(embedding_service)
    await ai_service.initialize()
    
    print("🚀 Warehouse Chatbot API started successfully!")

# Include routers
app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])
app.include_router(inventory.router, prefix="/api/v1/inventory", tags=["inventory"])
app.include_router(search.router, prefix="/api/v1/search", tags=["search"])

@app.get("/")
async def root():
    return {"message": "Warehouse Chatbot API is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "ai_service": ai_service is not None}