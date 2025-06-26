from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, List, Optional
from app.services.ai_service import AIService
from app.services.search_service import SearchService
from app.utils.logging import ChatLogger
from app.utils.metrics import MetricsCollector
import time
import uuid

router = APIRouter()

class ChatRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    context: Optional[Dict] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict] = []
    confidence: float = 0.0
    suggestions: List[str] = []
    context: Optional[Dict] = None
    response_time_ms: float = 0.0

# Initialize services
search_service = SearchService()
chat_logger = ChatLogger()
metrics_collector = MetricsCollector()

@router.post("/query", response_model=ChatResponse)
async def chat_query(request: ChatRequest):
    """Process chat query and return AI response"""
    start_time = time.time()
    
    try:
        # Get AI service from global state
        from app.main import ai_service
        
        if not ai_service:
            raise HTTPException(status_code=503, detail="AI service not available")
        
        user_id = request.user_id or str(uuid.uuid4())
        
        # Process query with AI service
        response_data = await ai_service.process_query(
            query=request.query,
            user_context=request.context
        )
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Log the interaction
        chat_logger.log_query(
            user_id=user_id,
            query=request.query,
            response=response_data,
            duration=response_time
        )
        
        # Record metrics
        metrics_collector.record_query(
            duration=response_time,
            confidence=response_data.get("confidence", 0.0)
        )
        
        # Format response
        response = ChatResponse(
            answer=response_data["answer"],
            sources=response_data.get("sources", []),
            confidence=response_data.get("confidence", 0.0),
            suggestions=response_data.get("suggestions", []),
            context=response_data.get("context"),
            response_time_ms=response_time * 1000
        )
        
        return response
        
    except Exception as e:
        print(f"Error in chat query: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/suggestions")
async def get_suggestions():
    """Get common query suggestions"""
    suggestions = [
        "Tìm vải cotton trắng 90cm còn trong kho không?",
        "Hướng dẫn kiểm tra nguyên phụ liệu đầu ca",
        "Mã hàng ABC123 nên lưu ở vị trí nào?",
        "Nếu hàng bị dư thừa thì phải làm gì?",
        "Hiển thị lịch sử nhập xuất hàng gần đây",
        "Danh sách các vị trí kho còn trống",
        "Quy trình đóng gói sản phẩm vải",
        "Kiểm tra tồn kho các mặt hàng sắp hết"
    ]
    
    return {
        "suggestions": suggestions,
        "categories": [
            {"name": "Tìm kiếm hàng hóa", "icon": "🔍"},
            {"name": "Quy trình thao tác", "icon": "📋"},
            {"name": "Vị trí lưu kho", "icon": "📍"},
            {"name": "Báo cáo tồn kho", "icon": "📊"}
        ]
    }

@router.post("/feedback")
async def submit_feedback(feedback: Dict):
    """Submit user feedback for response quality"""
    try:
        # Log feedback for training improvements
        chat_logger.logger.info(f"FEEDBACK: {feedback}")
        
        return {"status": "success", "message": "Feedback received"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))