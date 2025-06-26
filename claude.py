### 3.3 Main API (app/main.py)
```python
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
```

### 3.4 AI Service (app/services/ai_service.py)
```python
from langchain.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from .embedding_service import EmbeddingService
from .search_service import SearchService
import json

class AIService:
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.search_service = SearchService()
        self.llm = None
        self.chain = None
        self.memory = ConversationBufferWindowMemory(
            k=5,
            memory_key="chat_history",
            return_messages=True
        )
        
    async def initialize(self):
        """Initialize LLM and conversation chain"""
        try:
            # Initialize Ollama (hoặc OpenAI)
            self.llm = Ollama(
                model="llama2:7b-chat",  # Hoặc "mistral:7b"
                temperature=0.1,
                base_url="http://localhost:11434"
            )
            
            # Create conversation chain
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.embedding_service.get_retriever(),
                memory=self.memory,
                verbose=True,
                return_source_documents=True
            )
            
            print("✅ AI Service initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing AI Service: {e}")
            raise e
    
    async def process_query(self, query: str, user_context: dict = None):
        """Process user query and return response"""
        try:
            # Preprocess Vietnamese query
            processed_query = self._preprocess_vietnamese(query)
            
            # Get context from database
            context = await self._get_context(processed_query)
            
            # Create enhanced prompt
            enhanced_query = self._create_enhanced_prompt(processed_query, context)
            
            # Get response from LLM
            result = await self.chain.acall({"question": enhanced_query})
            
            # Post-process response
            response = self._postprocess_response(result, context)
            
            return response
            
        except Exception as e:
            print(f"Error processing query: {e}")
            return {
                "answer": "Xin lỗi, tôi không thể xử lý câu hỏi này lúc này. Vui lòng thử lại sau.",
                "sources": [],
                "confidence": 0.0
            }
    
    def _preprocess_vietnamese(self, text: str) -> str:
        """Preprocess Vietnamese text"""
        from underthesea import word_tokenize
        import re
        
        # Normalize text
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Tokenize Vietnamese
        tokens = word_tokenize(text)
        
        return ' '.join(tokens)
    
    async def _get_context(self, query: str) -> dict:
        """Get relevant context from database"""
        context = {
            "products": [],
            "procedures": [],
            "locations": [],
            "recent_transactions": []
        }
        
        try:
            # Search for relevant products
            products = await self.search_service.search_products(query)
            context["products"] = products[:5]  # Top 5 results
            
            # Search for relevant procedures
            procedures = await self.search_service.search_procedures(query)
            context["procedures"] = procedures[:3]  # Top 3 results
            
            # Get location info if product codes mentioned
            locations = await self.search_service.search_locations(query)
            context["locations"] = locations[:3]
            
        except Exception as e:
            print(f"Error getting context: {e}")
        
        return context
    
    def _create_enhanced_prompt(self, query: str, context: dict) -> str:
        """Create enhanced prompt with context"""
        
        template = """
Bạn là trợ lý AI của kho hàng, chuyên hỗ trợ nhân viên kho tra cứu thông tin.

THÔNG TIN BỐI CẢNH:
Sản phẩm liên quan: {products}
Quy trình liên quan: {procedures}
Vị trí kho: {locations}

NGUYÊN TẮC TRẢ LỜI:
1. Trả lời bằng tiếng Việt, ngắn gọn và chính xác
2. Ưu tiên thông tin từ dữ liệu kho hàng
3. Nếu không tìm thấy thông tin, hướng dẫn cách tra cứu
4. Đưa ra gợi ý cụ thể và thực tế

CÂU HỎI: {query}

TRẢ LỜI:
"""
        
        return template.format(
            products=json.dumps(context["products"], ensure_ascii=False),
            procedures=json.dumps(context["procedures"], ensure_ascii=False),
            locations=json.dumps(context["locations"], ensure_ascii=False),
            query=query
        )
    
    def _postprocess_response(self, result: dict, context: dict) -> dict:
        """Post-process LLM response"""
        
        response = {
            "answer": result.get("answer", ""),
            "sources": [],
            "confidence": 0.8,  # Default confidence
            "context": context,
            "suggestions": []
        }
        
        # Extract sources from result
        if "source_documents" in result:
            for doc in result["source_documents"]:
                response["sources"].append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
        
        # Add suggestions based on context
        if context["products"]:
            response["suggestions"].extend([
                f"Xem chi tiết sản phẩm {p['name']}" 
                for p in context["products"][:2]
            ])
        
        return response
```

## 5. Deployment

### 5.1 Docker Configuration
```yaml
# docker-compose.yml
version: '3.8'

services:
  # PostgreSQL Database
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: warehouse_db
      POSTGRES_USER: warehouse_user
      POSTGRES_PASSWORD: warehouse_pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  # Ollama for Local LLM
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    environment:
      - OLLAMA_HOST=0.0.0.0

  # API Backend
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://warehouse_user:warehouse_pass@postgres:5432/warehouse_db
      - OLLAMA_BASE_URL=http://ollama:11434
    depends_on:
      - postgres
      - ollama
    volumes:
      - ./app:/app

  # ChromaDB Vector Store
  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8001:8000"
    volumes:
      - chroma_data:/chroma/chroma

volumes:
  postgres_data:
  ollama_data:
  chroma_data:
```

### 5.2 Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

## 6. Tài nguyên huấn luyện

### 6.1 Dữ liệu huấn luyện mẫu
```json
{
  "products": [
    {
      "code": "VT001",
      "name": "Vải cotton trắng 90cm",
      "category": "Vải",
      "keywords": ["vải", "cotton", "trắng", "90cm", "cotton trắng"],
      "common_questions": [
        "Vải cotton trắng 90cm còn không?",
        "Mã VT001 ở đâu?",
        "Tìm vải cotton trắng"
      ]
    }
  ],
  "procedures": [
    {
      "title": "Kiểm tra nguyên phụ liệu đầu ca",
      "steps": [
        "Kiểm tra phiếu nhập kho",
        "Kiểm tra ngoại quan sản phẩm",
        "Đo kiểm kích thước",
        "Ghi nhận kết quả"
      ],
      "keywords": ["kiểm tra", "đầu ca", "nguyên liệu", "phụ liệu"]
    }
  ],
  "faqs": [
    {
      "question": "Hàng bị dư thừa thì phải làm gì?",
      "answer": "Khi phát hiện hàng dư thừa: 1) Kiểm tra lại số lượng, 2) Báo cáo cho trưởng ca, 3) Tạo phiếu điều chỉnh, 4) Cập nhật hệ thống"
    }
  ]
}
```

### 6.2 Script chuẩn bị dữ liệu
```python
# scripts/prepare_training_data.py
import json
import asyncio
from app.services.embedding_service import EmbeddingService

async def prepare_embeddings():
    """Prepare embeddings for training data"""
    embedding_service = EmbeddingService()
    await embedding_service.initialize()
    
    # Load training data
    with open('data/training_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create embeddings for products
    for product in data['products']:
        text = f"{product['name']} {product['code']} {' '.join(product['keywords'])}"
        await embedding_service.add_document(
            text=text,
            metadata={
                'type': 'product',
                'code': product['code'],
                'name': product['name']
            }
        )
    
    # Create embeddings for procedures
    for procedure in data['procedures']:
        text = f"{procedure['title']} {' '.join(procedure['steps'])}"
        await embedding_service.add_document(
            text=text,
            metadata={
                'type': 'procedure',
                'title': procedure['title']
            }
        )
    
    print("✅ Embeddings created successfully!")

if __name__ == "__main__":
    asyncio.run(prepare_embeddings())
```

## 7. Monitoring & Analytics

### 7.1 Logging System
```python
# app/utils/logging.py
import logging
import json
from datetime import datetime

class ChatLogger:
    def __init__(self):
        self.logger = logging.getLogger("warehouse_chatbot")
        self.logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler("logs/chatbot.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_query(self, user_id: str, query: str, response: dict, duration: float):
        """Log user query and response"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "query": query,
            "response_length": len(response.get("answer", "")),
            "confidence": response.get("confidence", 0),
            "sources_count": len(response.get("sources", [])),
            "duration_ms": duration * 1000
        }
        
        self.logger.info(f"QUERY: {json.dumps(log_data, ensure_ascii=False)}")
```

### 7.2 Performance Metrics
```python
# app/utils/metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
query_counter = Counter('chatbot_queries_total', 'Total number of queries')
response_time = Histogram('chatbot_response_time_seconds', 'Response time in seconds')
confidence_gauge = Gauge('chatbot_confidence_score', 'Average confidence score')

class MetricsCollector:
    @staticmethod
    def record_query(duration: float, confidence: float):
        query_counter.inc()
        response_time.observe(duration)
        confidence_gauge.set(confidence)
```

## 10. Mã nguồn chi tiết

### 10.1 Search Service (app/services/search_service.py)
```python
from sqlalchemy import and_, or_, func, text
from app.database import get_db
from app.models.product import Product
from app.models.inventory import Inventory, InventoryTransaction
from app.models.procedure import Procedure
from app.models.warehouse_location import WarehouseLocation
from fuzzywuzzy import fuzz
from typing import List, Dict, Optional
import re

class SearchService:
    def __init__(self):
        self.db = get_db()
    
    async def search_products(self, query: str, limit: int = 10) -> List[Dict]:
        """Search products by name, code, or keywords"""
        try:
            # Normalize query
            normalized_query = self._normalize_vietnamese(query)
            
            # SQL search with full-text and fuzzy matching
            sql_query = """
            SELECT p.*, i.quantity, i.last_updated, wl.location_code
            FROM products p
            LEFT JOIN inventory i ON p.id = i.product_id
            LEFT JOIN warehouse_locations wl ON i.location_id = wl.id
            WHERE 
                LOWER(p.name) LIKE LOWER(%s) OR
                LOWER(p.code) LIKE LOWER(%s) OR
                LOWER(p.description) LIKE LOWER(%s) OR
                to_tsvector('simple', p.name || ' ' || p.code || ' ' || COALESCE(p.description, '')) 
                @@ plainto_tsquery('simple', %s)
            ORDER BY 
                CASE 
                    WHEN LOWER(p.code) = LOWER(%s) THEN 1
                    WHEN LOWER(p.name) LIKE LOWER(%s) THEN 2
                    ELSE 3
                END,
                p.name
            LIMIT %s
            """
            
            search_term = f"%{normalized_query}%"
            
            result = await self.db.execute(
                text(sql_query),
                (search_term, search_term, search_term, normalized_query, 
                 normalized_query, search_term, limit)
            )
            
            products = []
            for row in result.fetchall():
                products.append({
                    "id": str(row.id),
                    "code": row.code,
                    "name": row.name,
                    "category": row.category,
                    "description": row.description,
                    "unit": row.unit,
                    "current_stock": row.quantity or 0,
                    "location": row.location_code,
                    "last_updated": row.last_updated.isoformat() if row.last_updated else None
                })
            
            return products
            
        except Exception as e:
            print(f"Error searching products: {e}")
            return []
    
    async def search_procedures(self, query: str, limit: int = 5) -> List[Dict]:
        """Search procedures by title, content, or tags"""
        try:
            normalized_query = self._normalize_vietnamese(query)
            
            sql_query = """
            SELECT *
            FROM procedures
            WHERE 
                LOWER(title) LIKE LOWER(%s) OR
                LOWER(content) LIKE LOWER(%s) OR
                %s = ANY(LOWER(tags::text)::text[]) OR
                to_tsvector('simple', title || ' ' || content) 
                @@ plainto_tsquery('simple', %s)
            ORDER BY 
                CASE 
                    WHEN LOWER(title) LIKE LOWER(%s) THEN 1
                    WHEN %s = ANY(LOWER(tags::text)::text[]) THEN 2
                    ELSE 3
                END,
                created_at DESC
            LIMIT %s
            """
            
            search_term = f"%{normalized_query}%"
            
            result = await self.db.execute(
                text(sql_query),
                (search_term, search_term, normalized_query.lower(), normalized_query,
                 search_term, normalized_query.lower(), limit)
            )
            
            procedures = []
            for row in result.fetchall():
                procedures.append({
                    "id": str(row.id),
                    "title": row.title,
                    "category": row.category,
                    "content": row.content[:200] + "..." if len(row.content) > 200 else row.content,
                    "steps": row.steps,
                    "tags": row.tags
                })
            
            return procedures
            
        except Exception as e:
            print(f"Error searching procedures: {e}")
            return []
    
    async def search_locations(self, query: str, limit: int = 5) -> List[Dict]:
        """Search warehouse locations"""
        try:
            # Extract potential location codes from query
            location_patterns = re.findall(r'[A-Z]-?\d+-?\d+-?\d+', query.upper())
            
            conditions = []
            params = []
            
            if location_patterns:
                for pattern in location_patterns:
                    conditions.append("location_code LIKE %s")
                    params.append(f"%{pattern}%")
            
            # Add general search
            if query:
                normalized_query = self._normalize_vietnamese(query)
                conditions.append("LOWER(location_code) LIKE LOWER(%s)")
                params.append(f"%{normalized_query}%")
            
            if not conditions:
                return []
            
            sql_query = f"""
            SELECT wl.*, p.name as product_name, p.code as product_code, i.quantity
            FROM warehouse_locations wl
            LEFT JOIN inventory i ON wl.id = i.location_id
            LEFT JOIN products p ON i.product_id = p.id
            WHERE {' OR '.join(conditions)}
            ORDER BY wl.zone, wl.row_number, wl.shelf_number, wl.level
            LIMIT %s
            """
            
            params.append(limit)
            
            result = await self.db.execute(text(sql_query), params)
            
            locations = []
            for row in result.fetchall():
                locations.append({
                    "id": str(row.id),
                    "location_code": row.location_code,
                    "zone": row.zone,
                    "row_number": row.row_number,
                    "shelf_number": row.shelf_number,
                    "level": row.level,
                    "capacity": row.capacity,
                    "current_product": {
                        "name": row.product_name,
                        "code": row.product_code,
                        "quantity": row.quantity
                    } if row.product_name else None
                })
            
            return locations
            
        except Exception as e:
            print(f"Error searching locations: {e}")
            return []
    
    async def get_recent_transactions(self, product_code: str = None, limit: int = 10) -> List[Dict]:
        """Get recent inventory transactions"""
        try:
            conditions = []
            params = []
            
            if product_code:
                conditions.append("p.code = %s")
                params.append(product_code)
            
            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
            
            sql_query = f"""
            SELECT it.*, p.name as product_name, p.code as product_code,
                   wl.location_code, it.created_at
            FROM inventory_transactions it
            JOIN products p ON it.product_id = p.id
            JOIN warehouse_locations wl ON it.location_id = wl.id
            {where_clause}
            ORDER BY it.created_at DESC
            LIMIT %s
            """
            
            params.append(limit)
            
            result = await self.db.execute(text(sql_query), params)
            
            transactions = []
            for row in result.fetchall():
                transactions.append({
                    "id": str(row.id),
                    "product_name": row.product_name,
                    "product_code": row.product_code,
                    "location_code": row.location_code,
                    "transaction_type": row.transaction_type,
                    "quantity": row.quantity,
                    "reference_number": row.reference_number,
                    "notes": row.notes,
                    "created_by": row.created_by,
                    "created_at": row.created_at.isoformat()
                })
            
            return transactions
            
        except Exception as e:
            print(f"Error getting recent transactions: {e}")
            return []
    
    def _normalize_vietnamese(self, text: str) -> str:
        """Normalize Vietnamese text for better search"""
        import unicodedata
        
        # Remove diacritics
        normalized = unicodedata.normalize('NFD', text)
        ascii_text = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
        
        # Clean up
        cleaned = re.sub(r'[^\w\s]', ' ', ascii_text)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    async def get_stock_status(self, product_code: str) -> Dict:
        """Get detailed stock status for a product"""
        try:
            sql_query = """
            SELECT p.*, 
                   SUM(i.quantity) as total_stock,
                   SUM(i.reserved_quantity) as total_reserved,
                   COUNT(DISTINCT i.location_id) as location_count,
                   array_agg(
                       json_build_object(
                           'location_code', wl.location_code,
                           'quantity', i.quantity,
                           'last_updated', i.last_updated
                       )
                   ) as locations
            FROM products p
            LEFT JOIN inventory i ON p.id = i.product_id
            LEFT JOIN warehouse_locations wl ON i.location_id = wl.id
            WHERE p.code = %s
            GROUP BY p.id, p.code, p.name, p.category, p.description, p.unit, p.min_stock
            """
            
            result = await self.db.execute(text(sql_query), (product_code,))
            row = result.fetchone()
            
            if not row:
                return {"error": f"Không tìm thấy sản phẩm với mã {product_code}"}
            
            # Get recent transactions
            recent_transactions = await self.get_recent_transactions(product_code, 5)
            
            available_stock = (row.total_stock or 0) - (row.total_reserved or 0)
            
            return {
                "product": {
                    "code": row.code,
                    "name": row.name,
                    "category": row.category,
                    "description": row.description,
                    "unit": row.unit,
                    "min_stock": row.min_stock
                },
                "stock_info": {
                    "total_stock": row.total_stock or 0,
                    "reserved": row.total_reserved or 0,
                    "available": available_stock,
                    "location_count": row.location_count or 0,
                    "status": self._get_stock_status_text(available_stock, row.min_stock)
                },
                "locations": row.locations or [],
                "recent_transactions": recent_transactions
            }
            
        except Exception as e:
            print(f"Error getting stock status: {e}")
            return {"error": f"Lỗi khi tra cứu thông tin: {str(e)}"}
    
    def _get_stock_status_text(self, available: int, min_stock: int) -> str:
        """Get stock status description"""
        if available <= 0:
            return "Hết hàng"
        elif available <= min_stock:
            return "Sắp hết hàng"
        elif available <= min_stock * 2:
            return "Tồn kho thấp"
        else:
            return "Tồn kho đủ"
```

### 10.2 Chat API Endpoint (app/api/chat.py)
```python
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
```

### 10.3 Inventory API (app/api/inventory.py)
```python
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from app.services.search_service import SearchService

router = APIRouter()
search_service = SearchService()

@router.get("/products")
async def get_products(
    query: Optional[str] = Query(None, description="Search query"),
    category: Optional[str] = Query(None, description="Product category"),
    limit: int = Query(50, description="Maximum number of results")
):
    """Get products with optional search and filtering"""
    try:
        if query:
            products = await search_service.search_products(query, limit)
        else:
            # Get all products (implement pagination if needed)
            products = await search_service.search_products("", limit)
        
        return {
            "products": products,
            "total": len(products),
            "query": query,
            "category": category
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/products/{product_code}/status")
async def get_product_status(product_code: str):
    """Get detailed status of a specific product"""
    try:
        status = await search_service.get_stock_status(product_code)
        
        if "error" in status:
            raise HTTPException(status_code=404, detail=status["error"])
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/locations")
async def get_locations(
    zone: Optional[str] = Query(None, description="Warehouse zone"),
    available_only: bool = Query(False, description="Show only available locations")
):
    """Get warehouse locations"""
    try:
        query = zone if zone else ""
        locations = await search_service.search_locations(query, 100)
        
        if available_only:
            locations = [loc for loc in locations if not loc.get("current_product")]
        
        return {
            "locations": locations,
            "total": len(locations),
            "zone": zone,
            "available_only": available_only
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/transactions")
async def get_transactions(
    product_code: Optional[str] = Query(None, description="Product code"),
    limit: int = Query(20, description="Maximum number of results")
):
    """Get recent inventory transactions"""
    try:
        transactions = await search_service.get_recent_transactions(product_code, limit)
        
        return {
            "transactions": transactions,
            "total": len(transactions),
            "product_code": product_code
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 10.4 Enhanced Flet Frontend với Voice Input
# Enhanced main.py with voice and advanced features
import flet as ft
import asyncio
import aiohttp
import speech_recognition as sr
import pyttsx3
from typing import Dict, List
import json
import threading

class AdvancedWarehouseChatbot:
    def __init__(self):
        self.api_base_url = "http://localhost:8000/api/v1"
        self.chat_history = []
        self.voice_enabled = True
        self.recognizer = sr.Recognizer()
        self.tts_engine = pyttsx3.init()
        self.is_listening = False

        self.page = None
        
    async def main(self, page: ft.Page):
        page.title = "Trợ lý kho ảo - Nâng cao"
        page.theme_mode = ft.ThemeMode.LIGHT
        page.padding = 20
        page.window_width = 1200
        page.window_height = 800

        self.page = page
        
        # Initialize TTS
        self.tts_engine.setProperty('rate', 150)
        voices = self.tts_engine.getProperty('voices')
        if voices:
            self.tts_engine.setProperty('voice', voices[0].id)
        
        # Create layout components
        await self.create_layout(page)
        
        # Load initial suggestions
        await self.load_suggestions()
        
        # Welcome message
        await self.add_message(
            "🤖 Chào bạn! Tôi là trợ lý kho ảo thông minh.\n"
            "✨ Tính năng mới: Hỗ trợ giọng nói, tìm kiếm thông minh, gợi ý tự động\n"
            "📝 Bạn có thể hỏi về mã hàng, quy trình, vị trí kho...", 
            "bot"
        )
    
    async def create_layout(self, page: ft.Page):
        """Create advanced UI layout"""
        
        # Header
        header = ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.WAREHOUSE, size=40, color=ft.Colors.BLUE),
                ft.Text("Trợ lý kho ảo", size=28, weight=ft.FontWeight.BOLD),
                ft.Container(expand=True),
                ft.IconButton(
                    icon=ft.Icons.MIC if not self.is_listening else ft.Icons.MIC_OFF,
                    tooltip="Bật/tắt microphone",
                    on_click=self.toggle_voice,
                    bgcolor=ft.Colors.GREEN_100 if self.voice_enabled else ft.Colors.RED_100
                ),
                ft.IconButton(
                    icon=ft.Icons.VOLUME_UP,
                    tooltip="Đọc câu trả lời",
                    on_click=self.toggle_tts
                )
            ]),
            padding=20,
            bgcolor=ft.Colors.BLUE_50,
            border_radius=10
        )
        
        # Main content area
        content_area = ft.Row([
            # Left sidebar - Quick actions
            ft.Container(
                content=await self.create_sidebar(),
                width=300,
                bgcolor=ft.Colors.GREY_50,
                padding=20,
                border_radius=10
            ),
            
            # Center - Chat area
            ft.Container(
                content=await self.create_chat_area(),
                expand=True,
                bgcolor=ft.Colors.WHITE,
                padding=20,
                border_radius=10,
                margin=ft.margin.only(left=10, right=10)
            ),
            
            # Right sidebar - Context info
            ft.Container(
                content=await self.create_context_panel(),
                width=300,
                bgcolor=ft.Colors.GREY_50,
                padding=20,
                border_radius=10
            )
        ])
        
        # Add to page
        page.add(
            ft.Column([
                header,
                ft.Container(height=10),
                content_area
            ])
        )
    
    async def create_sidebar(self):
        """Create left sidebar with quick actions"""
        
        # Category buttons
        categories = [
            {"name": "Tìm hàng", "icon": "🔍", "query": "tìm kiếm sản phẩm"},
            {"name": "Quy trình", "icon": "📋", "query": "quy trình làm việc"},
            {"name": "Vị trí kho", "icon": "📍", "query": "vị trí lưu kho"},
            {"name": "Tồn kho", "icon": "📊", "query": "báo cáo tồn kho"},
            {"name": "Nhập/Xuất", "icon": "📦", "query": "lịch sử giao dịch"},
            {"name": "Hướng dẫn", "icon": "❓", "query": "hướng dẫn sử dụng"}
        ]
        
        category_buttons = []
        for cat in categories:
            btn = ft.ElevatedButton(
                content=ft.Row([
                    ft.Text(cat["icon"], size=20),
                    ft.Text(cat["name"], expand=True)
                ]),
                width=250,
                height=50,
                on_click=lambda e, query=cat["query"]: asyncio.create_task(
                    self.send_message_programmatically(query)
                )
            )
            category_buttons.append(btn)
        
        # Recent queries
        self.recent_queries_list = ft.Column([
            ft.Text("📝 Câu hỏi gần đây", size=16, weight=ft.FontWeight.W_500)
        ])
        
        return ft.Column([
            ft.Text("🚀 Thao tác nhanh", size=18, weight=ft.FontWeight.BOLD),
            ft.Divider(),
            *category_buttons,
            ft.Divider(),
            self.recent_queries_list
        ])
    
    async def create_chat_area(self):
        """Create main chat area"""
        
        # Chat messages container
        self.chat_container = ft.Column(
            controls=[],
            scroll=ft.ScrollMode.AUTO,
            auto_scroll=True,
            height=400,
            expand=True
        )
        
        # Input area
        self.input_field = ft.TextField(
            hint_text="Nhập câu hỏi hoặc nhấn mic để nói...",
            expand=True,
            multiline=True,
            max_lines=3,
            on_submit=self.send_message,
            border_radius=10
        )
        
        # Action buttons
        input_actions = ft.Row([
            ft.IconButton(
                icon=ft.Icons.MIC,
                tooltip="Nói",
                on_click=self.start_voice_input,
                bgcolor=ft.Colors.GREEN_100
            ),
            ft.IconButton(
                icon=ft.Icons.PHOTO_CAMERA,
                tooltip="Chụp ảnh mã vạch",
                on_click=self.scan_barcode
            ),
            ft.ElevatedButton(
                "Gửi",
                icon=ft.Icons.SEND,
                on_click=self.send_message
            )
        ])
        
        return ft.Column([
            ft.Text("💬 Trò chuyện", size=18, weight=ft.FontWeight.BOLD),
            ft.Divider(),
            ft.Container(
                content=self.chat_container,
                border=ft.border.all(1, ft.Colors.GREY_300),
                border_radius=10,
                padding=10,
                bgcolor=ft.Colors.GREY_50
            ),
            ft.Container(height=10),
            self.input_field,
            input_actions
        ])
    
    async def create_context_panel(self):
        """Create right context panel"""
        
        self.context_panel = ft.Column([
            ft.Text("📊 Thông tin ngữ cảnh", size=18, weight=ft.FontWeight.BOLD),
            ft.Divider(),
            ft.Text("Sẽ hiển thị thông tin liên quan khi bạn đặt câu hỏi", 
                   color=ft.Colors.GREY_600)
        ])
        
        return self.context_panel
    
    async def send_message(self, e=None):
        """Enhanced send message with context updates"""
        if not self.input_field.value.strip():
            return
            
        user_message = self.input_field.value.strip()
        self.input_field.value = ""
        await self.input_field.update()
        
        # Add to recent queries
        await self.add_recent_query(user_message)
        
        # Add user message
        await self.add_message(user_message, "user")
        
        # Show typing indicator
        typing_msg = await self.add_message("🤔 Đang suy nghĩ...", "bot")
        
        try:
            # Call API
            response = await self.call_chat_api(user_message)
            
            # Remove typing indicator
            self.chat_container.controls.remove(typing_msg)
            
            # Add bot response
            await self.add_message(response["answer"], "bot")
            
            # Update context panel
            await self.update_context_panel(response.get("context", {}))
            
            # Add sources if available
            if response.get("sources"):
                sources_text = "📚 Nguồn tham khảo:\n" + "\n".join([
                    f"• {source.get('metadata', {}).get('title', 'Tài liệu')}"
                    for source in response["sources"][:3]
                ])
                await self.add_message(sources_text, "info")
            
            # Add suggestions if available
            if response.get("suggestions"):
                suggestions_text = "💡 Gợi ý câu hỏi tiếp theo:\n" + "\n".join([
                    f"• {suggestion}" for suggestion in response["suggestions"][:3]
                ])
                await self.add_message(suggestions_text, "suggestion")
            
            # Text-to-speech if enabled
            if self.voice_enabled:
                await self.speak_text(response["answer"])
                
        except Exception as error:
            # Remove typing indicator
            if typing_msg in self.chat_container.controls:
                self.chat_container.controls.remove(typing_msg)
            await self.add_message(f"❌ Lỗi: {str(error)}", "error")
    
    async def add_message(self, message: str, sender: str):
        """Enhanced message display with better formatting"""
        Colors = {
            "user": ft.Colors.BLUE_50,
            "bot": ft.Colors.GREEN_50,
            "info": ft.Colors.ORANGE_50,
            "error": ft.Colors.RED_50,
            "suggestion": ft.Colors.PURPLE_50
        }
        
        Icons = {
            "user": "👤",
            "bot": "🤖",
            "info": "ℹ️",
            "error": "❌",
            "suggestion": "💡"
        }
        
        # Create interactive suggestions
        if sender == "suggestion":
            suggestion_buttons = []
            suggestions = message.split("\n")[1:]  # Skip header
            for suggestion in suggestions:
                if suggestion.strip().startswith("•"):
                    clean_suggestion = suggestion.strip()[1:].strip()
                    btn = ft.TextButton(
                        clean_suggestion,
                        on_click=lambda e, text=clean_suggestion: asyncio.create_task(
                            self.send_message_programmatically(text)
                        )
                    )
                    suggestion_buttons.append(btn)
            
            message_content = ft.Column([
                ft.Text("💡 Gợi ý câu hỏi tiếp theo:", weight=ft.FontWeight.W_500),
                ft.Column(suggestion_buttons)
            ])
        else:
            message_content = ft.Row([
                ft.Text(Icons.get(sender, ""), size=20),
                ft.Text(message, expand=True, selectable=True)
            ])
        
        message_container = ft.Container(
            content=message_content,
            bgcolor=Colors.get(sender, ft.Colors.GREY_50),
            padding=15,
            border_radius=10,
            margin=ft.margin.only(bottom=10),
            animate=ft.Animation(300, ft.AnimationCurve.EASE_OUT)
        )
        
        self.chat_container.controls.append(message_container)
        self.chat_container.update()
        
        return message_container
    
    async def start_voice_input(self, e=None):
        """Start voice recognition"""
        if self.is_listening:
            return
        
        self.is_listening = True
        
        # Update UI
        await self.add_message("🎤 Đang nghe... Nói câu hỏi của bạn", "info")
        
        def voice_thread():
            try:
                with sr.Microphone() as source:
                    # Adjust for ambient noise
                    self.recognizer.adjust_for_ambient_noise(source, duration=1)
                    
                    # Listen for audio
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                    
                    # Recognize speech using Google Speech Recognition
                    text = self.recognizer.recognize_google(audio, language='vi-VN')
                    
                    # Update input field
                    asyncio.create_task(self.handle_voice_result(text))
                    
            except sr.UnknownValueError:
                asyncio.create_task(self.handle_voice_error("Không thể nhận diện giọng nói"))
            except sr.RequestError as e:
                asyncio.create_task(self.handle_voice_error(f"Lỗi dịch vụ: {e}"))
            except sr.WaitTimeoutError:
                asyncio.create_task(self.handle_voice_error("Hết thời gian chờ"))
            finally:
                self.is_listening = False
        
        # Run in separate thread
        threading.Thread(target=voice_thread, daemon=True).start()
    
    async def handle_voice_result(self, text: str):
        """Handle successful voice recognition"""
        await self.add_message(f"🎤 Đã nghe: {text}", "info")
        self.input_field.value = text
        self.input_field.update()
        await self.send_message()
    
    async def handle_voice_error(self, error_msg: str):
        """Handle voice recognition error"""
        await self.add_message(f"🎤 {error_msg}", "error")
    
    async def speak_text(self, text: str):
        """Text-to-speech"""
        def tts_thread():
            try:
                # Clean text for TTS
                clean_text = text.replace("•", "").replace("📚", "").replace("💡", "")
                self.tts_engine.say(clean_text[:200])  # Limit length
                self.tts_engine.runAndWait()
            except:
                pass
        
        threading.Thread(target=tts_thread, daemon=True).start()
    
    async def scan_barcode(self, e=None):
        """Simulate barcode scanning (would integrate with camera)"""
        await self.add_message("📷 Tính năng quét mã vạch đang được phát triển...", "info")
    
    async def toggle_voice(self, e=None):
        """Toggle voice input"""
        self.voice_enabled = not self.voice_enabled
        # Update button appearance would go here
    
    async def toggle_tts(self, e=None):
        """Toggle text-to-speech"""
        # Implementation for TTS toggle
        pass
    
    async def update_context_panel(self, context: dict):
        """Update right context panel with relevant information"""
        if not context:
            return
        
        context_items = []
        
        # Products context
        if context.get("products"):
            context_items.append(ft.Text("📦 Sản phẩm liên quan:", weight=ft.FontWeight.W_500))
            for product in context["products"][:3]:
                context_items.append(
                    ft.Container(
                        content=ft.Column([
                            ft.Text(f"• {product['name']}", size=12),
                            ft.Text(f"  Mã: {product['code']}", size=10, color=ft.Colors.GREY_600),
                            ft.Text(f"  Tồn: {product.get('current_stock', 0)}", size=10, color=ft.Colors.GREY_600)
                        ]),
                        padding=5,
                        bgcolor=ft.Colors.BLUE_50,
                        border_radius=5,
                        margin=ft.margin.only(bottom=5)
                    )
                )
        
        # Locations context
        if context.get("locations"):
            context_items.append(ft.Text("📍 Vị trí liên quan:", weight=ft.FontWeight.W_500))
            for location in context["locations"][:3]:
                context_items.append(
                    ft.Container(
                        content=ft.Text(f"• {location['location_code']}", size=12),
                        padding=5,
                        bgcolor=ft.Colors.GREEN_50,
                        border_radius=5,
                        margin=ft.margin.only(bottom=5)
                    )
                )
        
        # Procedures context
        if context.get("procedures"):
            context_items.append(ft.Text("📋 Quy trình liên quan:", weight=ft.FontWeight.W_500))
            for procedure in context["procedures"][:2]:
                context_items.append(
                    ft.Container(
                        content=ft.Text(f"• {procedure['title']}", size=12),
                        padding=5,
                        bgcolor=ft.Colors.ORANGE_50,
                        border_radius=5,
                        margin=ft.margin.only(bottom=5)
                    )
                )
        
        # Update context panel
        self.context_panel.controls = [
            ft.Text("📊 Thông tin ngữ cảnh", size=18, weight=ft.FontWeight.BOLD),
            ft.Divider(),
            *context_items
        ]
        
        self.context_panel.update()
    
    async def add_recent_query(self, query: str):
        """Add query to recent queries list"""
        if len(self.chat_history) >= 5:
            self.chat_history.pop(0)
        
        self.chat_history.append(query)
        
        # Update recent queries UI
        recent_items = [ft.Text("📝 Câu hỏi gần đây", size=16, weight=ft.FontWeight.W_500)]
        
        for q in reversed(self.chat_history[-3:]):  # Show last 3
            recent_items.append(
                ft.TextButton(
                    content=ft.Text(q[:30] + "..." if len(q) > 30 else q, size=10),
                    on_click=lambda e, query=q: asyncio.create_task(
                        self.send_message_programmatically(query)
                    )
                )
            )
        
        self.recent_queries_list.controls = recent_items
        self.recent_queries_list.update()
    
    async def load_suggestions(self):
        """Load initial suggestions from API"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_base_url}/chat/suggestions") as response:
                    if response.status == 200:
                        data = await response.json()
                        # Process suggestions if needed
        except:
            pass  # Fail silently
    
    async def call_chat_api(self, message: str) -> Dict:
        """Enhanced API call with better error handling"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.api_base_url}/chat/query",
                    json={
                        "query": message,
                        "user_id": "flet_user",
                        "context": {"interface": "flet", "voice_enabled": self.voice_enabled}
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        raise Exception(f"API Error {response.status}: {error_text}")
            except asyncio.TimeoutError:
                raise Exception("Request timeout - please try again")
            except aiohttp.ClientError as e:
                raise Exception(f"Connection error: {str(e)}")
    
    async def send_message_programmatically(self, message: str):
        """Send message programmatically from buttons/suggestions"""
        self.input_field.value = message
        await self.input_field.update()
        await self.send_message()

# Additional utility functions and configurations

class VoiceCommands:
    """Voice command processing for hands-free operation"""
    
    COMMANDS = {
        "tìm": "search",
        "tìm kiếm": "search", 
        "mở": "open",
        "hiển thị": "show",
        "báo cáo": "report",
        "quy trình": "procedure",
        "hướng dẫn": "guide"
    }
    
    @staticmethod
    def process_voice_command(text: str) -> str:
        """Process voice command and convert to proper query"""
        text_lower = text.lower()
        
        # Check for specific commands
        for cmd, action in VoiceCommands.COMMANDS.items():
            if cmd in text_lower:
                return f"{action}: {text}"
        
        return text

class DataExport:
    """Export chat history and data"""
    
    @staticmethod
    async def export_chat_history(chat_history: List[dict], format: str = "json"):
        """Export chat history to file"""
        import json
        from datetime import datetime
        
        export_data = {
            "exported_at": datetime.now().isoformat(),
            "chat_count": len(chat_history),
            "messages": chat_history
        }
        
        if format == "json":
            return json.dumps(export_data, ensure_ascii=False, indent=2)
        elif format == "txt":
            lines = [f"Chat History - Exported: {export_data['exported_at']}\n"]
            for msg in chat_history:
                lines.append(f"[{msg.get('timestamp', 'N/A')}] {msg.get('sender', 'Unknown')}: {msg.get('message', '')}\n")
            return "\n".join(lines)

# Configuration and settings
class AppConfig:
    """Application configuration"""
    
    # API Settings
    API_BASE_URL = "http://localhost:8000/api/v1"
    API_TIMEOUT = 30
    
    # Voice Settings
    VOICE_LANGUAGE = "vi-VN"
    TTS_RATE = 150
    VOICE_TIMEOUT = 10
    
    # UI Settings
    WINDOW_WIDTH = 1200
    WINDOW_HEIGHT = 800
    CHAT_HISTORY_LIMIT = 100
    RECENT_QUERIES_LIMIT = 5
    
    # Colors
    Colors = {
        "primary": ft.Colors.BLUE,
        "secondary": ft.Colors.GREEN,
        "accent": ft.Colors.ORANGE,
        "error": ft.Colors.RED,
        "success": ft.Colors.GREEN,
        "warning": ft.Colors.ORANGE,
        "info": ft.Colors.BLUE
    }

# Main application entry point
def main():
    """Enhanced main function with configuration"""
    import os
    import sys
    import logging
    os.makedirs('logs', exist_ok=True)
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/flet_app.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger("WarehouseChatbot")
    logger.info("Starting Warehouse Chatbot Flet Application")
    
    try:
        # Create and run chatbot
        chatbot = AdvancedWarehouseChatbot()
        
        # Configure Flet app
        ft.app(
            target=chatbot.main,
            port=8080,
            view=ft.AppView.WEB_BROWSER,
            assets_dir="assets",  # For storing images, sounds, etc.
            upload_dir="uploads"  # For file uploads if needed
        )
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()