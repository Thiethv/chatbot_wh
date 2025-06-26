# Kiến trúc chi tiết hệ thống Warehouse Chatbot

## 🏗️ Tổng quan kiến trúc

Hệ thống được thiết kế theo mô hình **Microservices** với các tầng rõ ràng:

### 1. **Frontend Layer (Tầng giao diện)**
- **Flet Desktop App**: Ứng dụng desktop Python với tính năng voice
- **Web Interface**: Giao diện web responsive
- **Mobile App**: Ứng dụng di động (tương lai)

### 2. **API Gateway Layer (Tầng API)**
- **FastAPI Server**: REST API với auto-documentation
- **CORS Middleware**: Hỗ trợ cross-origin requests
- **Authentication**: Xác thực người dùng (tương lai)

### 3. **Service Layer (Tầng dịch vụ)**
- **AI Service**: Xử lý trí tuệ nhân tạo
- **Search Service**: Tìm kiếm và truy vấn dữ liệu
- **Embedding Service**: Vector embeddings và semantic search
- **Voice Service**: Xử lý giọng nói

### 4. **Data Layer (Tầng dữ liệu)**
- **PostgreSQL**: Cơ sở dữ liệu chính
- **ChromaDB**: Vector database
- **File Storage**: Lưu trữ files

## 🔄 Luồng xử lý chính

### A. **Khởi tạo hệ thống**
```
1. FastAPI startup
2. Initialize EmbeddingService
3. Initialize AIService 
4. Load vector embeddings
5. Setup LLM connection
6. Ready to serve requests
```

### B. **Xử lý câu hỏi người dùng**

#### B.1. Input Processing
```
User Input → Voice/Text → Preprocessing → Vietnamese Tokenization
```

#### B.2. Context Gathering
```python
# Parallel search across data sources
async def _get_context(query):
    context = await asyncio.gather(
        search_products(query),      # PostgreSQL
        search_procedures(query),    # PostgreSQL  
        search_locations(query),     # PostgreSQL
        get_recent_transactions()    # PostgreSQL
    )
```

#### B.3. Semantic Search
```python
# Vector similarity search
embedding_service.get_retriever() → ChromaDB.similarity_search()
```

#### B.4. LLM Processing
```python
# Enhanced prompt with context
prompt = create_enhanced_prompt(query, context)
response = ConversationalRetrievalChain.acall(prompt)
```

#### B.5. Post-processing
```python
# Enhance response with metadata
response = {
    "answer": llm_response,
    "sources": retrieved_docs,
    "confidence": confidence_score,
    "suggestions": generated_suggestions,
    "context": gathered_context
}
```

## 📊 Các thành phần chi tiết

### 1. **AI Service Architecture**
```
AIService:
├── LLM Integration (Ollama)
│   ├── Llama2-7B-Chat
│   ├── Mistral-7B
│   └── Custom fine-tuned models
├── RAG Pipeline (LangChain)
│   ├── ConversationalRetrievalChain
│   ├── ConversationBufferWindowMemory
│   └── Custom PromptTemplate
├── Vietnamese Processing
│   ├── underthesea tokenizer
│   ├── Text normalization
│   └── Keyword extraction
└── Response Enhancement
    ├── Confidence scoring
    ├── Source attribution
    └── Suggestion generation
```

### 2. **Search Service Architecture**
```
SearchService:
├── Product Search
│   ├── Full-text search (PostgreSQL)
│   ├── Fuzzy matching (fuzzywuzzy)
│   └── Category filtering
├── Procedure Search
│   ├── Title/content search
│   ├── Tag-based search
│   └── Vector similarity
├── Location Search
│   ├── Code pattern matching
│   ├── Zone-based search
│   └── Availability filtering
└── Transaction History
    ├── Time-based queries
    ├── Product-specific history
    └── Audit trail
```

### 3. **Embedding Service Architecture**
```
EmbeddingService:
├── Vector Generation
│   ├── Sentence-BERT models
│   ├── Vietnamese embeddings
│   └── Domain-specific fine-tuning
├── ChromaDB Integration
│   ├── Collection management
│   ├── Metadata filtering
│   └── Similarity search
└── Document Processing
    ├── Text chunking
    ├── Metadata extraction
    └── Index optimization
```

## 🔧 Cấu hình Docker

### Docker Compose Services:
```yaml
Services:
├── postgres (Port: 5432)
│   ├── warehouse_db
│   ├── Products, Inventory, Procedures
│   └── Persistent volume
├── ollama (Port: 11434)  
│   ├── LLM model hosting
│   ├── Model management
│   └── GPU acceleration (optional)
├── chromadb (Port: 8001)
│   ├── Vector storage
│   ├── Embedding management
│   └── Persistence layer
├── api (Port: 8000)
│   ├── FastAPI application
│   ├── Service orchestration
│   └── Business logic
└── frontend (Optional)
    ├── Web interface
    ├── Asset serving
    └── Static files
```

## 📈 Monitoring & Analytics

### 1. **Logging System**
```python
ChatLogger:
├── Query logging
├── Response tracking  
├── Performance metrics
├── Error monitoring
└── Usage analytics
```

### 2. **Metrics Collection**
```python
Prometheus Metrics:
├── chatbot_queries_total
├── chatbot_response_time_seconds
├── chatbot_confidence_score
├── chatbot_errors_total
└── chatbot_active_users
```

## 🛡️ Security & Performance

### Security:
- API rate limiting
- Input validation and sanitization
- SQL injection prevention
- XSS protection
- CORS configuration

### Performance:
- Database connection pooling
- Vector search optimization
- Response caching
- Async/await throughout
- Memory management

## 🚀 Deployment Strategy

### Development:
```bash
docker-compose up -d
python scripts/prepare_training_data.py
uvicorn app.main:app --reload
```

### Production:
```bash
docker-compose -f docker-compose.prod.yml up -d  
nginx reverse proxy
SSL certificates
Health checks
Auto-scaling
```

## 📱 Frontend Features

### Flet Desktop App:
- **Voice Input/Output**: Speech recognition + TTS
- **Smart Context Panel**: Dynamic information display
- **Quick Actions**: Category-based shortcuts
- **Chat History**: Persistent conversation memory
- **Export Functionality**: Save conversations
- **Offline Mode**: Cache for basic operations

### Advanced Features:
- **Barcode Scanning**: Camera integration
- **Multi-language Support**: Vietnamese + English
- **Theme Customization**: Dark/Light modes
- **Keyboard Shortcuts**: Power user features
- **File Upload**: Document processing

## 🎯 Use Cases & Workflows

### 1. **Product Inquiry**
```
User: "Vải cotton trắng 90cm còn không?"
├── Voice → Text conversion
├── Vietnamese preprocessing  
├── Product search in PostgreSQL
├── Stock level checking
├── Location information
└── Contextual response with suggestions
```

### 2. **Procedure Guidance**
```
User: "Hướng dẫn kiểm tra hàng đầu ca"
├── Procedure search
├── Step-by-step retrieval
├── Related document linking
├── Video/image references
└── Interactive checklist
```

### 3. **Location Management**
```
User: "Vị trí A-12-03-02 đang chứa gì?"  
├── Location code parsing
├── Current inventory check
├── Capacity information
├── Recent transaction history
└── Optimization suggestions
```

## 📊 Data Flow Patterns

### Real-time Updates:
```
Database Change → Event Trigger → Cache Invalidation → UI Update
```

### Batch Processing:
```
Nightly Jobs → Data Aggregation → Report Generation → Notification
```

### Analytics Pipeline:
```
User Interactions → Log Aggregation → ML Analysis → Insights Dashboard
```

Hệ thống này tạo ra một giải pháp toàn diện cho quản lý kho hàng với AI, kết hợp tính năng voice, tìm kiếm thông minh và giao diện thân thiện.