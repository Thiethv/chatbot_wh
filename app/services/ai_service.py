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