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