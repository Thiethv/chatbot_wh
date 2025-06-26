"""
Configuration settings for warehouse chatbot application
"""

import os
from pydantic_settings import BaseSettings
from typing import Optional, List
import logging

class Settings(BaseSettings):
    """Application configuration settings"""
    
    # Application settings
    APP_NAME: str = "Warehouse AI Chatbot"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = False
    
    # Database settings
    DATABASE_URL: str = "sqlite:///./warehouse.db"
    DATABASE_ECHO: bool = False
    
    # AI/LLM settings
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    OPENAI_MAX_TOKENS: int = 1000
    OPENAI_TEMPERATURE: float = 0.3
    
    # Alternative AI providers
    GEMINI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    
    # Vector database settings (for embedding search)
    VECTOR_DB_PATH: str = "./vector_db"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    
    # Vietnamese language processing
    VIETNAMESE_MODEL_PATH: str = "./models/vietnamese"
    USE_VIETNAMESE_SEGMENTATION: bool = True
    
    # Search settings
    MAX_SEARCH_RESULTS: int = 10
    SEARCH_SIMILARITY_THRESHOLD: float = 0.7
    
    # Chat settings
    MAX_CHAT_HISTORY: int = 50
    CONTEXT_WINDOW_SIZE: int = 10
    DEFAULT_SUGGESTIONS_COUNT: int = 3
    
    # File upload settings
    UPLOAD_DIR: str = "./uploads"
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: List[str] = [".pdf", ".docx", ".txt", ".csv", ".xlsx"]
    
    # Logging settings
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/app.log"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_ROTATION: str = "1 day"
    LOG_RETENTION: str = "30 days"
    
    # Redis settings (for caching and sessions)
    REDIS_URL: Optional[str] = None
    CACHE_TTL: int = 3600  # 1 hour
    
    # Security settings
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    CORS_ORIGINS: List[str] = ["*"]
    
    # API rate limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_PERIOD: int = 60  # seconds
    
    # Feature flags
    ENABLE_VOICE_PROCESSING: bool = True
    ENABLE_IMAGE_PROCESSING: bool = False
    ENABLE_BARCODE_SCANNING: bool = False
    ENABLE_EXPORT_FEATURES: bool = True
    ENABLE_ANALYTICS: bool = True
    
    # Monitoring and metrics
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    HEALTH_CHECK_INTERVAL: int = 30
    
    # External services
    WEBHOOK_URL: Optional[str] = None
    NOTIFICATION_ENABLED: bool = False
    EMAIL_SMTP_SERVER: Optional[str] = None
    EMAIL_SMTP_PORT: int = 587
    EMAIL_USERNAME: Optional[str] = None
    EMAIL_PASSWORD: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

# Global settings instance
settings = Settings()

# Logging configuration
def setup_logging():
    """Setup application logging"""
    
    # Create logs directory if not exists
    os.makedirs(os.path.dirname(settings.LOG_FILE), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper()),
        format=settings.LOG_FORMAT,
        handlers=[
            logging.FileHandler(settings.LOG_FILE, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # Set specific loggers
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)

# Validation functions
def validate_ai_config():
    """Validate AI configuration"""
    if not any([settings.OPENAI_API_KEY, settings.GEMINI_API_KEY, settings.ANTHROPIC_API_KEY]):
        logging.warning("No AI API keys configured. Some features may not work.")
        return False
    return True

def validate_database_config():
    """Validate database configuration"""
    if settings.DATABASE_URL.startswith("sqlite"):
        # Create directory for SQLite database
        db_path = settings.DATABASE_URL.replace("sqlite:///", "")
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    
    return True

def get_ai_provider():
    """Get the first available AI provider"""
    if settings.OPENAI_API_KEY:
        return "openai"
    elif settings.GEMINI_API_KEY:
        return "gemini"
    elif settings.ANTHROPIC_API_KEY:
        return "anthropic"
    else:
        raise ValueError("No AI provider configured")

# Initialize configuration on import
setup_logging()
validate_database_config()

logger = logging.getLogger(__name__)
logger.info(f"Configuration loaded: {settings.APP_NAME} v{settings.APP_VERSION}")

if not validate_ai_config():
    logger.warning("AI configuration incomplete - some features may be limited")