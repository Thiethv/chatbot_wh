"""
Database configuration and connection management
"""

from sqlalchemy import create_engine, MetaData, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import logging
from typing import Generator
from contextlib import contextmanager

from .config import settings

logger = logging.getLogger(__name__)

# Database engine configuration
if settings.DATABASE_URL.startswith("sqlite"):
    # SQLite specific configuration
    engine = create_engine(
        settings.DATABASE_URL,
        connect_args={
            "check_same_thread": False,
            "timeout": 20
        },
        poolclass=StaticPool,
        echo=settings.DATABASE_ECHO
    )
    
    # Enable foreign key constraints for SQLite
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()
        
else:
    # PostgreSQL/MySQL configuration
    engine = create_engine(
        settings.DATABASE_URL,
        pool_pre_ping=True,
        pool_recycle=300,
        echo=settings.DATABASE_ECHO
    )

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base class for models
Base = declarative_base()

# Metadata for database operations
metadata = MetaData()

class DatabaseManager:
    """Database management utilities"""
    
    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal
        
    def create_tables(self):
        """Create all database tables"""
        try:
            Base.metadata.create_all(bind=engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            raise
    
    def drop_tables(self):
        """Drop all database tables"""
        try:
            Base.metadata.drop_all(bind=engine)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Error dropping database tables: {e}")
            raise
    
    def get_table_info(self):
        """Get information about database tables"""
        try:
            from sqlalchemy import inspect
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            
            table_info = {}
            for table in tables:
                columns = inspector.get_columns(table)
                table_info[table] = {
                    'columns': [col['name'] for col in columns],
                    'column_count': len(columns)
                }
            
            return table_info
        except Exception as e:
            logger.error(f"Error getting table info: {e}")
            return {}
    
    def health_check(self) -> bool:
        """Check database health"""
        try:
            with engine.connect() as connection:
                connection.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

# Global database manager instance
db_manager = DatabaseManager()

# Dependency to get database session
def get_database() -> Generator[Session, None, None]:
    """
    Database session dependency for FastAPI
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()

@contextmanager
def get_db_session():
    """
    Context manager for database sessions
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        logger.error(f"Database transaction error: {e}")
        db.rollback()
        raise
    finally:
        db.close()

# Database initialization
def init_database():
    """Initialize database with sample data"""
    logger.info("Initializing database...")
    
    try:
        # Create tables
        db_manager.create_tables()
        
        # Insert sample data
        with get_db_session() as db:
            # Check if data already exists
            from .models.product import Product
            if db.query(Product).count() == 0:
                create_sample_data(db)
        
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

def create_sample_data(db: Session):
    """Create sample data for testing"""
    from .models.product import Product
    from .models.inventory import Inventory, InventoryTransaction
    from .models.procedure import Procedure
    from datetime import datetime
    
    # Sample products
    sample_products = [
        {
            "code": "SP001",
            "name": "Laptop Dell Inspiron 15",
            "category": "Electronics",
            "description": "Laptop Dell Inspiron 15 inch, RAM 8GB, SSD 256GB",
            "unit": "chiếc",
            "price": 15000000,
            "barcode": "1234567890123"
        },
        {
            "code": "SP002", 
            "name": "Chuột không dây Logitech",
            "category": "Electronics",
            "description": "Chuột không dây Logitech MX Master 3",
            "unit": "chiếc",
            "price": 2000000,
            "barcode": "1234567890124"
        },
        {
            "code": "SP003",
            "name": "Bàn phím cơ Keychron K2",
            "category": "Electronics", 
            "description": "Bàn phím cơ Keychron K2 wireless",
            "unit": "chiếc",
            "price": 3000000,
            "barcode": "1234567890125"
        }
    ]
    
    # Create products
    products = []
    for product_data in sample_products:
        product = Product(**product_data)
        db.add(product)
        products.append(product)
    
    db.flush()  # Get IDs
    
    # Sample inventory
    sample_inventory = [
        {
            "product_id": products[0].id,
            "location_code": "A01-01-01",
            "current_stock": 50,
            "min_stock": 10,
            "max_stock": 100
        },
        {
            "product_id": products[1].id,
            "location_code": "A01-01-02", 
            "current_stock": 100,
            "min_stock": 20,
            "max_stock": 200
        },
        {
            "product_id": products[2].id,
            "location_code": "A01-01-03",
            "current_stock": 75,
            "min_stock": 15,
            "max_stock": 150
        }
    ]
    
    # Create inventory records
    for inventory_data in sample_inventory:
        inventory = Inventory(**inventory_data)
        db.add(inventory)
    
    # Sample procedures
    sample_procedures = [
        {
            "title": "Quy trình nhập kho",
            "description": "Hướng dẫn quy trình nhập hàng vào kho",
            "category": "Nhập kho",
            "content": """
            1. Kiểm tra phiếu nhập hàng
            2. Kiểm tra chất lượng sản phẩm
            3. Cập nhật số lượng vào hệ thống
            4. Sắp xếp hàng vào vị trí quy định
            5. In phiếu nhập kho hoàn thành
            """,
            "steps": [
                "Kiểm tra phiếu nhập hàng",
                "Kiểm tra chất lượng sản phẩm", 
                "Cập nhật số lượng vào hệ thống",
                "Sắp xếp hàng vào vị trí quy định",
                "In phiếu nhập kho hoàn thành"
            ]
        },
        {
            "title": "Quy trình xuất kho",
            "description": "Hướng dẫn quy trình xuất hàng ra khỏi kho",
            "category": "Xuất kho",
            "content": """
            1. Kiểm tra phiếu xuất hàng
            2. Tìm vị trí hàng trong kho
            3. Kiểm tra số lượng và chất lượng
            4. Đóng gói hàng hóa
            5. Cập nhật tồn kho trong hệ thống
            6. Giao hàng cho khách hàng
            """,
            "steps": [
                "Kiểm tra phiếu xuất hàng",
                "Tìm vị trí hàng trong kho",
                "Kiểm tra số lượng và chất lượng",
                "Đóng gói hàng hóa", 
                "Cập nhật tồn kho trong hệ thống",
                "Giao hàng cho khách hàng"
            ]
        }
    ]
    
    # Create procedures
    for procedure_data in sample_procedures:
        procedure = Procedure(**procedure_data)
        db.add(procedure)
    
    db.commit()
    logger.info("Sample data created successfully")

# Health check function
async def check_database_health():
    """Async health check for database"""
    return db_manager.health_check()

# Export main components
__all__ = [
    "engine",
    "SessionLocal", 
    "Base",
    "get_database",
    "get_db_session",
    "db_manager",
    "init_database",
    "check_database_health"
]