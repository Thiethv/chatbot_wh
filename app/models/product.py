"""
Product model for warehouse management
"""

from sqlalchemy import Column, Integer, String, Text, Float, DateTime, Boolean, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional, List, Dict, Any

from ..database import Base

class Product(Base):
    """Product model"""
    
    __tablename__ = "products"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    
    # Basic product information
    code = Column(String(50), unique=True, index=True, nullable=False)
    name = Column(String(200), nullable=False, index=True)
    category = Column(String(100), index=True)
    description = Column(Text)
    
    # Product specifications
    unit = Column(String(20), default="pcs")  # đơn vị tính
    weight = Column(Float)  # kg
    dimensions = Column(JSON)  # {"length": 10, "width": 5, "height": 3}
    
    # Pricing
    cost_price = Column(Float)  # giá nhập
    selling_price = Column(Float)  # giá bán
    price = Column(Float)  # giá hiện tại (tương thích với code cũ)
    
    # Identification
    barcode = Column(String(50), index=True)
    sku = Column(String(50), index=True)
    qr_code = Column(String(100))
    
    # Product attributes
    brand = Column(String(100))
    model = Column(String(100))
    color = Column(String(50))
    size = Column(String(50))
    
    # Status and flags
    is_active = Column(Boolean, default=True)
    is_serialized = Column(Boolean, default=False)  # có theo dõi serial number
    is_batch_tracked = Column(Boolean, default=False)  # có theo dõi lô hàng
    is_expirable = Column(Boolean, default=False)  # có hạn sử dụng
    
    # Additional information
    manufacturer = Column(String(100))
    supplier = Column(String(100))
    country_of_origin = Column(String(50))
    
    # Metadata
    tags = Column(JSON)  # ["electronics", "laptop", "dell"]
    custom_fields = Column(JSON)  # custom attributes
    
    # Audit fields
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    created_by = Column(String(50))
    updated_by = Column(String(50))
    
    # Relationships
    inventory_records = relationship("Inventory", back_populates="product")
    transactions = relationship("InventoryTransaction", back_populates="product")
    
    def __repr__(self):
        return f"<Product(code='{self.code}', name='{self.name}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert product to dictionary"""
        return {
            "id": self.id,
            "code": self.code,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "unit": self.unit,
            "weight": self.weight,
            "dimensions": self.dimensions,
            "cost_price": self.cost_price,
            "selling_price": self.selling_price,
            "price": self.price or self.selling_price,
            "barcode": self.barcode,
            "sku": self.sku,
            "qr_code": self.qr_code,
            "brand": self.brand,
            "model": self.model,
            "color": self.color,
            "size": self.size,
            "is_active": self.is_active,
            "is_serialized": self.is_serialized,
            "is_batch_tracked": self.is_batch_tracked,
            "is_expirable": self.is_expirable,
            "manufacturer": self.manufacturer,
            "supplier": self.supplier,
            "country_of_origin": self.country_of_origin,
            "tags": self.tags or [],
            "custom_fields": self.custom_fields or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "created_by": self.created_by,
            "updated_by": self.updated_by
        }
    
    def get_current_stock(self, location_code: Optional[str] = None) -> int:
        """Get current stock for this product"""
        total_stock = 0
        for inventory in self.inventory_records:
            if location_code is None or inventory.location_code == location_code:
                total_stock += inventory.current_stock or 0
        return total_stock
    
    def get_total_value(self, location_code: Optional[str] = None) -> float:
        """Get total value of current stock"""
        current_stock = self.get_current_stock(location_code)
        price = self.price or self.cost_price or 0
        return current_stock * price
    
    def is_low_stock(self, location_code: Optional[str] = None) -> bool:
        """Check if product is low in stock"""
        for inventory in self.inventory_records:
            if location_code is None or inventory.location_code == location_code:
                if inventory.current_stock <= inventory.min_stock:
                    return True
        return False
    
    def get_stock_status(self, location_code: Optional[str] = None) -> str:
        """Get stock status description"""
        current_stock = self.get_current_stock(location_code)
        
        if current_stock <= 0:
            return "Hết hàng"
        elif self.is_low_stock(location_code):
            return "Sắp hết hàng"
        else:
            return "Còn hàng"
    
    def search_text(self) -> str:
        """Get searchable text for this product"""
        searchable_fields = [
            self.code or "",
            self.name or "",
            self.category or "",
            self.description or "",
            self.brand or "",
            self.model or "",
            self.barcode or "",
            self.sku or "",
            self.supplier or ""
        ]
        
        # Add tags if available
        if self.tags:
            searchable_fields.extend(self.tags)
        
        return " ".join(searchable_fields).lower()
    
    @classmethod
    def search_by_text(cls, db, query: str, limit: int = 10):
        """Search products by text query"""
        from sqlalchemy import or_, func
        
        # Split query into words
        words = query.lower().split()
        
        # Build search conditions
        conditions = []
        for word in words:
            word_pattern = f"%{word}%"
            conditions.append(
                or_(
                    func.lower(cls.code).like(word_pattern),
                    func.lower(cls.name).like(word_pattern),
                    func.lower(cls.category).like(word_pattern),
                    func.lower(cls.description).like(word_pattern),
                    func.lower(cls.brand).like(word_pattern),
                    func.lower(cls.model).like(word_pattern),
                    func.lower(cls.barcode).like(word_pattern),
                    func.lower(cls.sku).like(word_pattern)
                )
            )
        
        # Execute search
        query_obj = db.query(cls)
        for condition in conditions:
            query_obj = query_obj.filter(condition)
        
        return query_obj.filter(cls.is_active == True).limit(limit).all()
    
    @classmethod
    def get_by_code(cls, db, code: str):
        """Get product by code"""
        return db.query(cls).filter(
            cls.code == code,
            cls.is_active == True
        ).first()
    
    @classmethod
    def get_by_barcode(cls, db, barcode: str):
        """Get product by barcode"""
        return db.query(cls).filter(
            cls.barcode == barcode,
            cls.is_active == True
        ).first()
    
    @classmethod
    def get_low_stock_products(cls, db, limit: int = 50):
        """Get products with low stock"""
        from .inventory import Inventory
        from sqlalchemy import and_
        
        return db.query(cls).join(Inventory).filter(
            and_(
                cls.is_active == True,
                Inventory.current_stock <= Inventory.min_stock
            )
        ).limit(limit).all()
    
    @classmethod
    def get_by_category(cls, db, category: str, limit: int = 50):
        """Get products by category"""
        return db.query(cls).filter(
            cls.category == category,
            cls.is_active == True
        ).limit(limit).all()
    
    def update_from_dict(self, data: Dict[str, Any]):
        """Update product from dictionary"""
        updatable_fields = [
            'name', 'category', 'description', 'unit', 'weight', 'dimensions',
            'cost_price', 'selling_price', 'price', 'barcode', 'sku', 'qr_code',
            'brand', 'model', 'color', 'size', 'is_active', 'is_serialized',
            'is_batch_tracked', 'is_expirable', 'manufacturer', 'supplier',
            'country_of_origin', 'tags', 'custom_fields', 'updated_by'
        ]
        
        for field in updatable_fields:
            if field in data:
                setattr(self, field, data[field])
        
        self.updated_at = datetime.utcnow()