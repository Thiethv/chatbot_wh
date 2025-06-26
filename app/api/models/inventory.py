"""
Inventory models for warehouse management
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, JSON, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional, Dict, Any, List
import enum

from ..database import Base

class TransactionType(enum.Enum):
    """Transaction types"""
    INBOUND = "inbound"      # nhập kho
    OUTBOUND = "outbound"    # xuất kho
    TRANSFER = "transfer"    # chuyển kho
    ADJUSTMENT = "adjustment" # điều chỉnh
    RETURN = "return"        # trả hàng
    DAMAGE = "damage"        # hỏng hóc
    LOSS = "loss"           # mất hàng

class Inventory(Base):
    """Inventory management model"""
    
    __tablename__ = "inventory"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    
    # Foreign keys
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False, index=True)
    
    # Location information
    location_code = Column(String(50), nullable=False, index=True)  # A01-01-01
    location_name = Column(String(100))  # Kệ A, Tầng 1, Ngăn 1
    zone = Column(String(50))  # Zone A, Zone B
    warehouse = Column(String(50), default="MAIN")  # Kho chính
    
    # Stock quantities
    current_stock = Column(Integer, default=0, nullable=False)
    available_stock = Column(Integer, default=0)  # Có thể bán
    reserved_stock = Column(Integer, default=0)   # Đã đặt hàng
    damaged_stock = Column(Integer, default=0)    # Hỏng
    
    # Stock thresholds
    min_stock = Column(Integer, default=0)        # Tồn kho tối thiểu
    max_stock = Column(Integer, default=1000)     # Tồn kho tối đa
    reorder_point = Column(Integer, default=10)  # Điểm đặt hàng lại
    reorder_quantity = Column(Integer, default=50) # Số lượng đặt hàng
    
    # Batch and serial tracking
    batch_number = Column(String(50))
    serial_numbers = Column(JSON)  # List of serial numbers
    expiry_date = Column(DateTime)
    manufacturing_date = Column(DateTime)
    
    # Cost information
    unit_cost = Column(Float, default=0)
    total_cost = Column(Float, default=0)
    last_cost = Column(Float)  # Giá nhập cuối cùng
    
    # Physical attributes
    physical_count = Column(Integer)  # Số lượng kiểm kê thực tế
    last_count_date = Column(DateTime)
    cycle_count_due = Column(DateTime)
    
    # Status and flags
    is_active = Column(Boolean, default=True)
    is_blocked = Column(Boolean, default=False)  # Bị khóa
    is_quarantined = Column(Boolean, default=False)  # Cách ly
    
    # Additional information
    notes = Column(Text)
    custom_fields = Column(JSON)
    
    # Audit fields
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    created_by = Column(String(50))
    updated_by = Column(String(50))
    
    # Relationships
    product = relationship("Product", back_populates="inventory_records")
    transactions = relationship("InventoryTransaction", back_populates="inventory")
    
    def __repr__(self):
        return f"<Inventory(product_id={self.product_id}, location='{self.location_code}', stock={self.current_stock})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert inventory to dictionary"""
        return {
            "id": self.id,
            "product_id": self.product_id,
            "location_code": self.location_code,
            "location_name": self.location_name,
            "zone": self.zone,
            "warehouse": self.warehouse,
            "current_stock": self.current_stock,
            "available_stock": self.available_stock,
            "reserved_stock": self.reserved_stock,
            "damaged_stock": self.damaged_stock,
            "min_stock": self.min_stock,
            "max_stock": self.max_stock,
            "reorder_point": self.reorder_point,
            "reorder_quantity": self.reorder_quantity,
            "batch_number": self.batch_number,
            "serial_numbers": self.serial_numbers or [],
            "expiry_date": self.expiry_date.isoformat() if self.expiry_date else None,
            "manufacturing_date": self.manufacturing_date.isoformat() if self.manufacturing_date else None,
            "unit_cost": self.unit_cost,
            "total_cost": self.total_cost,
            "last_cost": self.last_cost,
            "physical_count": self.physical_count,
            "last_count_date": self.last_count_date.isoformat() if self.last_count_date else None,
            "cycle_count_due": self.cycle_count_due.isoformat() if self.cycle_count_due else None,
            "is_active": self.is_active,
            "is_blocked": self.is_blocked,
            "is_quarantined": self.is_quarantined,
            "notes": self.notes,
            "custom_fields": self.custom_fields or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "created_by": self.created_by,
            "updated_by": self.updated_by
        }
    
    def get_stock_status(self) -> str:
        """Get stock status description"""
        if self.current_stock <= 0:
            return "out_of_stock"
        elif self.current_stock <= self.min_stock:
            return "low_stock"
        elif self.current_stock >= self.max_stock:
            return "overstock"
        else:
            return "in_stock"
    
    def get_stock_status_vietnamese(self) -> str:
        """Get stock status in Vietnamese"""
        status_map = {
            "out_of_stock": "Hết hàng",
            "low_stock": "Sắp hết hàng", 
            "overstock": "Quá tồn kho",
            "in_stock": "Còn hàng"
        }
        return status_map.get(self.get_stock_status(), "Không xác định")
    
    def update_stock(self, quantity: int, transaction_type: str, notes: str = None):
        """Update stock quantity"""
        old_stock = self.current_stock
        
        if transaction_type in ["inbound", "return"]:
            self.current_stock += quantity
        elif transaction_type in ["outbound", "damage", "loss"]:
            self.current_stock -= quantity
        elif transaction_type == "adjustment":
            self.current_stock = quantity
        
        # Update available stock
        self.available_stock = max(0, self.current_stock - self.reserved_stock)
        
        # Update total cost
        if self.unit_cost:
            self.total_cost = self.current_stock * self.unit_cost
        
        # Update timestamp
        self.updated_at = datetime.utcnow()
        
        return {
            "old_stock": old_stock,
            "new_stock": self.current_stock,
            "change": self.current_stock - old_stock
        }
    
    def reserve_stock(self, quantity: int) -> bool:
        """Reserve stock for orders"""
        if self.available_stock >= quantity:
            self.reserved_stock += quantity
            self.available_stock -= quantity
            return True
        return False
    
    def release_reserved_stock(self, quantity: int):
        """Release reserved stock"""
        released = min(quantity, self.reserved_stock)
        self.reserved_stock -= released
        self.available_stock += released
        return released
    
    @classmethod
    def get_by_product_and_location(cls, db, product_id: int, location_code: str):
        """Get inventory by product and location"""
        return db.query(cls).filter(
            cls.product_id == product_id,
            cls.location_code == location_code,
            cls.is_active == True
        ).first()
    
    @classmethod
    def get_low_stock_items(cls, db, limit: int = 50):
        """Get items with low stock"""
        return db.query(cls).filter(
            cls.current_stock <= cls.min_stock,
            cls.is_active == True
        ).limit(limit).all()
    
    @classmethod
    def get_by_location(cls, db, location_code: str):
        """Get all inventory items in a location"""
        return db.query(cls).filter(
            cls.location_code == location_code,
            cls.is_active == True
        ).all()

class InventoryTransaction(Base):
    """Inventory transaction history"""
    
    __tablename__ = "inventory_transactions"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    
    # Foreign keys
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False, index=True)
    inventory_id = Column(Integer, ForeignKey("inventory.id"), index=True)
    
    # Transaction details
    transaction_type = Column(Enum(TransactionType), nullable=False, index=True)
    reference_number = Column(String(50), index=True)  # Số chứng từ
    document_type = Column(String(50))  # Loại chứng từ
    
    # Quantities
    quantity = Column(Integer, nullable=False)
    unit_cost = Column(Float)
    total_cost = Column(Float)
    
    # Before/after stock levels
    stock_before = Column(Integer)
    stock_after = Column(Integer)
    
    # Location information
    location_from = Column(String(50))
    location_to = Column(String(50))
    
    # Additional information
    reason = Column(String(200))
    notes = Column(Text)
    batch_number = Column(String(50))
    serial_numbers = Column(JSON)
    
    # External references
    order_id = Column(String(50))
    supplier_id = Column(String(50))
    customer_id = Column(String(50))
    
    # Audit fields
    transaction_date = Column(DateTime(timezone=True), server_default=func.now())
    created_by = Column(String(50))
    approved_by = Column(String(50))
    
    # Status
    is_confirmed = Column(Boolean, default=False)
    is_cancelled = Column(Boolean, default=False)
    
    # Relationships
    product = relationship("Product", back_populates="transactions")
    inventory = relationship("Inventory", back_populates="transactions")
    
    def __repr__(self):
        return f"<InventoryTransaction(type='{self.transaction_type}', quantity={self.quantity})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert transaction to dictionary"""
        return {
            "id": self.id,
            "product_id": self.product_id,
            "inventory_id": self.inventory_id,
            "transaction_type": self.transaction_type.value if self.transaction_type else None,
            "reference_number": self.reference_number,
            "document_type": self.document_type,
            "quantity": self.quantity,
            "unit_cost": self.unit_cost,
            "total_cost": self.total_cost,
            "stock_before": self.stock_before,
            "stock_after": self.stock_after,
            "location_from": self.location_from,
            "location_to": self.location_to,
            "reason": self.reason,
            "notes": self.notes,
            "batch_number": self.batch_number,
            "serial_numbers": self.serial_numbers or [],
            "order_id": self.order_id,
            "supplier_id": self.supplier_id,
            "customer_id": self.customer_id,
            "transaction_date": self.transaction_date.isoformat() if self.transaction_date else None,
            "created_by": self.created_by,
            "approved_by": self.approved_by,
            "is_confirmed": self.is_confirmed,
            "is_cancelled": self.is_cancelled
        }
    
    @classmethod
    def create_transaction(cls, db, **kwargs):
        """Create a new inventory transaction"""
        transaction = cls(**kwargs)
        db.add(transaction)
        db.flush()
        return transaction
    
    @classmethod
    def get_by_product(cls, db, product_id: int, limit: int = 50):
        """Get transactions by product"""
        return db.query(cls).filter(
            cls.product_id == product_id
        ).order_by(cls.transaction_date.desc()).limit(limit).all()
    
    @classmethod
    def get_by_location(cls, db, location_code: str, limit: int = 50):
        """Get transactions by location"""
        return db.query(cls).filter(
            or_(cls.location_from == location_code, cls.location_to == location_code)
        ).order_by(cls.transaction_date.desc()).limit(limit).all()
    
    @classmethod
    def get_by_type(cls, db, transaction_type: TransactionType, limit: int = 50):
        """Get transactions by type"""
        return db.query(cls).filter(
            cls.transaction_type == transaction_type
        ).order_by(cls.transaction_date.desc()).limit(limit).all()
    
    @classmethod
    def get_recent_transactions(cls, db, limit: int = 100):
        """Get recent transactions"""
        return db.query(cls).order_by(
            cls.transaction_date.desc()
        ).limit(limit).all()

# Utility functions
def calculate_inventory_value(db, location_code: str = None) -> Dict[str, float]:
    """Calculate total inventory value"""
    from sqlalchemy import func, and_
    
    query = db.query(
        func.sum(Inventory.current_stock * Inventory.unit_cost).label('total_value'),
        func.count(Inventory.id).label('total_items'),
        func.sum(Inventory.current_stock).label('total_quantity')
    ).filter(Inventory.is_active == True)
    
    if location_code:
        query = query.filter(Inventory.location_code == location_code)
    
    result = query.first()
    
    return {
        "total_value": result.total_value or 0,
        "total_items": result.total_items or 0,
        "total_quantity": result.total_quantity or 0
    }

def get_inventory_summary(db) -> Dict[str, Any]:
    """Get inventory summary statistics"""
    from sqlalchemy import func, and_
    
    # Total inventory
    total_query = db.query(
        func.count(Inventory.id).label('total_items'),
        func.sum(Inventory.current_stock).label('total_quantity'),
        func.sum(Inventory.current_stock * Inventory.unit_cost).label('total_value')
    ).filter(Inventory.is_active == True)
    
    total_result = total_query.first()
    
    # Low stock items
    low_stock_count = db.query(Inventory).filter(
        and_(
            Inventory.current_stock <= Inventory.min_stock,
            Inventory.is_active == True
        )
    ).count()
    
    # Out of stock items
    out_of_stock_count = db.query(Inventory).filter(
        and_(
            Inventory.current_stock <= 0,
            Inventory.is_active == True
        )
    ).count()
    
    # Overstock items
    overstock_count = db.query(Inventory).filter(
        and_(
            Inventory.current_stock >= Inventory.max_stock,
            Inventory.is_active == True
        )
    ).count()
    
    return {
        "total_items": total_result.total_items or 0,
        "total_quantity": total_result.total_quantity or 0,
        "total_value": total_result.total_value or 0,
        "low_stock_count": low_stock_count,
        "out_of_stock_count": out_of_stock_count,
        "overstock_count": overstock_count
    }