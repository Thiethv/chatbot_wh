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