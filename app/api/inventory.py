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