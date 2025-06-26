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