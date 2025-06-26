"""
Warehouse Chatbot Application
============================

Ứng dụng chatbot thông minh cho quản lý kho hàng với hỗ trợ:
- Trả lời câu hỏi bằng tiếng Việt
- Tìm kiếm thông tin sản phẩm, tồn kho
- Hướng dẫn quy trình làm việc
- Hỗ trợ giọng nói và giao diện thân thiện

Author: Warehouse AI Team
Version: 2.0.0
"""

__version__ = "2.0.0"
__author__ = "Warehouse AI Team"

from .config import settings
from .database import get_database

__all__ = ["settings", "get_database"]