"""
InnoCore AI 核心模块
"""

from .config import InnoCoreConfig, get_config, update_config
from .exceptions import *


def get_database_manager():
    """惰性导入 DatabaseManager"""
    from .database import DatabaseManager
    return DatabaseManager


def get_vector_store_manager():
    """惰性导入 VectorStoreManager"""
    from .vector_store import VectorStoreManager
    return VectorStoreManager


__all__ = [
    "InnoCoreConfig",
    "get_config",
    "update_config",
    "get_database_manager",
    "get_vector_store_manager",
]
