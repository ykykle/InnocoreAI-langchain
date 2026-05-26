#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Embedding configuration diagnostic script
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from core.config import get_config

def test_embedding_config():
    """Test embedding configuration"""
    config = get_config()
    
    print("=" * 60)
    print("Embedding Configuration Diagnostic")
    print("=" * 60)
    
    # Check environment variables
    print("\nEnvironment Variables:")
    print(f"  EMBEDDING_API_KEY: {'OK' if os.getenv('EMBEDDING_API_KEY') else 'NOT SET'}")
    print(f"  EMBEDDING_MODEL: {'OK' if os.getenv('EMBEDDING_MODEL') else 'NOT SET'}")
    print(f"  EMBEDDING_BASE_URL: {'OK' if os.getenv('EMBEDDING_BASE_URL') else 'NOT SET'}")
    print(f"  OPENAI_API_KEY: {'OK' if os.getenv('OPENAI_API_KEY') else 'NOT SET'}")
    print(f"  OPENAI_BASE_URL: {'OK' if os.getenv('OPENAI_BASE_URL') else 'NOT SET'}")
    
    # Check loaded configuration
    print("\nLoaded Configuration:")
    print(f"  vector_db.embedding_model: {config.vector_db.embedding_model}")
    print(f"  vector_db.embedding_base_url: {config.vector_db.embedding_base_url}")
    print(f"  vector_db.api_key: {'LOADED' if config.vector_db.api_key else 'NOT LOADED'}")
    if config.vector_db.api_key:
        print(f"    Prefix: {config.vector_db.api_key[:10]}...")
    
    print(f"\n  llm.api_key: {'LOADED' if config.llm.api_key else 'NOT LOADED'}")
    if config.llm.api_key:
        print(f"    Prefix: {config.llm.api_key[:10]}...")
    print(f"  llm.base_url: {config.llm.base_url}")
    
    # Determine priority
    print("\nAPI Key Priority:")
    if config.vector_db.api_key:
        print(f"  Will use EMBEDDING_API_KEY")
        print(f"    Value: {config.vector_db.api_key[:20]}...")
    elif config.llm.api_key:
        print(f"  Will fallback to OPENAI_API_KEY")
        print(f"    Value: {config.llm.api_key[:20]}...")
    else:
        print(f"  ERROR: No API Key found!")
    
    # Determine Base URL
    print("\nBase URL Priority:")
    if config.vector_db.embedding_base_url:
        print(f"  Will use EMBEDDING_BASE_URL")
        print(f"    Value: {config.vector_db.embedding_base_url}")
    elif config.llm.base_url:
        print(f"  Will use LLM BASE_URL")
        print(f"    Value: {config.llm.base_url}")
    else:
        print(f"  Will use default OpenAI URL")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_embedding_config()
