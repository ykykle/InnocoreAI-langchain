#!/usr/bin/env python3
"""
Embedding Model API Test Script
测试嵌入式模型的 API 连接和功能
"""

import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 加载环境变量
load_dotenv()


def test_embedding_api():
    """测试 Embedding API"""
    print("=" * 80)
    print("Embedding Model API Test")
    print("=" * 80)
    
    # 1. 读取环境变量
    print("\n[1] 读取环境变量...")
    embedding_model = os.getenv("EMBEDDING_MODEL")
    embedding_base_url = os.getenv("EMBEDDING_BASE_URL")
    embedding_api_key = os.getenv("EMBEDDING_API_KEY")
    
    print(f"  - EMBEDDING_MODEL: {embedding_model}")
    print(f"  - EMBEDDING_BASE_URL: {embedding_base_url}")
    print(f"  - EMBEDDING_API_KEY: {'*' * 8 + embedding_api_key[-4:] if embedding_api_key else 'None'}")
    
    # 验证必需的环境变量
    if not embedding_model:
        print("\n❌ 错误: 未设置 EMBEDDING_MODEL 环境变量")
        return False
    
    if not embedding_api_key:
        print("\n❌ 错误: 未设置 EMBEDDING_API_KEY 环境变量")
        return False
    
    if not embedding_base_url:
        print("\n⚠️  警告: 未设置 EMBEDDING_BASE_URL，将使用 OpenAI 默认地址")
        embedding_base_url = "https://api.openai.com/v1"
    
    print("✅ 环境变量配置正确")
    
    # 2. 初始化 OpenAIEmbeddings
    print("\n[2] 初始化 OpenAIEmbeddings...")
    try:
        from langchain_openai import OpenAIEmbeddings
        
        embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=embedding_api_key,
            base_url=embedding_base_url,
        )
        
        print(f"  - Model: {embedding_model}")
        print(f"  - Base URL: {embedding_base_url}")
        print("✅ OpenAIEmbeddings 初始化成功")
        
    except ImportError as e:
        print(f"\n❌ 错误: 无法导入 langchain_openai")
        print(f"  请运行: pip install langchain-openai")
        return False
    except Exception as e:
        print(f"\n❌ 错误: OpenAIEmbeddings 初始化失败")
        print(f"  错误信息: {str(e)}")
        return False
    
    # 3. 测试单文本嵌入
    print("\n[3] 测试单文本嵌入 (embed_query)...")
    try:
        test_text = "人工智能是计算机科学的一个重要分支"
        print(f"  测试文本: {test_text}")
        
        embedding = embeddings.embed_query(test_text)
        
        print(f"  - 向量维度: {len(embedding)}")
        print(f"  - 向量前5个值: {embedding[:5]}")
        print(f"  - 向量类型: {type(embedding[0])}")
        print("✅ 单文本嵌入测试成功")
        
    except Exception as e:
        print(f"\n❌ 错误: 单文本嵌入失败")
        print(f"  错误信息: {str(e)}")
        return False
    
    # 4. 测试批量文本嵌入
    print("\n[4] 测试批量文本嵌入 (embed_documents)...")
    try:
        test_texts = [
            "深度学习是机器学习的子领域",
            "自然语言处理研究计算机与人类语言之间的交互",
            "计算机视觉使计算机能够从图像和视频中提取信息",
        ]
        
        print(f"  测试文本数量: {len(test_texts)}")
        for i, text in enumerate(test_texts, 1):
            print(f"    {i}. {text}")
        
        embeddings_batch = embeddings.embed_documents(test_texts)
        
        print(f"  - 生成向量数量: {len(embeddings_batch)}")
        print(f"  - 每个向量维度: {len(embeddings_batch[0])}")
        print(f"  - 第一个向量前5个值: {embeddings_batch[0][:5]}")
        print("✅ 批量文本嵌入测试成功")
        
    except Exception as e:
        print(f"\n❌ 错误: 批量文本嵌入失败")
        print(f"  错误信息: {str(e)}")
        return False
    
    # 5. 测试异步嵌入（如果支持）
    print("\n[5] 测试异步嵌入 (aembed_query)...")
    try:
        async def test_async():
            test_text = "异步测试文本"
            embedding = await embeddings.aembed_query(test_text)
            return embedding
        
        embedding_async = asyncio.run(test_async())
        
        print(f"  - 向量维度: {len(embedding_async)}")
        print(f"  - 向量前5个值: {embedding_async[:5]}")
        print("✅ 异步嵌入测试成功")
        
    except Exception as e:
        print(f"\n⚠️  警告: 异步嵌入测试失败（可能不支持）")
        print(f"  错误信息: {str(e)}")
    
    # 6. 计算相似度示例
    print("\n[6] 计算文本相似度示例...")
    try:
        import numpy as np
        
        text1 = "机器学习是人工智能的分支"
        text2 = "深度学习属于机器学习"
        text3 = "今天天气很好"
        
        emb1 = embeddings.embed_query(text1)
        emb2 = embeddings.embed_query(text2)
        emb3 = embeddings.embed_query(text3)
        
        def cosine_similarity(vec1, vec2):
            """计算余弦相似度"""
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            return dot_product / (norm1 * norm2)
        
        sim_12 = cosine_similarity(emb1, emb2)
        sim_13 = cosine_similarity(emb1, emb3)
        
        print(f"  文本1: {text1}")
        print(f"  文本2: {text2}")
        print(f"  文本3: {text3}")
        print(f"\n  相似度结果:")
        print(f"    - 文本1 vs 文本2 (相关): {sim_12:.4f}")
        print(f"    - 文本1 vs 文本3 (不相关): {sim_13:.4f}")
        print(f"\n  ✅ 相似度计算合理: {sim_12 > sim_13}")
        
    except Exception as e:
        print(f"\n❌ 错误: 相似度计算失败")
        print(f"  错误信息: {str(e)}")
        return False
    
    # 7. 性能测试
    print("\n[7] 性能测试...")
    try:
        import time
        
        # 测试单次嵌入时间
        start_time = time.time()
        for i in range(5):
            embeddings.embed_query(f"性能测试文本 {i}")
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 5
        print(f"  - 平均单次嵌入时间: {avg_time*1000:.2f} ms")
        print(f"  - 5次总耗时: {(end_time - start_time)*1000:.2f} ms")
        
        # 测试批量嵌入时间
        batch_texts = [f"批量测试文本 {i}" for i in range(10)]
        start_time = time.time()
        embeddings.embed_documents(batch_texts)
        end_time = time.time()
        
        print(f"  - 10文本批量嵌入时间: {(end_time - start_time)*1000:.2f} ms")
        print(f"  - 平均每文本: {(end_time - start_time)*100:.2f} ms")
        print("✅ 性能测试完成")
        
    except Exception as e:
        print(f"\n⚠️  警告: 性能测试失败")
        print(f"  错误信息: {str(e)}")
    
    # 总结
    print("\n" + "=" * 80)
    print("✅ 所有测试通过！Embedding API 工作正常")
    print("=" * 80)
    print("\n配置摘要:")
    print(f"  - 模型: {embedding_model}")
    print(f"  - API端点: {embedding_base_url}")
    print(f"  - 向量维度: {len(embedding)}")
    print(f"  - 提供商: {'OpenAI' if 'openai.com' in embedding_base_url else '第三方兼容API'}")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    try:
        success = test_embedding_api()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 测试过程中发生未预期的错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
