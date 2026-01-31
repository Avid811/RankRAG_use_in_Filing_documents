from typing import List, Dict
from client.es_client import ElasticsearchClient
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
import traceback

from tools.processor.get_embeddings import get_embedding_func

nltk.download('punkt')


class HybridRetriever:
    def __init__(self):
        self.es_client = ElasticsearchClient()
        self.embedding_dim = 1024  # 阿里云API返回1024维向量

    def retrieve(self, query: str, top_k: int = 5,
                 use_hybrid: bool = True) -> List[Dict]:
        """
        检索相关文档

        Args:
            query: 查询文本
            top_k: 返回结果数量
            use_hybrid: 是否使用混合检索

        Returns:
            相关文档列表
        """
        # 生成查询的embedding
        try:
            print("正在生成向量...")
            embeddings = get_embedding_func([query])

            if not embeddings or len(embeddings) == 0:
                print("⚠️ 无法生成向量，将使用全零向量")
                query_vector = [0.0] * self.embedding_dim
            else:
                # 处理返回的向量格式
                query_vector = embeddings[0] if isinstance(embeddings, list) else embeddings

                # 确保是Python列表，而不是numpy数组
                if hasattr(query_vector, 'tolist'):
                    query_vector = query_vector.tolist()
                elif not isinstance(query_vector, list):
                    print(f"⚠️ 向量格式异常: {type(query_vector)}，将使用全零向量")
                    query_vector = [0.0] * self.embedding_dim

                # 确保维度正确
                actual_dim = len(query_vector)
                if actual_dim != self.embedding_dim:
                    print(f"⚠️ 向量维度不正确: {actual_dim}，期望{self.embedding_dim}")
                    if actual_dim > self.embedding_dim:
                        query_vector = query_vector[:self.embedding_dim]
                        print(f"已截断到{self.embedding_dim}维")
                    else:
                        query_vector = query_vector + [0.0] * (self.embedding_dim - actual_dim)
                        print(f"已填充到{self.embedding_dim}维")

                print(f"✅ 向量生成成功，维度: {len(query_vector)}")

        except Exception as e:
            print(f"❌ 生成向量时出错: {e}")
            traceback.print_exc()
            query_vector = [0.0] * self.embedding_dim

        if use_hybrid:
            # 使用混合检索
            print("🔍 执行混合检索...")
            results = self.es_client.hybrid_search(query, query_vector, top_k)
        else:
            # 仅使用向量检索
            print("🔍 执行纯向量检索...")
            results = self.es_client.pure_vector_search(query_vector, top_k)

        return results

    def rerank_with_bm25(self, query: str, documents: List[Dict]) -> List[Dict]:
        """使用BM25重新排序"""
        if not documents:
            return []

        # 准备BM25
        tokenized_docs = [word_tokenize(doc['content']) for doc in documents]
        bm25 = BM25Okapi(tokenized_docs)

        # 计算BM25分数
        tokenized_query = word_tokenize(query)
        bm25_scores = bm25.get_scores(tokenized_query)

        # 结合分数
        for i, doc in enumerate(documents):
            combined_score = 0.7 * doc.get('score', 0) + 0.3 * bm25_scores[i]
            doc['combined_score'] = combined_score
            doc['bm25_score'] = bm25_scores[i]

        # 按综合分数排序
        documents.sort(key=lambda x: x.get('combined_score', 0), reverse=True)

        return documents

    def format_results(self, results: List[Dict]) -> str:
        """格式化检索结果"""
        if not results:
            return "未找到相关结果"

        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(f"\n【结果 {i}】")
            formatted.append(f"分数: {result.get('score', 0):.4f}")
            metadata = result.get('metadata', {})
            source = metadata.get('source', '未知')
            page = metadata.get('page', '未知')
            chunk_id = metadata.get('chunk_id', '未知')
            formatted.append(f"来源: {source} (页{page}, 块{chunk_id})")

            # 截取适当长度的内容
            content = result.get('content', '')
            if len(content) > 200:
                content = content[:200] + "..."
            formatted.append(f"内容: {content}")

            if 'combined_score' in result:
                formatted.append(f"综合分数: {result['combined_score']:.4f}")

        return '\n'.join(formatted)