from typing import List, Dict
from client.es_client import ElasticsearchClient
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk

from tools.processor.get_embeddings import get_embedding_func

nltk.download('punkt')


class HybridRetriever:
    def __init__(self):
        self.es_client = ElasticsearchClient()

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
        query_vector = get_embedding_func([query])

        if use_hybrid:
            # 使用混合检索
            results = self.es_client.hybrid_search(query, query_vector, top_k)
        else:
            # 仅使用向量检索
            response = self.es_client.pure_vector_search(query_vector, top_k)
            results = []
            for hit in response['hits']['hits']:
                result = {
                    'content': hit['_source']['content'],
                    'metadata': hit['_source']['metadata'],
                    'score': hit['_score'],
                    'id': hit['_id']
                }
                results.append(result)

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
        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(f"\n【结果 {i}】")
            formatted.append(f"分数: {result.get('score', 0):.4f}")
            formatted.append(f"来源: {result.get('metadata', {}).get('source', '未知')}")
            formatted.append(f"内容: {result['content'][:200]}...")
            if 'combined_score' in result:
                formatted.append(f"综合分数: {result['combined_score']:.4f}")

        return '\n'.join(formatted)