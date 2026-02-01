from typing import List, Dict

from nltk import word_tokenize

from client.es_client import ElasticsearchClient
from rank_bm25 import BM25Okapi
import numpy as np
import traceback

from config.config import config
from tools.processor.get_embeddings import get_embedding_func



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

    def format_detailed_results(self, detailed_scores: Dict) -> str:
        """格式化详细得分结果"""
        if not detailed_scores or not detailed_scores.get('detailed_results'):
            return "未找到相关结果"

        formatted = []
        query = detailed_scores.get('query', '未知查询')
        weights = detailed_scores.get('weights', {})

        formatted.append(f"查询: {query}")
        formatted.append(
            f"权重设置: BM25={weights.get('bm25_weight', 0.5):.2f}, 向量={weights.get('vector_weight', 0.5):.2f}")
        formatted.append("=" * 60)

        for i, result in enumerate(detailed_scores['detailed_results'], 1):
            formatted.append(f"\n【结果 {i}】")
            formatted.append(f"ID: {result.get('id', 'N/A')}")

            # 显示各个得分
            formatted.append(f"BM25原始得分: {result.get('bm25_score', 0):.4f}")
            formatted.append(f"BM25归一化得分: {result.get('bm25_score_normalized', 0):.4f}")  # 新增
            formatted.append(f"向量相似度得分: {result.get('vector_score', 0):.4f}")

            # 使用归一化后的BM25得分显示计算公式
            formatted.append(
                f"BM25加权得分: {result.get('bm25_score_normalized', 0):.4f} × {result.get('bm25_weight', 0.5):.2f} = {result.get('weighted_bm25', 0):.4f}")

            formatted.append(
                f"向量加权得分: {result.get('vector_score', 0):.4f} × {result.get('vector_weight', 0.5):.2f} = {result.get('weighted_vector', 0):.4f}")
            formatted.append(f"加权和: {result.get('weighted_sum', 0):.4f}")
            formatted.append(f"ES最终得分: {result.get('es_final_score', 0):.4f}")

            # 元数据
            metadata = result.get('metadata', {})
            source = metadata.get('source', '未知')
            page = metadata.get('page', '未知')
            chunk_id = metadata.get('id', '未知')
            formatted.append(f"来源: {source} (页{page}, 块{chunk_id})")

            # 内容预览
            content = result.get('content', '')
            if len(content) > 150:
                content = content[:150] + "..."
            formatted.append(f"内容: {content}")

            formatted.append("-" * 40)

        return '\n'.join(formatted)

    def retrieve_with_detailed_scores(self, query: str, top_k: int = 5) -> Dict:
        """
        执行混合检索并返回详细的得分信息

        Returns:
            包含详细得分信息的字典
        """
        # 生成查询的embedding
        try:
            print("正在生成向量...")
            embeddings = get_embedding_func([query])

            if not embeddings or len(embeddings) == 0:
                print("无法生成向量，将使用全零向量")
                query_vector = [0.0] * self.embedding_dim
            else:
                query_vector = embeddings[0] if isinstance(embeddings, list) else embeddings
                if hasattr(query_vector, 'tolist'):
                    query_vector = query_vector.tolist()
                elif not isinstance(query_vector, list):
                    print(f"向量格式异常: {type(query_vector)}，将使用全零向量")
                    query_vector = [0.0] * self.embedding_dim

                # 确保维度正确
                actual_dim = len(query_vector)
                if actual_dim != self.embedding_dim:
                    print(f"向量维度不正确: {actual_dim}，期望{self.embedding_dim}")
                    if actual_dim > self.embedding_dim:
                        query_vector = query_vector[:self.embedding_dim]
                    else:
                        query_vector = query_vector + [0.0] * (self.embedding_dim - actual_dim)

            print(f"向量生成成功，维度: {len(query_vector)}")

        except Exception as e:
            print(f"生成向量时出错: {e}")
            traceback.print_exc()
            query_vector = [0.0] * self.embedding_dim

        # 使用新的方法获取详细得分
        print("执行混合检索（带具体得分）...")
        detailed_scores = self.es_client.get_separate_scores(query, query_vector, top_k)

        # 计算加权得分
        hybrid_results = detailed_scores.get('hybrid_results', [])

        # 找到BM25得分的最大值用于归一化
        bm25_scores = [result.get('bm25_score', 0.0) for result in hybrid_results]
        max_bm25 = max(bm25_scores) if bm25_scores else 1.0

        for result in hybrid_results:
            bm25_score = result.get('bm25_score', 0.0)
            vector_score = result.get('vector_score', 0.0)

            # 归一化BM25得分
            normalized_bm25 = bm25_score / max_bm25 if max_bm25 > 0 else 0.0

            # 从配置获取权重
            bm25_weight = getattr(config, 'BM25_WEIGHT', 0.5)
            vector_weight = getattr(config, 'VECTOR_WEIGHT', 0.5)

            # 使用归一化后的BM25得分计算加权得分
            weighted_bm25 = normalized_bm25 * bm25_weight
            weighted_vector = vector_score * vector_weight

            result['bm25_weight'] = bm25_weight
            result['vector_weight'] = vector_weight
            result['bm25_score_normalized'] = normalized_bm25
            result['weighted_bm25'] = weighted_bm25
            result['weighted_vector'] = weighted_vector
            result['weighted_sum'] = weighted_bm25 + weighted_vector

            # 注意：Elasticsearch的最终得分可能不是简单的加权和
            # 因为它会做归一化和其他处理
            result['es_final_score'] = result.get('hybrid_score', 0.0)

        return {
            'query': query,
            'detailed_results': hybrid_results,
            'bm25_scores': detailed_scores.get('bm25_scores', {}),
            'vector_scores': detailed_scores.get('vector_scores', {}),
            'weights': {
                'bm25_weight': getattr(config, 'BM25_WEIGHT', 0.5),
                'vector_weight': getattr(config, 'VECTOR_WEIGHT', 0.5)
            }
        }
