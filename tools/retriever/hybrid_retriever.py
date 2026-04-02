from typing import List, Dict
from nltk import word_tokenize
from client.es_client import ElasticsearchClient
from rank_bm25 import BM25Okapi
import traceback
from config.config import config
from tools.LLMs.get_embeddings import get_embedding_func


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

    def retrieve_with_detailed_scores(self, query: str, top_m: int = 5) -> Dict:
        """
        执行混合检索并返回详细的得分信息

        Returns:
            包含详细得分信息的字典
        """
        # ==================== 1. 向量生成部分 (保持原样不变) ====================
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

        # ==================== 2. 核心修复：获取更大的候选池 ====================
        # 必须放大召回数量 (例如 top_k 的 5 倍)，防止好文档在第一轮被 ES 错误的算分机制淘汰
        recall_size = top_k * 5
        print(f"执行混合检索（从底层捞取 {recall_size} 个候选文档进行精准打分重排）...")

        # 这里的 top_k 参数传入放大后的 recall_size
        detailed_scores = self.es_client.get_separate_scores(query, query_vector, top_k=recall_size)
        hybrid_results = detailed_scores.get('hybrid_results', [])

        # 找到这批候选里 BM25 的最高分，用于计算归一化
        bm25_scores = [result.get('bm25_score', 0.0) for result in hybrid_results]
        max_bm25 = max(bm25_scores) if bm25_scores and max(bm25_scores) > 0 else 1.0

        bm25_weight = getattr(config, 'BM25_WEIGHT', 0.4)
        vector_weight = getattr(config, 'VECTOR_WEIGHT', 0.6)

        # ==================== 3. 核心修复：重新算分并覆盖旧分数 ====================
        for result in hybrid_results:
            bm25_score = result.get('bm25_score', 0.0)
            vector_score = result.get('vector_score', 0.0)

            # 归一化BM25得分 (压缩到 0~1)
            normalized_bm25 = bm25_score / max_bm25

            # 计算真正的加权得分
            weighted_bm25 = normalized_bm25 * bm25_weight
            weighted_vector = vector_score * vector_weight
            weighted_sum = weighted_bm25 + weighted_vector

            result['bm25_weight'] = bm25_weight
            result['vector_weight'] = vector_weight
            result['bm25_score_normalized'] = normalized_bm25
            result['weighted_bm25'] = weighted_bm25
            result['weighted_vector'] = weighted_vector

            # 【修复张冠李戴】：将真实算对的分数保存为 hybrid_score，以便外部统一打印
            result['weighted_sum'] = weighted_sum
            result['es_final_score'] = result.get('hybrid_score', 0.0)  # 把 ES 瞎算的原始分备份一下
            result['hybrid_score'] = weighted_sum  # 覆盖旧分数！

        # ==================== 4. 核心修复：根据新分数重新排序并截取 ====================
        # 必须要 sort，否则依然是按照 ES 瞎算的顺序返回
        hybrid_results.sort(key=lambda x: x['hybrid_score'], reverse=True)

        # 排序完成后，再精准切出用户实际想要的 top_k 个结果
        final_results = hybrid_results[:top_k]

        return {
            'query': query,
            'detailed_results': final_results,
            'bm25_scores': detailed_scores.get('bm25_scores', {}),
            'vector_scores': detailed_scores.get('vector_scores', {}),
            'weights': {
                'bm25_weight': bm25_weight,
                'vector_weight': vector_weight
            }
        }
