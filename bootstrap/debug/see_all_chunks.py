import os
from tools.retriever.hybrid_retriever import HybridRetriever

def simplified_rag_test():
    """诊断版RAG测试：增加数据量检查与报错溯源"""
    print("=" * 50)
    print("诊断版 RAG 检索测试")
    print("=" * 50)

    retriever = HybridRetriever()

    # 【诊断点 1】：检查 ES 里到底有没有数据
    doc_count = retriever.es_client.get_document_count()
    print(f"👉 [诊断 1] 当前知识库 (ES索引) 的文档总数: {doc_count} 条")
    if doc_count == 0:
        print("❌ 致命错误：你的 Elasticsearch 里面没有数据！")
        print("💡 解决方案：你之前存的数据丢失了（或连错库了），请先运行一次包含 `index_documents` 的代码，把 JSON 数据重新写进去。")
        return

    query = "个人信息可能在境外被访问"
    print(f"\n查询: {query}")
    print("-" * 30)

    try:
        # 执行混合检索获取详细分数
        detailed_results = retriever.retrieve_with_detailed_scores(query, top_k=3)

        if not detailed_results or not detailed_results.get('detailed_results'):
            print("❌ [诊断 2] 检索完成，但底层 ES 没返回任何东西！")
            print("💡 请往上翻一翻控制台输出，是否有看到：")
            print("   1. '无法生成向量，将使用全零向量' (说明 Embedding API 坏了)")
            print("   2. 'KNN混合检索失败' (说明 ES 拒绝了你的检索请求)")
        else:
            print("\n✅ 检索成功，找到相关结果！")

            for i, result in enumerate(detailed_results['detailed_results'], 1):
                raw_bm25 = result.get('bm25_score', 0.0)
                norm_bm25 = result.get('bm25_score_normalized', 0.0)
                vector_score = result.get('vector_score', 0.0)
                hybrid_score = result.get('hybrid_score', 0.0)

                print(f"\n【Top {i}】")
                print(f"ChunkID: {result.get('chunkId', '未知')}")
                print(f"来源: {result.get('metadata', {}).get('source', '未知')}")
                print(f"法律状态: {result.get('metadata', {}).get('status', '未知')}")
                print(f"条款: {result.get('metadata', {}).get('articleNumber', '未知')}")
                print(f"内容摘要: {result.get('content', '')[:150]}...")
                print("-" * 20)
                # ================= 展示各项分数的推导过程 =================
                print(f"BM25 原始分:     {raw_bm25:.4f} (按比例压缩为: {norm_bm25:.4f})")
                print(f"向量 检索得分:   {vector_score:.4f}")
                print(f"最终 混合得分:   {hybrid_score:.4f}  (计算公式: {norm_bm25:.4f}*0.4 + {vector_score:.4f}*0.6)")
                print("-" * 20)

    except Exception as e:
        print(f"检索过程中出错: {e}")
        import traceback
        traceback.print_exc()

    print("\n测试完成")


if __name__ == "__main__":
    simplified_rag_test()