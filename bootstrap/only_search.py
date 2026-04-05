from config.config import config
from tools.retriever.hybrid_retriever import HybridRetriever



def simplified_rag_test():
    """简化版RAG测试：复用已有知识库进行检索"""
    # 修改查询词，测试你的法律类数据
    query = "个人信息可能在境外被访问"
    print(f"查询: {query}")
    print("-" * 30)
    retriever = HybridRetriever()

    try:
        # 执行混合检索获取详细分数
        detailed_results = retriever.retrieve_with_detailed_scores(query, top_m=15)

        if detailed_results and detailed_results.get('detailed_results'):
            print("\n检索成功，找到相关结果！")

            for i, result in enumerate(detailed_results['detailed_results'], 1):
                # 读取全部细节分数
                raw_bm25 = result.get('bm25_score', 0.0)
                norm_bm25 = result.get('bm25_score_normalized', 0.0)
                vector_score = result.get('vector_score', 0.0)
                # 这个 hybrid_score 已经是 retriever 里被重写后的正确加权分数了
                hybrid_score = result.get('hybrid_score', 0.0)

                print(f"\n【Top {i}】")
                print(f"ChunkID: {result.get('chunkId', '未知')}")
                print(f"来源: {result.get('metadata', {}).get('source', '未知')}")
                print(f"法律状态: {result.get('metadata', {}).get('status', '未知')}")
                print(f"条款: {result.get('metadata', {}).get('articleNumber', '未知')}")
                print(f"内容摘要: {result.get('content', '')}")
                print("-" * 20)
                # ================= 展示各项分数的推导过程 =================
                print(f"BM25 原始分:     {raw_bm25:.4f} (按比例压缩为: {norm_bm25:.4f})")
                print(f"向量 检索得分:   {vector_score:.4f}")
                print(f"最终 混合得分:   {hybrid_score:.4f}  (计算公式: {norm_bm25:.4f}*{config.BM25_WEIGHT} + {vector_score:.4f}*{config.VECTOR_WEIGHT})")
                print("-" * 20)
        else:
            print("未找到相关结果")

    except Exception as e:
        print(f"检索过程中出错: {e}")
        import traceback
        traceback.print_exc()

    print("\n测试完成")


if __name__ == "__main__":
    simplified_rag_test()