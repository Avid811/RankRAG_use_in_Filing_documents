import os
from tools.processor.knowledge_loader import KnowledgeBaseLoader
from tools.retriever.hybrid_retriever import HybridRetriever


def simplified_rag_test():
    """简化版RAG测试：入库一次，然后执行混合检索"""
    print("=" * 50)
    print("RAG知识库简化测试")
    print("=" * 50)

    # 初始化组件
    loader = KnowledgeBaseLoader()
    retriever = HybridRetriever()

    # 1. 初始化知识库
    print("\n1. 正在初始化知识库...")
    data_dir = "/Users/aono/Desktop/RankRAG/RankRAG_use_in_Filing_documents/data/knowledge.txt"

    if not os.path.exists(data_dir):
        print(f"❌ 数据文件不存在: {data_dir}")
        return

    # 删除旧索引
    print("正在删除旧索引...")
    if retriever.es_client.delete_index():
        print("✅ 旧索引已删除")
    else:
        print("⚠️  删除旧索引失败或索引不存在")

    # 创建新索引
    print("正在创建新索引（1024维度）...")
    if retriever.es_client.create_index(embedding_dim=1024):
        print("✅ 新索引创建成功")
    else:
        print("❌ 索引创建失败")
        return

    # 加载知识库
    print("正在加载知识库...")
    loader.initialize_knowledge_base(data_dir)
    print("✅ 知识库初始化完成")

    # 2. 详细混合检索测试
    print("\n" + "=" * 50)
    print("2. 执行详细混合检索测试...")
    print("=" * 50)

    query = "网络暴力"
    print(f"查询: {query}")
    print("-" * 30)

    try:
        # 执行详细混合检索
        detailed_results = retriever.retrieve_with_detailed_scores(query, top_k=3)

        if detailed_results and detailed_results.get('detailed_results'):
            print("\n详细检索结果:")
            print(retriever.format_detailed_results(detailed_results))

            # 额外的分析
            print("\n" + "=" * 50)
            print("检索结果分析:")
            print("=" * 50)

            for i, result in enumerate(detailed_results['detailed_results'], 1):
                print(f"\n文档 {i} 分析:")
                print(f"BM25贡献: {result.get('weighted_bm25', 0):.4f}")
                print(f"向量贡献: {result.get('weighted_vector', 0):.4f}")
                print(f"总和: {result.get('weighted_sum', 0):.4f}")
                print(f"ES评分: {result.get('es_final_score', 0):.4f}")
        else:
            print("未找到相关结果")

    except Exception as e:
        print(f" 检索过程中出错: {e}")
        import traceback
        traceback.print_exc()

    print("\n 测试完成")


if __name__ == "__main__":
    simplified_rag_test()