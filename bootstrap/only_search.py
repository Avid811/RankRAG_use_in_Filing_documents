import os
from tools.processor.knowledge_loader import KnowledgeBaseLoader
from tools.retriever.hybrid_retriever import HybridRetriever


def simplified_rag_test():
    """简化版RAG测试：复用已有知识库进行检索"""
    print("=" * 50)
    print("RAG知识库检索测试（复用现有知识库）")
    print("=" * 50)

    # 初始化检索器
    retriever = HybridRetriever()

    # 1. 检查索引是否存在 ,索引我们在config配置
    print("\n1. 检查索引状态...")
    if retriever.es_client.index_exists():
        print("索引已存在，直接复用")
    else:
        print("索引不存在，需要重新初始化知识库")
        # 如果索引不存在，再初始化
        loader = KnowledgeBaseLoader()
        data_dir = "/Users/aono/Desktop/RankRAG/RankRAG_use_in_Filing_documents/data/knowledge.txt"

        if not os.path.exists(data_dir):
            print(f"数据文件不存在: {data_dir}")
            return

        # 创建新索引
        print("正在创建新索引（1024维度）...")
        if retriever.es_client.create_index(embedding_dim=1024):
            print("新索引创建成功")
        else:
            print("索引创建失败")
            return

        # 加载知识库
        print("正在加载知识库...")
        loader.initialize_knowledge_base(data_dir)
        print("知识库初始化完成")

    # 2. 执行混合检索测试
    print("\n" + "=" * 50)
    print("2. 执行混合检索测试...")
    print("=" * 50)

    query = "我们可能会推送一些含有'血腥'、'暴力'的内容，用户可以选择不感兴趣"
    print(f"查询: {query}")
    print("-" * 30)

    try:
        # 执行详细混合检索
        detailed_results = retriever.retrieve_with_detailed_scores(query, top_k=3)

        if detailed_results and detailed_results.get('detailed_results'):
            print("\n详细检索结果:")
            print(retriever.format_detailed_results(detailed_results))

    except Exception as e:
        print(f"检索过程中出错: {e}")
        import traceback
        traceback.print_exc()

    print("\n测试完成")


if __name__ == "__main__":
    simplified_rag_test()