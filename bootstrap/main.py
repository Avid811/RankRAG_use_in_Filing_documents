import os

from config import config
import argparse
from tools.processor.knowledge_loader import KnowledgeBaseLoader
from tools.retriever.hybrid_retriever import HybridRetriever


def demo_rag_pipeline():
    """演示RAG完整流程"""

    # 1. 初始化
    print("=" * 50)
    print("RAG知识库系统演示")
    print("=" * 50)

    loader = KnowledgeBaseLoader()
    retriever = HybridRetriever()

    while True:
        print("\n请选择操作:")
        print("1. 初始化知识库（会删除现有数据！）")
        print("2. 混合检索")
        print("3. 纯向量检索")
        print("4. 纯BM25检索")
        print("5. 退出")

        choice = input("请输入选择 (1-5): ").strip()

        if choice == "1":
            # 初始化知识库
            data_dir = "/Users/aono/Desktop/RankRAG/RankRAG_use_in_Filing_documents/data/knowledge.txt"
            if os.path.exists(data_dir):
                loader.initialize_knowledge_base(data_dir)
                print("已创建RAG文档")


        elif choice == "2":
            # 混合检索
            query = input("请输入查询: ").strip()
            if query:
                print(f"\n查询: {query}")
                print("-" * 30)

                # 执行混合检索
                results = retriever.retrieve(query, top_k=3, use_hybrid=True)

                if results:
                    print("混合检索结果:")
                    print(retriever.format_results(results))
                else:
                    print("未找到相关结果")

        elif choice == "3":
            # 纯向量检索
            query = input("请输入查询: ").strip()
            if query:
                print(f"\n查询: {query}")
                print("-" * 30)

                results = retriever.retrieve(query, top_k=3, use_hybrid=False)

                if results:
                    print("向量检索结果:")
                    print(retriever.format_results(results))
                else:
                    print("未找到相关结果")

        elif choice == "4":
            # 纯BM25检索
            query = input("请输入查询: ").strip()
            if query:
                print(f"\n查询: {query}")
                print("-" * 30)

                # 通过ES进行BM25检索
                response = retriever.es_client.pure_bm25_search(query, top_k=3)

                if response['hits']['total']['value'] > 0:
                    print("BM25检索结果:")
                    for i, hit in enumerate(response['hits']['hits'], 1):
                        print(f"\n【结果 {i}】")
                        print(f"分数: {hit['_score']:.4f}")
                        print(f"来源: {hit['_source']['metadata']['source']}")
                        print(f"内容: {hit['_source']['content'][:200]}...")
                else:
                    print("未找到相关结果")

        elif choice == "5":
            print("退出系统")
            break

        else:
            print("无效选择，请重新输入")


if __name__ == "__main__":
        demo_rag_pipeline()