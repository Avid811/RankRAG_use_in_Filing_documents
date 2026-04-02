import os
import json
from tools.retriever.hybrid_retriever import HybridRetriever


def simplified_rag_test():
    """改进版RAG测试：读取准备好的JSON数据入库，然后执行混合检索"""
    print("=" * 50)
    print("RAG知识库入库与检索测试 (适配JSON数据)")
    print("=" * 50)

    # 1. 实例化 Retriever (内部包含了对 es_client 的封装)
    retriever = HybridRetriever()

    # 2. 读取已经准备好的、带向量和完整metadata的JSON文件
    data_dir = r"/data/source_data/processed_chunks.json"

    print("\n1. 正在读取并初始化知识库...")
    if not os.path.exists(data_dir):
        print(f"数据文件不存在: {data_dir}")
        print("请确认你已经将 List[dict] 数据保存为了该 JSON 文件。")
        return

    # 读取 JSON 数据
    with open(data_dir, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    print(f"成功读取 JSON 文件，共 {len(documents)} 条 chunk 数据。")

    # 3. 索引操作
    print("正在删除旧索引...")
    if retriever.es_client.delete_index():
        print("旧索引已删除")
    else:
        print("旧索引不存在或删除失败")

    print("正在创建新索引（1024维度）...")
    if retriever.es_client.create_index(embedding_dim=1024):
        print("新索引创建成功")
    else:
        print("索引创建失败，退出程序")
        return

    # 直接调用 es_client.index_documents 批量入库
    print("正在将 JSON 数据批量推送到 Elasticsearch...")
    success, failed = retriever.es_client.index_documents(documents, batch_size=100)
    print(f"知识库初始化完成！成功: {success}, 失败: {failed}")

    # 4. 详细混合检索测试
    print("\n" + "=" * 50)
    print("2. 执行详细混合检索测试...")
    print("=" * 50)

    # 修改查询词，测试你的法律类数据
    query = "个人信息可能在境外被访问"
    print(f"查询: {query}")
    print("-" * 30)

    try:
        # 执行混合检索获取详细分数
        detailed_results = retriever.retrieve_with_detailed_scores(query, top_k=3)

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
                print(f"内容摘要: {result.get('content', '')[:150]}...")
                print("-" * 20)
                # ================= 展示各项分数的推导过程 =================
                print(f"BM25 原始分:     {raw_bm25:.4f} (按比例压缩为: {norm_bm25:.4f})")
                print(f"向量 检索得分:   {vector_score:.4f}")
                print(f"最终 混合得分:   {hybrid_score:.4f}  (计算公式: {norm_bm25:.4f}*0.4 + {vector_score:.4f}*0.6)")
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