import os
from tools.retriever.hybrid_retriever import HybridRetriever


def inspect_es_data():
    """查看ES中存储的所有chunk数据"""

    retriever = HybridRetriever()

    print("🔍 查看ES中存储的所有文档...")
    print("=" * 60)

    # 查询所有文档
    try:
        # 获取ES客户端
        es = retriever.es_client.client
        index_name = retriever.es_client.index_name

        print(f"ES客户端类型: {type(es)}")
        print(f"索引名称: {index_name}")

        # 检查索引是否存在
        if not es.indices.exists(index=index_name):
            print(f"❌ 索引 {index_name} 不存在")
            return

        # 使用搜索API获取所有文档
        response = es.search(
            index=index_name,
            body={
                "query": {"match_all": {}},
                "size": 100,  # 可以调整数量
                "sort": [{"metadata.chunk_id": "asc"}]  # 按chunk_id排序
            }
        )

        total = response['hits']['total']['value']
        print(f"📊 ES索引: {index_name}")
        print(f"📄 文档总数: {total}")
        print("=" * 60)

        if total > 0:
            for i, hit in enumerate(response['hits']['hits'], 1):
                source = hit['_source']
                print(f"\n🔹 文档 {i} (ID: {hit['_id']})")
                print(f"  文档ID: {source.get('doc_id', 'N/A')}")
                print(f"  Chunk ID: {source.get('metadata', {}).get('chunk_id', 'N/A')}")
                print(f"  来源文件: {source.get('metadata', {}).get('source', 'N/A')}")
                print(f"  内容长度: {len(source.get('content', ''))} 字符")

                # 检查向量
                embedding = source.get('embedding', [])
                if embedding:
                    print(f"  向量维度: {len(embedding)}")
                    # 检查向量是否全是0
                    if all(abs(v) < 0.0001 for v in embedding[:10]):  # 检查前10个
                        print("  ⚠️ 警告: 向量可能全是0！")
                else:
                    print("  ⚠️ 没有向量数据")

                print(f"  📝 内容预览: {source.get('content', '')[:200]}...")
                print("-" * 40)
        else:
            print("❌ 索引中没有文档")

    except Exception as e:
        print(f"❌ 查询失败: {e}")
        import traceback
        traceback.print_exc()


def check_index_settings():
    """检查ES索引的设置和映射"""

    retriever = HybridRetriever()

    print("🔧 检查索引设置和映射...")
    print("=" * 60)

    try:
        es = retriever.es_client.client
        index_name = retriever.es_client.index_name

        # 检查索引是否存在
        if not es.indices.exists(index=index_name):
            print(f"❌ 索引 {index_name} 不存在")
            return

        # 获取索引设置
        settings = es.indices.get_settings(index=index_name)
        print(f"索引: {index_name}")

        # 获取映射
        mapping = es.indices.get_mapping(index=index_name)

        print(f"\n📁 索引映射:")
        for field, props in mapping[index_name]['mappings']['properties'].items():
            print(f"  {field}: {props.get('type', 'unknown')} | 索引: {props.get('index', 'N/A')}")

        print(f"\n⚙️ 索引设置:")
        index_settings = settings[index_name]['settings']['index']
        for key, value in index_settings.items():
            if key in ['number_of_shards', 'number_of_replicas']:
                print(f"  {key}: {value}")

    except Exception as e:
        print(f"❌ 获取索引信息失败: {e}")


def get_sample_documents(count=3):
    """获取几个样本文档的完整信息"""

    retriever = HybridRetriever()

    print(f"🔍 获取 {count} 个样本文档的完整信息...")
    print("=" * 60)

    try:
        es = retriever.es_client.client
        index_name = retriever.es_client.index_name

        if not es.indices.exists(index=index_name):
            print(f"❌ 索引 {index_name} 不存在")
            return

        # 获取样本文档
        response = es.search(
            index=index_name,
            body={
                "query": {"match_all": {}},
                "size": count
            }
        )

        if response['hits']['total']['value'] > 0:
            for i, hit in enumerate(response['hits']['hits'], 1):
                source = hit['_source']
                print(f"\n📄 样本文档 {i}/{response['hits']['total']['value']}")
                print("=" * 60)

                # 显示元数据
                print(f"📊 元数据:")
                print(f"  ES文档ID: {hit['_id']}")
                print(f"  文档ID: {source.get('doc_id', 'N/A')}")
                print(f"  Chunk ID: {source.get('metadata', {}).get('chunk_id', 'N/A')}")
                print(f"  来源文件: {source.get('metadata', {}).get('source', 'N/A')}")
                print(f"  总块数: {source.get('metadata', {}).get('total_chunks', 'N/A')}")

                # 显示内容
                content = source.get('content', '')
                print(f"\n📝 内容 ({len(content)} 字符):")
                print("-" * 40)
                print(content[:500] + ("..." if len(content) > 500 else ""))
                print("-" * 40)

                # 检查向量
                embedding = source.get('embedding', [])
                if embedding:
                    print(f"\n🧮 向量 ({len(embedding)} 维):")
                    print(f"  前5个值: {embedding[:5]}")
                    print(f"  后5个值: {embedding[-5:]}")
                    print(f"  最小值: {min(embedding):.6f}")
                    print(f"  最大值: {max(embedding):.6f}")
                    print(f"  平均值: {sum(embedding) / len(embedding):.6f}")

                    # 检查是否全是0
                    if all(abs(v) < 0.0001 for v in embedding):
                        print("  ⚠️ 警告: 向量全是0！")
                    elif all(v == 0 for v in embedding):
                        print("  ⚠️ 严重警告: 向量全是0！")
                else:
                    print("⚠️ 警告: 没有向量数据！")

        else:
            print("❌ 索引中没有文档")

    except Exception as e:
        print(f"❌ 获取样本失败: {e}")
        import traceback
        traceback.print_exc()


def test_retrieval_methods():
    """测试检索方法"""

    retriever = HybridRetriever()
    query = "网络暴力"

    print(f"🔍 测试检索方法: '{query}'")
    print("=" * 60)

    # 1. 先检查索引状态
    es = retriever.es_client.client
    index_name = retriever.es_client.index_name

    if not es.indices.exists(index=index_name):
        print(f"❌ 索引 {index_name} 不存在")
        return

    count = es.count(index=index_name)['count']
    print(f"📄 索引中的文档数: {count}")

    if count == 0:
        print("❌ 索引为空，无法检索")
        return

    # 2. 测试各种检索方法
    print("\n1. 测试纯BM25检索...")
    try:
        bm25_results = retriever.es_client.pure_bm25_search(query, top_k=3)
        if bm25_results['hits']['total']['value'] > 0:
            print(f"   ✅ BM25检索成功")
            print(f"   返回 {bm25_results['hits']['total']['value']} 个结果")

            for i, hit in enumerate(bm25_results['hits']['hits'][:3], 1):
                print(f"\n   结果 {i}:")
                print(f"   分数: {hit['_score']:.4f}")
                print(f"   内容: {hit['_source']['content'][:100]}...")
        else:
            print("   ⚠️ BM25没有找到结果")
    except Exception as e:
        print(f"   ❌ BM25检索失败: {e}")

    print("\n2. 测试纯向量检索...")
    try:
        vector_results = retriever.es_client.pure_vector_search(query, top_k=3)
        if vector_results['hits']['total']['value'] > 0:
            print(f"   ✅ 向量检索成功")
            print(f"   返回 {vector_results['hits']['total']['value']} 个结果")

            for i, hit in enumerate(vector_results['hits']['hits'][:3], 1):
                print(f"\n   结果 {i}:")
                print(f"   分数: {hit['_score']:.4f}")
                print(f"   内容: {hit['_source']['content'][:100]}...")
        else:
            print("   ⚠️ 向量检索没有找到结果")
    except Exception as e:
        print(f"   ❌ 向量检索失败: {e}")

    print("\n3. 测试混合检索...")
    try:
        hybrid_results = retriever.retrieve(query, top_k=3, use_hybrid=True)
        if hybrid_results:
            print(f"   ✅ 混合检索成功")
            print(f"   返回 {len(hybrid_results)} 个结果")

            for i, result in enumerate(hybrid_results[:3], 1):
                print(f"\n   结果 {i}:")
                print(f"   分数: {result.get('score', 0):.4f}")
                print(f"   内容: {result.get('content', '')[:100]}...")
        else:
            print("   ⚠️ 混合检索没有找到结果")
    except Exception as e:
        print(f"   ❌ 混合检索失败: {e}")
        import traceback
        traceback.print_exc()


def check_embedding_quality():
    """检查向量质量"""

    retriever = HybridRetriever()

    print("🔍 检查向量质量...")
    print("=" * 60)

    try:
        es = retriever.es_client.client
        index_name = retriever.es_client.index_name

        if not es.indices.exists(index=index_name):
            print(f"❌ 索引 {index_name} 不存在")
            return

        # 检查向量维度
        mapping = es.indices.get_mapping(index=index_name)
        if 'embedding' in mapping[index_name]['mappings']['properties']:
            dim = mapping[index_name]['mappings']['properties']['embedding'].get('dims', 1024)
            print(f"✅ 向量维度配置: {dim}")
        else:
            print("⚠️ 索引中没有embedding字段")
            return

        # 采样检查向量
        response = es.search(
            index=index_name,
            body={
                "query": {"match_all": {}},
                "size": 5,
                "_source": ["embedding", "metadata.chunk_id"]
            }
        )

        print(f"\n📊 向量采样检查 (共{response['hits']['total']['value']}个文档):")

        for i, hit in enumerate(response['hits']['hits'], 1):
            embedding = hit['_source'].get('embedding', [])
            chunk_id = hit['_source'].get('metadata', {}).get('chunk_id', 'N/A')

            if embedding:
                print(f"\n  文档 {i} (Chunk ID: {chunk_id}):")
                print(f"    维度: {len(embedding)}")
                print(f"    前3个值: {embedding[:3]}")

                # 统计零值
                zero_count = sum(1 for v in embedding if abs(v) < 0.0001)
                zero_percent = zero_count / len(embedding) * 100
                print(f"    接近0的值: {zero_count}/{len(embedding)} ({zero_percent:.1f}%)")

                if zero_percent > 90:
                    print("    ⚠️ 警告: 向量大部分为0！")
            else:
                print(f"\n  文档 {i}: 没有向量数据")

    except Exception as e:
        print(f"❌ 检查失败: {e}")


def test_query_embedding():
    """测试查询向量生成"""

    retriever = HybridRetriever()
    query = "网络暴力"

    print(f"🔍 测试查询向量生成: '{query}'")
    print("=" * 60)

    try:
        # 检查是否有embedding_model属性
        if hasattr(retriever, 'embedding_model'):
            print("✅ 找到embedding_model")

            # 生成查询向量
            print(f"生成查询向量...")
            query_embedding = retriever.embedding_model.encode([query])[0]

            print(f"✅ 向量生成成功")
            print(f"维度: {len(query_embedding)}")
            print(f"前5个值: {query_embedding[:5]}")
            print(f"向量范数: {sum(x * x for x in query_embedding) ** 0.5:.4f}")

        else:
            print("❌ 没有embedding_model属性")
            print("可用的属性:")
            for attr in dir(retriever):
                if 'embed' in attr.lower():
                    print(f"  - {attr}: {getattr(retriever, attr)}")

    except Exception as e:
        print(f"❌ 向量生成失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 60)
    print("ES数据检查和调试工具")
    print("=" * 60)

    # 1. 检查索引设置
    check_index_settings()

    print("\n" + "=" * 60)

    # 2. 查看所有文档
    inspect_es_data()

    print("\n" + "=" * 60)

    # 3. 获取样本文档详情
    get_sample_documents(count=3)

    print("\n" + "=" * 60)

    # 4. 检查向量质量
    check_embedding_quality()

    print("\n" + "=" * 60)

    # 5. 测试查询向量生成
    test_query_embedding()

    print("\n" + "=" * 60)

    # 6. 测试检索方法
    test_retrieval_methods()