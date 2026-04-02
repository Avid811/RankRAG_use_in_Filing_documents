from tools.LLMs.get_embeddings import get_embedding_func

# 测试向量生成
test_query = "网络暴力"
print(f"测试查询: {test_query}")

try:
    embeddings = get_embedding_func([test_query])
    print(f"返回值类型: {type(embeddings)}")

    if isinstance(embeddings, list):
        print(f"列表长度: {len(embeddings)}")
        if len(embeddings) > 0:
            vector = embeddings[0]
            print(f"向量类型: {type(vector)}")
            print(f"向量维度: {len(vector) if isinstance(vector, list) else '非列表'}")

            # 显示前5个值
            if isinstance(vector, list) and len(vector) >= 5:
                print(f"前5个值: {vector[:5]}")
    else:
        print(f"返回值不是列表: {embeddings}")

except Exception as e:
    print(f"错误: {e}")
    import traceback

    traceback.print_exc()