"""
这段代码用于全流程
 input： 待审查文档
 result： 文档是否合规的报告：①触犯条目 & 阴阳性 ②整体回复
"""
from tools.LLMs.chat_models import chat
from tools.LLMs.rerank_model import text_rerank
from tools.retriever.hybrid_retriever import HybridRetriever


# 先写一个方法，传入的是单个chunk文档
def get_single_chunk_result(chunk:str ,):
    # 先检索topm
    retriever = HybridRetriever()
    top_m = retriever.retrieve_with_detailed_scores(chunk , 20)

    # 对topm，改变格式为List[str]调用rerank模型
    list_top_m = [item['detailed_results'] for item in top_m]

    # 传入rerank模型，获取topk
    top_k = text_rerank(list_top_m,"",10)
    top_k_content = [item['document']['text'] for item in top_k]

    # 拼接topk作为prompt 给chat model输出 ①触犯条目 & 阴阳性 ②整体回复
    prompt = ""
    chat(prompt)


