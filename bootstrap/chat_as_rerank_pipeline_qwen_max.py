"""
这段代码用于全流程
 input： 待审查文档
 result： 文档是否合规的报告：①触犯条目 & 阴阳性 ②整体回复
"""



import os
import json
from tools.LLMs.chat_models import chat
from tools.LLMs.rerank_model import text_rerank
from tools.prompts.jinjia_obj_factory import get_jinjia_obj
from tools.retriever.hybrid_retriever import HybridRetriever
from tools.processor.doc_to_chunk_by_recursion import PatentChunker
from tools.processor.document_processor import DocumentProcessor
from jinja2 import Template

# 尝试导入tqdm用于进度条，如果没有安装则使用简单的进度显示
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# 先写一个方法，传入的是单个chunk文档
def get_single_chunk_result(chunk:str ,top_k:int = 10):

    # 先检索topm
    retriever = HybridRetriever()
    top_m = retriever.retrieve_with_detailed_scores(chunk , 20).get('detailed_results')

    # 对topm，改变格式为List[str]并创建map
    list_top_m = [item['content'] for item in top_m]
    
    # 拼接topm与query（原文）作为prompt 传入chat_as_rerank模型，获取topk索引
    chat_rerank_template = get_jinjia_obj(r'D:\WORK\school\BI_YE_ARTICLE\RankRAG_use_in_Filing_documents\tools\prompts\chat_as_rerank.j2')
    chat_rerank_data = {
        'top_m':list_top_m,
        'part_article_content':chunk,
        'top_k':top_k
    }
    chat_rerank_prompt = chat_rerank_template.render(chat_rerank_data)

    # 调用chat模型进行重排序
    rerank_result = chat('qwen3-max', chat_rerank_prompt,asRerank=True)

    
    # 解析chat模型输出的索引列表
    try:
        # 提取方括号内的内容并转换为列表
        import re
        match = re.search(r'\[(.*?)\]', rerank_result)
        if match:
            index_str = match.group(1)
            # 处理可能的空格和换行
            index_str = index_str.replace('\n', '').replace(' ', '')
            # 转换为整数列表
            ranked_indices = [int(idx) for idx in index_str.split(',') if idx.strip()]
            # 确保索引在有效范围内
            ranked_indices = [idx for idx in ranked_indices if 0 <= idx < len(list_top_m)]
            # 取前10个
            ranked_indices = ranked_indices[:10]
            # 根据索引获取排序后的top_k内容
            top_k_content = [list_top_m[idx] for idx in ranked_indices]


        else:
            # 如果解析失败，使用原始的top_m作为top_k
            top_k_content = list_top_m[:10]
    except Exception as e:
        print(f"解析重排序结果失败: {e}")
        # 如果解析失败，使用原始的top_m作为top_k
        top_k_content = list_top_m[:10]

    # 拼接topk作为prompt 给chat model输出 ①触犯条目 & 阴阳性 ②整体回复
    chat_template = get_jinjia_obj(r'D:\WORK\school\BI_YE_ARTICLE\RankRAG_use_in_Filing_documents\tools\prompts\chat_model.j2')
    chat_data = {
        'top_k':top_k_content,
        'part_article_content':chunk
    }
    chat_prompt = chat_template.render(chat_data)

    return chat('qwen3-max',chat_prompt,asRerank=False)


def process_pdf_file(pdf_path):
    """处理单个PDF文件并返回结果"""
    # 1. 读取PDF并转换为文本
    processor = DocumentProcessor()
    text = processor.load_document(pdf_path)
    
    # 2. 使用PatentChunker切分文本
    chunker = PatentChunker(target_size=500, base_window=400, overlap=100)
    
    # 临时保存文本到文件，以便使用PatentChunker的process_file方法
    temp_txt_path = pdf_path.replace('.pdf', '.temp.txt')
    with open(temp_txt_path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    chunks = chunker.process_file(temp_txt_path)
    
    # 删除临时文件
    os.remove(temp_txt_path)
    
    # 3. 处理每个chunk并收集结果
    pdf_name = os.path.basename(pdf_path)
    results = []
    
    print(f"开始处理：{pdf_name}，共{len(chunks)}个chunk")
    
    # 使用tqdm或简单的进度显示
    if tqdm:
        for i, chunk in enumerate(tqdm(chunks, desc=f"处理 {pdf_name}", unit="chunk")):
            result = get_single_chunk_result(chunk)
            result_str = str(result)
            # 只收集非"当前chunk未触犯任何法律法规"的结果
            if "当前chunk未触犯任何法律法规" not in result_str:
                # 检查是否包含模板路径（LLM幻觉）
                if "tools\\prompts\\chat_model.j2" in result_str:
                    print(f"警告：{pdf_name} 的结果包含模板路径，可能是LLM幻觉，跳过该结果")
                    continue
                try:
                    # 尝试解析结果为JSON
                    result_json = json.loads(result_str)
                    # 确保结果是列表格式
                    if isinstance(result_json, list):
                        results.extend(result_json)
                    else:
                        print(f"警告：{pdf_name} 的结果不是列表格式，跳过该结果")
                except json.JSONDecodeError:
                    print(f"警告：无法解析 {pdf_name} 的结果为JSON，跳过该结果")
    else:
        total_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            # 简单的进度显示
            progress = (i + 1) / total_chunks * 100
            print(f"处理 {pdf_name}: {i+1}/{total_chunks} ({progress:.1f}%)", end="\r")
            result = get_single_chunk_result(chunk)
            result_str = str(result)
            # 只收集非"当前chunk未触犯任何法律法规"的结果
            if "当前chunk未触犯任何法律法规" not in result_str:
                # 检查是否包含模板路径（LLM幻觉）
                if "tools\\prompts\\chat_model.j2" in result_str:
                    print(f"警告：{pdf_name} 的结果包含模板路径，可能是LLM幻觉，跳过该结果")
                    continue
                try:
                    # 尝试解析结果为JSON
                    result_json = json.loads(result_str)
                    # 确保结果是列表格式
                    if isinstance(result_json, list):
                        results.extend(result_json)
                    else:
                        print(f"警告：{pdf_name} 的结果不是列表格式，跳过该结果")
                except json.JSONDecodeError:
                    print(f"警告：无法解析 {pdf_name} 的结果为JSON，跳过该结果")
        print()  # 换行
    
    # 保存单个PDF的结果
    result_dir = r"D:\WORK\school\BI_YE_ARTICLE\RankRAG_use_in_Filing_documents\data\result\chatAsReRank_pipeline"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)
    
    output_file = os.path.join(result_dir, f"{os.path.splitext(pdf_name)[0]}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成：{pdf_name}，共收集到 {len(results)} 个违规结果")
    print(f"单个PDF结果已保存到：{output_file}")
    return results


def find_all_pdf_files(directory):
    """递归查找目录下所有PDF文件"""
    pdf_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                pdf_files.append(pdf_path)
    return pdf_files


def process_single_pdf(pdf_path):
    """单独处理一个PDF文件
    
    Args:
        pdf_path: PDF文件的绝对路径
    
    Returns:
        list: 处理结果的JSON数组
    """
    if not os.path.exists(pdf_path):
        print(f"错误：文件不存在：{pdf_path}")
        return []
    
    if not pdf_path.lower().endswith('.pdf'):
        print(f"错误：不是PDF文件：{pdf_path}")
        return []
    
    print(f"开始处理单个PDF文件：{os.path.basename(pdf_path)}")
    results = process_pdf_file(pdf_path)
    
    print(f"单个PDF文件处理完成！")
    print(f"共收集到 {len(results)} 个违规结果")
    return results


if __name__ == "__main__":
    # 选项1：处理input_data目录下所有文件夹中的PDF文件
    input_dir = r"D:\WORK\school\BI_YE_ARTICLE\RankRAG_use_in_Filing_documents\data\input_data"
    
    # 递归查找所有PDF文件
    pdf_files = find_all_pdf_files(input_dir)
    total_pdfs = len(pdf_files)
    
    print(f"发现{total_pdfs}个PDF文件待处理")
    
    # 收集所有结果
    all_results = []
    
    # 遍历所有PDF文件
    if tqdm:
        for i, pdf_path in enumerate(tqdm(pdf_files, desc="处理PDF文件", unit="file")):
            file_name = os.path.basename(pdf_path)
            print(f"\n开始处理第{i+1}/{total_pdfs}个文件: {file_name}")
            pdf_results = process_pdf_file(pdf_path)
            all_results.extend(pdf_results)
    else:
        for i, pdf_path in enumerate(pdf_files):
            file_name = os.path.basename(pdf_path)
            print(f"\n处理第{i+1}/{total_pdfs}个文件: {file_name}")
            pdf_results = process_pdf_file(pdf_path)
            all_results.extend(pdf_results)
    
    # 保存合并后的结果
    result_dir = r"D:\WORK\school\BI_YE_ARTICLE\RankRAG_use_in_Filing_documents\data\result\chatAsReRank_pipeline_qwen_max"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)
    
    output_file = os.path.join(result_dir, "merged_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n所有PDF文件处理完成！")
    print(f"合并后的结果已保存到：{output_file}")
    print(f"共收集到 {len(all_results)} 个违规结果")
    
    # # 选项2：单独处理一个PDF文件（示例）
    # # 取消下面的注释并修改为实际的PDF文件路径
    # single_pdf_path = r"D:\WORK\school\BI_YE_ARTICLE\RankRAG_use_in_Filing_documents\data\input_data\阿里健康医学内容生成式算法备案_专利\多模态大模型的训练方法及相关装置.pdf"
    # process_single_pdf(single_pdf_path)