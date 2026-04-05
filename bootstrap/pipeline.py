"""
这段代码用于全流程
 input： 待审查文档
 result： 文档是否合规的报告：①触犯条目 & 阴阳性 ②整体回复
"""



import os
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
def get_single_chunk_result(chunk:str ,):

    # 先检索topm
    retriever = HybridRetriever()
    top_m = retriever.retrieve_with_detailed_scores(chunk , 20).get('detailed_results')

    # 对topm，改变格式为List[str]调用rerank模型
    list_top_m = [item['content'] for item in top_m]

    # 拼接topm与query（原文）作为prompt 传入rerank模型，获取topk
    topm_template = get_jinjia_obj(r'D:\WORK\school\BI_YE_ARTICLE\RankRAG_use_in_Filing_documents\tools\prompts\rerank_model.j2')
    rerank_data = {
        'top_m':list_top_m,
        'part_article_content':chunk
    }
    rerank_prompt = topm_template.render(rerank_data)

    top_k = text_rerank(list_top_m,rerank_prompt,10)
    top_k_content = [item['document']['text'] for item in top_k]

    # 拼接topk作为prompt 给chat model输出 ①触犯条目 & 阴阳性 ②整体回复
    chat_template = get_jinjia_obj(r'D:\WORK\school\BI_YE_ARTICLE\RankRAG_use_in_Filing_documents\tools\prompts\chat_model.j2')
    chat_data = {
        'top_k':top_k_content,
        'part_article_content':chunk
    }
    chat_prompt = chat_template.render(chat_data)

    return chat('qwen-plus',chat_prompt)


def process_pdf_file(pdf_path):
    """处理单个PDF文件"""
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
    
    # 3. 处理每个chunk并保存结果
    pdf_name = os.path.basename(pdf_path)
    result_dir = r"D:\WORK\school\BI_YE_ARTICLE\RankRAG_use_in_Filing_documents\data\result"
    
    # 确保结果目录存在
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    output_file = os.path.join(result_dir, f"{os.path.splitext(pdf_name)[0]}.txt")
    
    print(f"开始处理：{pdf_name}，共{len(chunks)}个chunk")
    
    with open(output_file, "w", encoding="utf-8") as f:
        # 使用tqdm或简单的进度显示
        if tqdm:
            for i, chunk in enumerate(tqdm(chunks, desc=f"处理 {pdf_name}", unit="chunk")):
                result = get_single_chunk_result(chunk)
                result_str = str(result)
                # 只写入非"当前chunk未触犯任何法律法规"的结果
                if "当前chunk未触犯任何法律法规" not in result_str:
                    f.write(result_str + "\n\n")
        else:
            total_chunks = len(chunks)
            for i, chunk in enumerate(chunks):
                # 简单的进度显示
                progress = (i + 1) / total_chunks * 100
                print(f"处理 {pdf_name}: {i+1}/{total_chunks} ({progress:.1f}%)", end="\r")
                result = get_single_chunk_result(chunk)
                result_str = str(result)
                # 只写入非"当前chunk未触犯任何法律法规"的结果
                if "当前chunk未触犯任何法律法规" not in result_str:
                    f.write(result_str + "\n\n")
            print()  # 换行
    
    print(f"处理完成：{pdf_name}，结果保存到：{output_file}")


if __name__ == "__main__":
    input_dir = r"D:\WORK\school\BI_YE_ARTICLE\RankRAG_use_in_Filing_documents\data\input_data"
    
    # 获取所有PDF文件
    pdf_files = [file_name for file_name in os.listdir(input_dir) if file_name.lower().endswith('.pdf')]
    total_pdfs = len(pdf_files)
    
    print(f"发现{total_pdfs}个PDF文件待处理")
    
    # 遍历目录下的所有PDF文件
    if tqdm:
        for i, file_name in enumerate(tqdm(pdf_files, desc="处理PDF文件", unit="file")):
            pdf_path = os.path.join(input_dir, file_name)
            print(f"\n开始处理第{i+1}/{total_pdfs}个文件: {file_name}")
            process_pdf_file(pdf_path)
    else:
        for i, file_name in enumerate(pdf_files):
            pdf_path = os.path.join(input_dir, file_name)
            print(f"\n处理第{i+1}/{total_pdfs}个文件: {file_name}")
            process_pdf_file(pdf_path)
    
    print("\n所有PDF文件处理完成！")