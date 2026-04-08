import os
import threading
import importlib.util
import time

# 定义要运行的pipeline文件
pipelines = [
    'chat_as_rerank_pipeline_qwen_flash',
    'chat_as_rerank_pipeline_qwen_max',
    'chat_as_rerank_pipeline_qwen_plus',
    'normal_rerank_pipeline',
    'only_search_pipeline'
]

def run_pipeline(pipeline_name):
    """运行单个pipeline"""
    print(f"开始运行：{pipeline_name}.py")
    try:
        # 动态导入pipeline模块
        current_dir = os.path.dirname(os.path.abspath(__file__))
        spec = importlib.util.spec_from_file_location(pipeline_name, os.path.join(current_dir, f"{pipeline_name}.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # 检查模块是否有main函数
        if hasattr(module, 'main'):
            module.main()
        else:
            # 如果没有main函数，执行原来的main部分逻辑
            print(f"{pipeline_name} 没有main函数，执行默认逻辑")
            # 这里可以添加默认逻辑，或者直接执行模块的__main__部分
            # 由于模块已经导入，其__name__不是__main__，所以需要手动执行
            input_dir = r"D:\WORK\school\BI_YE_ARTICLE\RankRAG_use_in_Filing_documents\data\input_data"
            
            # 递归查找所有PDF文件
            pdf_files = module.find_all_pdf_files(input_dir)
            total_pdfs = len(pdf_files)
            
            print(f"发现{total_pdfs}个PDF文件待处理")
            
            # 收集所有结果
            all_results = []
            
            # 遍历所有PDF文件
            if hasattr(module, 'tqdm') and module.tqdm:
                for i, pdf_path in enumerate(module.tqdm(pdf_files, desc="处理PDF文件", unit="file")):
                    file_name = os.path.basename(pdf_path)
                    print(f"\n开始处理第{i+1}/{total_pdfs}个文件: {file_name}")
                    pdf_results = module.process_pdf_file(pdf_path)
                    all_results.extend(pdf_results)
            else:
                for i, pdf_path in enumerate(pdf_files):
                    file_name = os.path.basename(pdf_path)
                    print(f"\n处理第{i+1}/{total_pdfs}个文件: {file_name}")
                    pdf_results = module.process_pdf_file(pdf_path)
                    all_results.extend(pdf_results)
            
            # 保存合并后的结果
            # 根据pipeline名称确定结果目录
            if 'chat_as_rerank_pipeline_qwen_flash' in pipeline_name:
                result_dir = r"D:\WORK\school\BI_YE_ARTICLE\RankRAG_use_in_Filing_documents\data\result\chatAsReRank_pipeline_qwen_flash"
            elif 'chat_as_rerank_pipeline_qwen_max' in pipeline_name:
                result_dir = r"D:\WORK\school\BI_YE_ARTICLE\RankRAG_use_in_Filing_documents\data\result\chatAsReRank_pipeline_qwen_max"
            elif 'chat_as_rerank_pipeline_qwen_plus' in pipeline_name:
                result_dir = r"D:\WORK\school\BI_YE_ARTICLE\RankRAG_use_in_Filing_documents\data\result\chatAsReRank_pipeline_qwen_plus"
            elif 'normal_rerank_pipeline' in pipeline_name:
                result_dir = r"D:\WORK\school\BI_YE_ARTICLE\RankRAG_use_in_Filing_documents\data\result\normal_rerank_pipeline"
            elif 'only_search_pipeline' in pipeline_name:
                result_dir = r"D:\WORK\school\BI_YE_ARTICLE\RankRAG_use_in_Filing_documents\data\result\only_search_pipeline"
            else:
                result_dir = r"D:\WORK\school\BI_YE_ARTICLE\RankRAG_use_in_Filing_documents\data\result"
            
            if not os.path.exists(result_dir):
                os.makedirs(result_dir, exist_ok=True)
            
            output_file = os.path.join(result_dir, "merged_results.json")
            with open(output_file, "w", encoding="utf-8") as f:
                import json
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            
            print(f"\n所有PDF文件处理完成！")
            print(f"合并后的结果已保存到：{output_file}")
            print(f"共收集到 {len(all_results)} 个违规结果")
            
    except Exception as e:
        print(f"运行 {pipeline_name} 时出错：{e}")
        import traceback
        traceback.print_exc()

def main():
    print("开始运行所有pipeline...")
    print("=" * 60)
    
    # 创建线程
    threads = []
    for pipeline in pipelines:
        t = threading.Thread(target=run_pipeline, args=(pipeline,))
        threads.append(t)
        t.start()
        # 稍微延迟一下，避免同时启动过多进程
        time.sleep(1)
    
    # 等待所有线程完成
    for t in threads:
        t.join()
    
    print("=" * 60)
    print("所有pipeline运行完成！")

if __name__ == "__main__":
    main()
