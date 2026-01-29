import os
from client.es_client import ElasticsearchClient
import glob

from tools.processor.document_processor import DocumentProcessor


class KnowledgeBaseLoader:
    def __init__(self):
        self.processor = DocumentProcessor()
        self.es_client = ElasticsearchClient()

    def load_from_directory(self, directory_path: str):
        """从目录加载所有文档"""
        all_documents = []

        # 支持的文件格式
        patterns = ['*.txt', '*.pdf', '*.docx']

        for pattern in patterns:
            files = glob.glob(os.path.join(directory_path, pattern))
            for file_path in files:
                try:
                    metadata = {
                        'source': os.path.basename(file_path),
                        'file_path': file_path
                    }

                    documents = self.processor.process_document_file(
                        file_path,
                        metadata
                    )
                    all_documents.extend(documents)

                except Exception as e:
                    print(f"处理文件 {file_path} 失败: {e}")

        return all_documents

    def initialize_knowledge_base(self, data_dir: str):
        """初始化知识库"""
        print("创建Elasticsearch索引...")
        self.es_client.create_index()

        print("加载和处理文档...")
        documents = self.load_from_directory(data_dir)

        print(f"准备索引 {len(documents)} 个文档...")
        self.es_client.index_documents(documents)

        print("知识库初始化完成！")