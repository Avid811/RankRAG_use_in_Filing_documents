import os
import time
import glob
from client.es_client import ElasticsearchClient
from tools.processor.document_processor import DocumentProcessor


class KnowledgeBaseLoader:
    def __init__(self, es_client=None):
        self.processor = DocumentProcessor()
        self.es_client = es_client or ElasticsearchClient()

    def load_from_file(self, file_path: str):
        """从单个文件加载文档"""
        documents = []

        if not os.path.exists(file_path):
            print(f"❌ 文件不存在: {file_path}")
            return documents

        try:
            print(f"📄 处理文件: {os.path.basename(file_path)}")

            metadata = {
                'source': os.path.basename(file_path),
                'file_path': file_path
            }

            documents = self.processor.process_document_file(file_path, metadata)
            print(f"✅ 生成 {len(documents)} 个chunks")

        except Exception as e:
            print(f"❌ 处理文件失败: {e}")
            import traceback
            traceback.print_exc()

        return documents

    def load_from_directory(self, directory_path: str):
        """从目录加载所有文档"""
        all_documents = []

        if not os.path.exists(directory_path):
            print(f"❌ 目录不存在: {directory_path}")
            return all_documents

        # 检查是否是文件
        if os.path.isfile(directory_path):
            print(f"⚠️  传入的是文件路径，自动转换为目录处理")
            return self.load_from_file(directory_path)

        # 支持的文件格式
        patterns = ['*.txt', '*.pdf', '*.docx']

        for pattern in patterns:
            files = glob.glob(os.path.join(directory_path, pattern))
            for file_path in files:
                try:
                    docs = self.load_from_file(file_path)
                    all_documents.extend(docs)
                except Exception as e:
                    print(f"处理文件 {file_path} 失败: {e}")

        return all_documents

    def initialize_knowledge_base(self, data_path: str, recreate_index=False):
        """初始化知识库，支持文件和目录

        Args:
            data_path: 文件路径或目录路径
            recreate_index: 是否重新创建索引
        """
        print("=" * 60)
        print("知识库初始化")
        print("=" * 60)

        # 检查路径
        if not os.path.exists(data_path):
            print(f"路径不存在: {data_path}")
            return False

        # 检查连接
        if not self.es_client.check_connection():
            print("Elasticsearch 连接失败")
            return False

        # 处理索引
        if recreate_index:
            print("删除旧索引...")
            self.es_client.delete_index()

        print("创建/检查索引...")
        if not self.es_client.create_index(embedding_dim=1024):
            print("索引创建失败")
            return False

        # 加载文档
        print("\n加载文档...")
        if os.path.isfile(data_path):
            print(f"从文件加载: {os.path.basename(data_path)}")
        else:
            print(f"从目录加载: {data_path}")

        documents = self.load_from_directory(data_path)

        if not documents:
            print("没有找到任何文档")
            return False

        print(f"共处理 {len(documents)} 个文档chunks")

        # 转换格式，把基于langchain工具切分的 chunks 转换成 es可以存储的分块
        print("转换文档格式...")
        es_documents = []

        for i, doc in enumerate(documents):
            # 处理不同的文档格式
            if isinstance(doc, dict):
                # 字典格式
                content = doc.get('content', '')
                metadata = doc.get('metadata', {})
                if not isinstance(metadata, dict):
                    metadata = {}

                # 添加chunk_index到metadata
                metadata = metadata.copy()
                metadata['chunk_index'] = doc.get('chunk_index', i)

                # 获取embedding
                embedding = doc.get('embedding')
                if embedding is None:
                    import random
                    embedding = [random.random() for _ in range(1024)]

            else:
                print(f"跳过不支持格式的文档: {type(doc)}")
                continue

            es_doc = {
                'content': content,
                'metadata': metadata,
                'embedding': embedding,
                'created_at': int(time.time() * 1000)
            }
            es_documents.append(es_doc)

            # 显示进度
            if (i + 1) % 10 == 0 or i + 1 == len(documents):
                print(f"  → 已转换 {i + 1}/{len(documents)} 个文档")

        if not es_documents:
            print("转换后没有可用的文档")
            return False

        # 索引到ES
        print(f"\n正在索引 {len(es_documents)} 个文档到ES...")
        start_time = time.time()

        success, failed = self.es_client.index_documents(es_documents)

        elapsed = time.time() - start_time
        print(f"\n📊 索引结果:")
        print(f"  ✅ 成功: {success} 个文档")
        print(f"  ❌ 失败: {failed} 个文档")
        print(f"  ⏱️  耗时: {elapsed:.2f}秒")
        print(f"  🚀 速度: {success / elapsed:.1f} docs/sec")

        if success > 0:
            # 验证
            final_count = self.es_client.get_document_count()
            print(f"  📄 ES中文档数: {final_count}")
            return True
        else:
            return False