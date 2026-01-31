import os
import re
from datetime import datetime
from typing import List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF
from docx import Document

from config.config import config
from tools.processor.get_embeddings import get_embedding_func



class DocumentProcessor:
    def __init__(self):

        # 初始化不同的文本分割器
        self.paragraph_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "\r\n"],
            chunk_size=config.CHUNK_SIZE * 2,  # 设置大一些，主要用于分段
            chunk_overlap=0
        )

        # 用于最终回退的分割器
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["。", "！", "？", "；", "\n", "，", "、", " ", ""]
        )

    def load_document(self, file_path: str) -> str:
        """加载文档"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        _, ext = os.path.splitext(file_path)

        if ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()

        elif ext == '.pdf':
            text = ""
            pdf_document = fitz.open(file_path)
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                text += page.get_text()
            return text

        elif ext == '.docx':
            doc = Document(file_path)
            text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
            return text

        else:
            raise ValueError(f"不支持的文件格式: {ext}")

    def split_by_paragraphs(self, text: str) -> List[str]:
        """第一步：按段落分割"""
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        return paragraphs

    def split_by_legal_structure(self, text: str) -> List[str]:
        """第二步：按法律结构分割（第X条、第X款等）"""
        # 定义法律结构模式
        patterns = [
            r'(第[零一二三四五六七八九十百千万\d]+条\s*[^\s])',  # 第X条
            r'(第[零一二三四五六七八九十百千万\d]+款\s*[^\s])',  # 第X款
            r'(第[零一二三四五六七八九十百千万\d]+项\s*[^\s])',  # 第X项
            r'(第[零一二三四五六七八九十百千万\d]+章\s*[^\s])',  # 第X章
            r'(第[零一二三四五六七八九十百千万\d]+节\s*[^\s])',  # 第X节
            r'((?:[一二三四五六七八九十]、|[0-9]+\.)\s*[^\s])',  # 一、 或 1.
        ]

        combined_pattern = '|'.join(patterns)
        chunks = []
        start = 0

        # 查找所有匹配位置
        matches = list(re.finditer(combined_pattern, text))

        if not matches:
            return [text]

        for i, match in enumerate(matches):
            if i > 0:
                # 获取前一个匹配到当前匹配之间的内容
                chunk = text[start:match.start()]
                if chunk.strip():
                    chunks.append(chunk)
            start = match.start()

        # 添加最后一个块
        last_chunk = text[start:]
        if last_chunk.strip():
            chunks.append(last_chunk)

        return chunks

    def split_by_sentences(self, text: str) -> List[str]:
        """第三步：按句子分割（使用回退分割器）"""
        if not text.strip():
            return []

        # 使用回退分割器
        chunks = self.fallback_splitter.split_text(text)
        return chunks

    def hierarchical_split(self, text: str) -> List[str]:
        """多层次分割：段落 -> 法律结构 -> 句子"""
        all_chunks = []

        # 第一步：按段落分割
        paragraphs = self.split_by_paragraphs(text)

        for paragraph in paragraphs:
            # 如果段落长度小于CHUNK_SIZE，直接作为一个chunk
            if len(paragraph) <= config.CHUNK_SIZE:
                all_chunks.append(paragraph)
                continue

            # 第二步：尝试按法律结构分割
            legal_chunks = self.split_by_legal_structure(paragraph)

            # 如果按法律结构分割产生了多个块
            if len(legal_chunks) > 1:
                for chunk in legal_chunks:
                    if chunk.strip():
                        all_chunks.append(chunk.strip())
                continue

            # 第三步：如果段落没有法律结构，按句子分割
            if legal_chunks and len(legal_chunks[0]) > config.CHUNK_SIZE:
                # 段落仍然太长，需要进一步分割
                sentence_chunks = self.split_by_sentences(legal_chunks[0])
                all_chunks.extend([chunk.strip() for chunk in sentence_chunks if chunk.strip()])
            else:
                # 段落长度合适
                for chunk in legal_chunks:
                    if chunk.strip():
                        all_chunks.append(chunk.strip())

        return all_chunks

    def split_document(self, text: str, metadata: Dict = None) -> List[Dict]:
        """切分文档为chunks（使用多层次分割策略）"""
        chunks = self.hierarchical_split(text)

        documents = []
        for i, chunk in enumerate(chunks):
            doc = {
                'content': chunk,
                'metadata': metadata or {},
                'chunk_index': i + 1
            }
            documents.append(doc)

        return documents

    def generate_embeddings(self, documents: List[Dict]) -> List[Dict]:
        """生成文档的embedding（分批处理）"""
        contents = [doc['content'] for doc in documents]

        # 分批处理，每批最大10条（根据API限制）
        batch_size = 10
        embeddings = []

        print(f"开始生成embedding，共{len(contents)}条，分批处理，每批{batch_size}条")

        for i in range(0, len(contents), batch_size):
            batch_contents = contents[i:i + batch_size]
            batch_docs = documents[i:i + batch_size]

            try:
                print(f"处理第{i // batch_size + 1}批，共{batch_size}条" if i + batch_size < len(
                    contents) else f"处理最后一批，共{len(contents) - i}条")
                batch_embeddings = get_embedding_func(batch_contents)

                if len(batch_embeddings) == len(batch_contents):
                    embeddings.extend(batch_embeddings)
                    print(f"第{i // batch_size + 1}批处理成功")
                else:
                    print(f"警告：第{i // batch_size + 1}批返回的embedding数量不匹配")
                    # 如果批量失败，回退到单条处理
                    for j, content in enumerate(batch_contents):
                        try:
                            embedding = get_embedding_func(content)
                            embeddings.append(embedding)
                            print(f"  单条处理第{i + j + 1}条成功")
                        except Exception as e:
                            print(f"  第{i + j + 1}条处理失败: {e}")
                            embeddings.append([])
            except Exception as e:
                print(f"第{i // batch_size + 1}批处理失败: {e}")
                print("回退到单条处理...")
                # 批次失败，逐条处理
                for j, content in enumerate(batch_contents):
                    try:
                        embedding = get_embedding_func(content)
                        embeddings.append(embedding)
                        print(f"  单条处理第{i + j + 1}条成功")
                    except Exception as e2:
                        print(f"  第{i + j + 1}条处理失败: {e2}")
                        embeddings.append([])

        # 将embedding添加到文档
        if len(embeddings) != len(documents):
            print(f"警告：生成的embedding数量({len(embeddings)})与文档数量({len(documents)})不匹配")
            # 确保长度一致
            embeddings = embeddings[:len(documents)] if len(embeddings) > len(documents) else embeddings + [[]] * (
                        len(documents) - len(embeddings))

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            if embedding:  # 确保embedding不为空
                doc['embedding'] = embedding
            else:
                print(f"警告: 第 {i + 1} 个文档的embedding为空")
                doc['embedding'] = []

        return documents

    def process_document_file(self, file_path: str, metadata: Dict = None) -> List[Dict]:
        """处理单个文档文件"""
        print(f"处理文档: {file_path}")

        # 加载文档
        text = self.load_document(file_path)

        # 准备元数据
        if metadata is None:
            metadata = {
                'source': os.path.basename(file_path),
                'page': 1
            }

        # 切分文档
        chunks = self.split_document(text, metadata)

        print(f"切分结果：\n{chunks}")

        # 生成embedding
        documents_with_embeddings = self.generate_embeddings(chunks)

        print(f"生成 {len(documents_with_embeddings)} 个chunks")
        return documents_with_embeddings

    def save_results(self, documents: List[Dict], output_path: str = None):
        """保存处理结果到文件"""
        import json

        if output_path is None:
            output_path = f"processed_documents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # 简化数据以便保存（embedding可能很大）
        simplified_docs = []
        for doc in documents:
            simplified_doc = {
                'content': doc['content'],
                'metadata': doc['metadata'],
                'chunk_index': doc['chunk_index'],
                'has_embedding': 'embedding' in doc and bool(doc['embedding'])
            }
            if 'embedding' in doc and doc['embedding']:
                simplified_doc['embedding_dim'] = len(doc['embedding'])
            simplified_docs.append(simplified_doc)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(simplified_docs, f, ensure_ascii=False, indent=2)

        print(f"结果已保存到: {output_path}")

        # 同时保存完整的embedding到另一个文件
        embedding_data = []
        for doc in documents:
            if 'embedding' in doc and doc['embedding']:
                embedding_data.append({
                    'chunk_index': doc['chunk_index'],
                    'embedding': doc['embedding']
                })

        if embedding_data:
            embedding_path = output_path.replace('.json', '_embeddings.json')
            with open(embedding_path, 'w', encoding='utf-8') as f:
                json.dump(embedding_data, f, ensure_ascii=False)
            print(f"Embedding数据已保存到: {embedding_path}")


