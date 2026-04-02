"""
代码的作用是：
针对输入的专利文档，进行chunk切分（递归切分 + 滑动窗口兜底）
下一步是逐块调用 全流程pipeline
"""


import os
import re
from typing import List


class PatentChunker:
    def __init__(self, target_size: int = 500, base_window: int = 400, overlap: int = 100):
        self.target_size = target_size
        self.base_window = base_window
        self.overlap = overlap

        # 专利及常规文档的标题层级正则 (按优先级排序)
        # 使用正则表达式在这些标题前进行切分 (保留标题本身)
        self.heading_patterns = [
            r"\n\s*(?=\[\d{4}\])",  # 专利特有段落，如 [0001]
            r"\n\s*(?=[一二三四五六七八九十百千万]+[、.])",  # 一、 或 一.
            r"\n\s*(?=[（\(][一二三四五六七八九十百千万]+[）\)])",  # （一）
            r"\n\s*(?=\d+[、.])",  # 1. 或 1、
            r"\n\s*(?=[（\(]\d+[）\)])",  # (1)
            r"\n\s*(?=[①②③④⑤⑥⑦⑧⑨⑩])"  # ①
        ]

        # 用于语义切分的标点符号 (优先级：换行 > 句号 > 叹号/问号 > 分号)
        self.punctuations = ['\n', '。', '！', '？', '；']

    def process_file(self, file_path: str) -> List[str]:
        """读取文件并输出 chunks"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"未找到文件: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # 1. 递归切分
        raw_chunks = self._recursive_split(text, level=0)

        # 2. 尾盘处理：合并过小的碎片
        final_chunks = self._post_process_chunks(raw_chunks)

        return final_chunks

    def _recursive_split(self, text: str, level: int) -> List[str]:
        text = text.strip()
        if not text:
            return []

        # 1. 满足大小，直接成块
        if len(text) <= self.target_size:
            return [text]

        # 2. 还有可用的标题层级，继续往下递归
        if level < len(self.heading_patterns):
            pattern = self.heading_patterns[level]
            # 按标题切分
            splits = re.split(pattern, text)
            splits = [s.strip() for s in splits if s.strip()]

            # 如果该层级没切出新东西（只有一个块），去下一个层级试试
            if len(splits) <= 1:
                return self._recursive_split(text, level + 1)

            # 核心优化：合并过小的碎片，防止产出太小的 Chunk
            merged_splits = self._merge_small_splits(splits)

            final_chunks = []
            for split_text in merged_splits:
                if len(split_text) > self.target_size:
                    # 依然超大，递归下一层级
                    final_chunks.extend(self._recursive_split(split_text, level + 1))
                else:
                    final_chunks.append(split_text)
            return final_chunks

        # 3. 标题层级用尽，依然超大，启用滑动窗口兜底
        else:
            return self._semantic_window_split(text)

    def _merge_small_splits(self, splits: List[str]) -> List[str]:
        """在按标题切分时，将过小的文本块合并，逼近 target_size"""
        merged = []
        current_chunk = ""

        for split in splits:
            # +1 是为了补偿合并时的换行符
            if len(current_chunk) + len(split) + 1 <= self.target_size:
                current_chunk = current_chunk + "\n" + split if current_chunk else split
            else:
                if current_chunk:
                    merged.append(current_chunk)
                current_chunk = split

        if current_chunk:
            merged.append(current_chunk)

        return merged

    def _semantic_window_split(self, text: str) -> List[str]:
        """按照基础窗口 + 重叠窗口切分，并在重叠区就近找句号"""
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.base_window + self.overlap

            if end >= text_len:
                chunks.append(text[start:])
                break

            # 1. 确定当前 Chunk 的实际结束点 (在 overlap 区域内找标点)
            overlap_text = text[start + self.base_window: end]
            match_idx = -1

            for p in self.punctuations:
                idx = overlap_text.rfind(p)  # 从后往前找，尽量保证 chunk 更大
                if idx != -1:
                    match_idx = max(match_idx, idx)

            if match_idx != -1:
                actual_end = start + self.base_window + match_idx + 1
            else:
                actual_end = end  # 找不到标点，硬切

            chunks.append(text[start:actual_end])

            # 2. 确定下一个 Chunk 的起点 (往回退 overlap，并在附近找标点对齐句子)
            next_start_guess = actual_end - self.overlap
            if next_start_guess <= start:
                next_start = actual_end  # 防止死循环
            else:
                look_ahead_text = text[next_start_guess:actual_end]
                start_match_idx = float('inf')

                for p in self.punctuations:
                    idx = look_ahead_text.find(p)  # 从前往后找，快速锁定句首
                    if idx != -1:
                        start_match_idx = min(start_match_idx, idx)

                if start_match_idx != float('inf'):
                    next_start = next_start_guess + start_match_idx + 1
                else:
                    next_start = next_start_guess

            start = next_start

        return chunks

    def _post_process_chunks(self, chunks: List[str]) -> List[str]:
        """最终遍历扫描：处理过小的 chunk，合并到前后较短的 chunk 中"""
        if len(chunks) <= 1:
            return chunks

        # 定义阈值：chunk_size * 20%
        min_size = self.target_size * 0.2

        i = 0
        while i < len(chunks):
            # 如果当前 chunk 太小，且列表中还有其他块可以合并
            if len(chunks[i]) < min_size and len(chunks) > 1:
                if i == 0:
                    # 第一个块：只能合并到后一个块
                    chunks[1] = chunks[0] + "\n" + chunks[1]
                    chunks.pop(0)
                    # pop 后，原本的 chunks[1] 变成新的 chunks[0]，继续在 i=0 处检查
                elif i == len(chunks) - 1:
                    # 最后一个块：只能合并到前一个块
                    chunks[i - 1] = chunks[i - 1] + "\n" + chunks[i]
                    chunks.pop(i)
                    # pop 后已经到达列表末尾，循环自然结束
                else:
                    # 中间的块：比较前后块的长度，合并到较短的那一个
                    prev_len = len(chunks[i - 1])
                    next_len = len(chunks[i + 1])

                    if prev_len <= next_len:
                        # 前面的块更短，合并到前面
                        chunks[i - 1] = chunks[i - 1] + "\n" + chunks[i]
                        chunks.pop(i)
                        # 注意：这里不需要 i += 1，因为 pop(i) 后，后面的元素补位到了索引 i
                    else:
                        # 后面的块更短，合并到后面
                        chunks[i + 1] = chunks[i] + "\n" + chunks[i + 1]
                        chunks.pop(i)
            else:
                # 当前块大小达标，检查下一个
                i += 1

        return chunks


if __name__ == "__main__":
    # 执行切分
    file_path = r"D:\WORK\school\BI_YE_ARTICLE\RankRAG_use_in_Filing_documents\data\input_data\demo专利.txt"

    chunker = PatentChunker(target_size=500, base_window=400, overlap=100)

    try:
        chunks = chunker.process_file(file_path)
        print(f"切分完成，共获得 {len(chunks)} 个 Chunks。\n")

        # 打印验证结果
        for i, chunk in enumerate(chunks):
            print(f"--- Chunk {i + 1} (长度: {len(chunk)}) ---")
            print(chunk)
            print("-" * 40 + "\n")

    except Exception as e:
        print(f"程序执行出错: {e}")