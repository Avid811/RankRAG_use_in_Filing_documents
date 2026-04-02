import os
from typing import List, Union
import dashscope
from http import HTTPStatus
from dotenv import load_dotenv

from config.config import config

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")


def get_embedding_func(input_text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
    """
    获取单个或多个文本的embedding

    参数:
        input_text: 单个字符串或字符串列表

    返回:
        单个embedding列表或embedding列表的列表
    """
    if not api_key:
        raise ValueError("请设置DASHSCOPE_API_KEY环境变量")

    resp = dashscope.TextEmbedding.call(
        model=config.EMBEDDING_MODEL,
        input=input_text,
        api_key=api_key
    )

    if resp.status_code != HTTPStatus.OK:
        raise Exception(f"Embedding API调用失败: {resp.code} - {resp.message}")

    if isinstance(input_text, str):
        # 单个文本
        return resp.output["embeddings"][0]["embedding"]
    else:
        # 多个文本
        return [embedding["embedding"] for embedding in resp.output["embeddings"]]