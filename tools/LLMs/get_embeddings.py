import os
import time
from typing import List, Union, Optional
import dashscope
from http import HTTPStatus
from dotenv import load_dotenv
import logging
from config.config import config
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

api_key = os.getenv("DASHSCOPE_API_KEY")

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局变量用于控制请求频率
_last_call_time = 0
_MIN_CALL_INTERVAL = 0.1  # 最小调用间隔100毫秒
_MAX_BATCH_SIZE = 25  # 根据Dashscope API限制设置合适的批次大小


class EmbeddingAPIError(Exception):
    """自定义API错误"""
    pass


class RateLimitError(EmbeddingAPIError):
    """频率限制错误"""
    pass


class APIQuotaError(EmbeddingAPIError):
    """API配额错误"""
    pass


@retry(
    stop=stop_after_attempt(3),  # 最多重试3次
    wait=wait_exponential(multiplier=1, min=1, max=10),  # 指数退避
    retry=retry_if_exception_type((RateLimitError, Exception)),  # 针对特定错误重试
    reraise=True
)
def get_embedding_func(input_text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
    """
    获取单个或多个文本的embedding，包含重试机制和频率控制
    
    参数:
        input_text: 单个字符串或字符串列表
    
    返回:
        单个embedding列表或embedding列表的列表
    """
    global _last_call_time
    
    if not api_key:
        raise ValueError("请设置DASHSCOPE_API_KEY环境变量")
    
    # 控制调用频率
    current_time = time.time()
    time_since_last_call = current_time - _last_call_time
    if time_since_last_call < _MIN_CALL_INTERVAL:
        sleep_time = _MIN_CALL_INTERVAL - time_since_last_call
        time.sleep(sleep_time)
    
    try:
        _last_call_time = time.time()
        
        resp = dashscope.TextEmbedding.call(
            model=config.EMBEDDING_MODEL,
            input=input_text,
            api_key=api_key
        )
        
        if resp.status_code == HTTPStatus.TOO_MANY_REQUESTS:
            logger.warning(f"频率限制触发，状态码: {resp.status_code}")
            raise RateLimitError(f"API频率限制: {resp.code} - {resp.message}")
        
        if resp.status_code == HTTPStatus.FORBIDDEN:
            logger.error(f"API权限或配额问题，状态码: {resp.status_code}")
            raise APIQuotaError(f"API权限或配额不足: {resp.code} - {resp.message}")
        
        if resp.status_code != HTTPStatus.OK:
            logger.error(f"API调用失败，状态码: {resp.status_code}, 错误: {resp.code} - {resp.message}")
            raise EmbeddingAPIError(f"Embedding API调用失败: {resp.code} - {resp.message}")
        
        if isinstance(input_text, str):
            return resp.output["embeddings"][0]["embedding"]
        else:
            return [embedding["embedding"] for embedding in resp.output["embeddings"]]
            
    except Exception as e:
        logger.error(f"获取embedding时发生错误: {str(e)}")
        raise


def get_batch_embeddings(
    texts: List[str], 
    batch_size: int = _MAX_BATCH_SIZE,
    delay_between_batches: float = 0.5
) -> List[List[float]]:
    """
    批量获取文本的embedding，自动分批次处理
    
    参数:
        texts: 文本列表
        batch_size: 每批处理的数量
        delay_between_batches: 批次之间的延迟（秒）
    
    返回:
        embedding列表的列表
    """
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        logger.info(f"处理批次 {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}, 数量: {len(batch)}")
        
        try:
            batch_embeddings = get_embedding_func(batch)
            all_embeddings.extend(batch_embeddings)
            
            # 如果不是最后一批，添加延迟
            if i + batch_size < len(texts):
                time.sleep(delay_between_batches)
                
        except Exception as e:
            logger.error(f"批次处理失败，跳过该批次: {str(e)}")
            # 可以选择记录错误并继续，或者抛出异常
            continue
    
    return all_embeddings


# 使用示例
if __name__ == "__main__":
    # 测试单个文本
    single_text = "这是一个测试文本"
    embedding = get_embedding_func(single_text)
    print(f"单个文本embedding长度: {embedding}")
