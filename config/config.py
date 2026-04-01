# config/config.py
import os
from dataclasses import dataclass


@dataclass
class Config:
    # Elasticsearch 配置
    ES_HOST = "192.168.1.103"  # 服务器IP
    ES_PORT = 9200
    ES_USERNAME = "elastic"  # 默认用户名
    ES_PASSWORD = os.getenv("ES_PASSWORD", "")  # 从环境变量读取密码
    ES_INDEX = "rankrag_demo_20260129"  # 创建的索引名称

    # SSL配置
    ES_USE_SSL = False  # 是否启用SSL
    ES_CA_CERTS = None  # CA证书路径
    ES_VERIFY_CERTS = False  # 是否验证证书

    # 文本处理配置
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    # 模型配置
    EMBEDDING_MODEL = "text-embedding-v3"

    # 混合检索权重
    BM25_WEIGHT = 0.3
    VECTOR_WEIGHT = 0.7

    # 连接配置
    ES_REQUEST_TIMEOUT = 30
    ES_MAX_RETRIES = 3
    ES_RETRY_ON_TIMEOUT = True


config = Config()