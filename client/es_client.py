# client/es_client.py
from elasticsearch import Elasticsearch, helpers, exceptions
from typing import List, Dict, Optional, Tuple
import logging
from time import time
import ssl

from config.config import config

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ElasticsearchClient:
    def __init__(self):
        """初始化Elasticsearch客户端（兼容9.x版本）"""
        self.client = None
        self.index_name = config.ES_INDEX
        self.connect()

    def connect(self):
        """连接Elasticsearch（9.x版本）"""
        try:
            # 构建连接URL
            es_url = f"{'https' if config.ES_USE_SSL else 'http'}://{config.ES_HOST}:{config.ES_PORT}"

            # 9.x版本的标准连接参数
            client_kwargs = {
                'hosts': [es_url],
                'request_timeout': config.ES_REQUEST_TIMEOUT,
            }

            # 如果需要认证
            if config.ES_USERNAME and config.ES_PASSWORD:
                client_kwargs['basic_auth'] = (config.ES_USERNAME, config.ES_PASSWORD)

            # SSL配置
            if config.ES_USE_SSL:
                if not config.ES_VERIFY_CERTS:
                    import warnings
                    warnings.filterwarnings("ignore")
                    import urllib3
                    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

                    client_kwargs['verify_certs'] = False

                    # 创建不验证证书的上下文
                    ctx = ssl.create_default_context()
                    ctx.check_hostname = False
                    ctx.verify_mode = ssl.CERT_NONE
                    client_kwargs['ssl_context'] = ctx

            # 9.x版本使用这些连接选项
            client_kwargs.update({
                'connections_per_node': 10,
                'retry_on_timeout': config.ES_RETRY_ON_TIMEOUT,
            })

            # 对于9.x版本，max_retries在Transport层面处理
            # sniff_on_start和sniff_on_connection_fail在9.x中已废弃

            self.client = Elasticsearch(**client_kwargs)

            # 测试连接
            try:
                info = self.client.info()
                if info.get('cluster_name'):
                    logger.info(f"成功连接到 Elasticsearch: {es_url}")
                    logger.info(f"集群名称: {info.get('cluster_name')}")
                    logger.info(f"Elasticsearch版本: {info.get('version', {}).get('number', '未知')}")
                else:
                    logger.error("无法获取集群信息")
            except Exception as e:
                logger.error(f"连接测试失败: {e}")
                self.client = None

        except Exception as e:
            logger.error(f"初始化失败: {e}")
            # 尝试更简单的连接方式
            self._try_simple_connection()

    def _try_simple_connection(self):
        """尝试简单的连接方式（9.x版本）"""
        try:
            logger.info("尝试简单连接方式...")
            es_url = f"{'https' if config.ES_USE_SSL else 'http'}://{config.ES_HOST}:{config.ES_PORT}"

            # 最简单的连接参数
            simple_kwargs = {
                'hosts': [es_url],
                'request_timeout': 10,
            }

            if config.ES_USERNAME and config.ES_PASSWORD:
                simple_kwargs['basic_auth'] = (config.ES_USERNAME, config.ES_PASSWORD)

            if config.ES_USE_SSL and not config.ES_VERIFY_CERTS:
                import warnings
                warnings.filterwarnings("ignore")
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

                simple_kwargs['verify_certs'] = False

                # 创建SSL上下文
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
                simple_kwargs['ssl_context'] = ctx

            self.client = Elasticsearch(**simple_kwargs)

            # 测试连接
            try:
                info = self.client.info()
                if info.get('cluster_name'):
                    logger.info("简单连接方式成功")
                else:
                    logger.error("简单连接方式失败")
            except Exception as e:
                logger.error(f"简单连接测试失败: {e}")
                self.client = None

        except Exception as e:
            logger.error(f"简单连接也失败: {e}")
            self.client = None

    def check_connection(self) -> bool:
        """检查连接状态"""
        try:
            if not self.client:
                return False
            # 发送一个简单的ping请求
            return self.client.ping()
        except Exception as e:
            logger.error(f"连接检查失败: {e}")
            return False

    def create_index(self, embedding_dim: int = 384) -> bool:
        """创建包含向量字段的索引（9.x版本）"""

        if not self.check_connection():
            logger.error("Elasticsearch 连接不可用")
            return False

        try:
            # 检查索引是否存在
            exists = self.client.indices.exists(index=self.index_name)
            if exists:
                logger.info(f"索引 {self.index_name} 已存在")
                return True

            # 9.x版本的mapping语法
            mappings = {
                "properties": {
                    "content": {
                        "type": "text",
                        "analyzer": "ik_max_word",  # 使用IK分词器
                        "search_analyzer": "ik_smart"
                    },
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "keyword"},
                            "page": {"type": "integer"},
                            "chunk_id": {"type": "integer"}
                        }
                    },
                    "embedding": {
                        "type": "dense_vector",
                        "dims": embedding_dim,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "tokens": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "created_at": {
                        "type": "date",
                        "format": "strict_date_optional_time||epoch_millis"
                    }
                }
            }

            settings = {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "default": {
                            "type": "standard"
                        }
                    }
                }
            }

            # 创建索引
            response = self.client.indices.create(
                index=self.index_name,
                mappings=mappings,
                settings=settings
            )

            if response.get('acknowledged', False):
                logger.info(f"索引 {self.index_name} 创建成功，向量维度: {embedding_dim}")
                return True
            else:
                logger.error(f"创建索引失败: {response}")
                return False

        except exceptions.RequestError as e:
            if "resource_already_exists_exception" in str(e):
                logger.info(f"索引 {self.index_name} 已存在")
                return True
            else:
                logger.error(f"创建索引失败: {e}")
                return False
        except Exception as e:
            logger.error(f"创建索引异常: {e}")
            return False

    def delete_index(self) -> bool:
        """删除索引"""
        if not self.check_connection():
            return False

        try:
            exists = self.client.indices.exists(index=self.index_name)
            if exists:
                response = self.client.indices.delete(index=self.index_name)
                if response.get('acknowledged', False):
                    logger.info(f"索引 {self.index_name} 已删除")
                    return True
                else:
                    logger.error(f"删除索引失败: {response}")
                    return False
            else:
                logger.warning(f"索引 {self.index_name} 不存在")
                return True
        except Exception as e:
            logger.error(f"删除索引失败: {e}")
            return False

    def index_documents(self, documents: List[Dict], batch_size: int = 100) -> Tuple[int, int]:
        """批量索引文档"""

        if not self.check_connection():
            logger.error("Elasticsearch 连接不可用")
            return 0, len(documents)

        if not documents:
            logger.warning("没有文档需要索引")
            return 0, 0

        try:
            start_time = time()
            total_docs = len(documents)

            def generate_actions():
                for i, doc in enumerate(documents):
                    # 确保每个文档都有时间戳
                    if 'created_at' not in doc:
                        doc['created_at'] = int(time() * 1000)

                    action = {
                        "_index": self.index_name,
                        "_id": f"{doc.get('metadata', {}).get('source', 'doc')}_{doc.get('chunk_index', i)}",
                        "_source": doc
                    }
                    yield action

            success, failed = helpers.bulk(
                self.client,
                generate_actions(),
                stats_only=True,
                chunk_size=batch_size,
                request_timeout=60
            )

            # 刷新索引
            self.client.indices.refresh(index=self.index_name)

            elapsed_time = time() - start_time
            logger.info(f"索引完成: 成功 {success}/{total_docs}, "
                        f"失败 {failed}, 耗时 {elapsed_time:.2f}秒")

            return success, failed

        except Exception as e:
            logger.error(f"批量索引失败: {e}")
            return 0, len(documents)

    def hybrid_search(self, query: str, query_vector: List[float], top_k: int = 5) -> List[Dict]:
        """混合检索：BM25 + 向量搜索（9.x版本）"""

        if not self.check_connection():
            logger.error("Elasticsearch 连接不可用")
            return []

        try:
            search_body = {
                "size": top_k,
                "_source": ["content", "metadata", "tokens"],
                "query": {
                    "bool": {
                        "should": [
                            {
                                "match": {
                                    "content": {
                                        "query": query,
                                        "boost": config.BM25_WEIGHT
                                    }
                                }
                            },
                            {
                                "script_score": {
                                    "query": {"match_all": {}},
                                    "script": {
                                        "source": """
                                        (1.0 + cosineSimilarity(params.query_vector, 'embedding')) / 2
                                        """,
                                        "params": {
                                            "query_vector": query_vector
                                        }
                                    },
                                    "boost": config.VECTOR_WEIGHT
                                }
                            }
                        ],
                        "minimum_should_match": 1
                    }
                }
            }

            response = self.client.search(
                index=self.index_name,
                body=search_body
            )

            results = []
            for hit in response['hits']['hits']:
                result = {
                    'content': hit['_source']['content'],
                    'metadata': hit['_source']['metadata'],
                    'score': hit['_score'],
                    'id': hit['_id'],
                    'tokens': hit['_source'].get('tokens', '')
                }
                results.append(result)

            logger.info(f"混合搜索完成，返回 {len(results)} 个结果")
            return results

        except Exception as e:
            logger.error(f"混合搜索失败: {e}")
            return []

    def pure_vector_search(self, query_vector: List[float], top_k: int = 5) -> List[Dict]:
        """纯向量搜索"""
        if not self.check_connection():
            return []

        try:
            search_body = {
                "size": top_k,
                "_source": ["content", "metadata"],
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                            "params": {
                                "query_vector": query_vector
                            }
                        }
                    }
                }
            }

            response = self.client.search(
                index=self.index_name,
                body=search_body
            )

            return [
                {
                    'content': hit['_source']['content'],
                    'metadata': hit['_source']['metadata'],
                    'score': hit['_score'],
                    'id': hit['_id']
                }
                for hit in response['hits']['hits']
            ]

        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            return []

    def pure_bm25_search(self, query: str, top_k: int = 5) -> Dict:
        """纯BM25搜索"""
        if not self.check_connection():
            return {'hits': {'hits': [], 'total': {'value': 0}}}

        try:
            search_body = {
                "size": top_k,
                "_source": ["content", "metadata"],
                "query": {
                    "match": {
                        "content": {
                            "query": query,
                            "operator": "and"
                        }
                    }
                }
            }

            response = self.client.search(
                index=self.index_name,
                body=search_body
            )
            return response

        except Exception as e:
            logger.error(f"BM25搜索失败: {e}")
            return {'hits': {'hits': [], 'total': {'value': 0}}}

    def get_index_stats(self) -> Optional[Dict]:
        """获取索引统计信息"""
        if not self.check_connection():
            return None

        try:
            stats = self.client.indices.stats(index=self.index_name)
            return stats
        except Exception as e:
            logger.error(f"获取索引统计失败: {e}")
            return None

    def get_document_count(self) -> int:
        """获取索引中文档数量"""
        if not self.check_connection():
            return 0

        try:
            count_response = self.client.count(index=self.index_name)
            return count_response.get('count', 0)
        except Exception as e:
            logger.error(f"获取文档数量失败: {e}")
            return 0

    def get_cluster_health(self) -> Optional[Dict]:
        """获取集群健康状态（9.x版本）"""
        if not self.check_connection():
            return None

        try:
            health = self.client.cluster.health()
            return health
        except Exception as e:
            logger.error(f"获取集群健康状态失败: {e}")
            return None


# 使用示例
if __name__ == "__main__":
    # 测试连接
    es_client = ElasticsearchClient()

    if es_client.check_connection():
        print("=" * 60)
        print("✅ Elasticsearch 连接成功")
        print("=" * 60)

        # 获取集群信息
        try:
            info = es_client.client.info()
            print(f"集群名称: {info.get('cluster_name', '未知')}")
            print(f"版本: {info.get('version', {}).get('number', '未知')}")

            # 获取集群健康状态
            health = es_client.get_cluster_health()
            if health:
                print(f"集群状态: {health.get('status', '未知')}")
                print(f"节点数量: {health.get('number_of_nodes', 0)}")

        except Exception as e:
            print(f"获取信息失败: {e}")

        # 获取索引统计
        count = es_client.get_document_count()
        print(f"文档数量: {count}")

        # 测试搜索
        if count > 0:
            test_query = "测试"
            test_vector = [0.1] * 384

            results = es_client.hybrid_search(test_query, test_vector, top_k=2)
            print(f"\n测试搜索返回: {len(results)} 条结果")
    else:
        print("=" * 60)
        print("❌ Elasticsearch 连接失败")
        print("=" * 60)
        print("请检查:")
        print("1. Elasticsearch 服务是否运行")
        print("2. 服务器IP和端口是否正确")
        print("3. 用户名密码是否正确")
        print("4. 防火墙设置")
        print("5. 网络连通性")

        # 测试网络连接
        print("\n网络诊断:")
        import socket

        try:
            sock = socket.create_connection((config.ES_HOST, config.ES_PORT), timeout=5)
            print(f"✅ 网络可以连接到 {config.ES_HOST}:{config.ES_PORT}")
            sock.close()
        except socket.timeout:
            print(f"❌ 连接到 {config.ES_HOST}:{config.ES_PORT} 超时")
        except ConnectionRefusedError:
            print(f"❌ 连接被拒绝，端口 {config.ES_PORT} 可能未开放")
        except Exception as e:
            print(f"❌ 网络连接错误: {e}")