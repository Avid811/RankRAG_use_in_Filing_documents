# client/es_client.py
from elasticsearch import Elasticsearch, helpers, exceptions
from typing import List, Dict, Optional, Tuple
import logging
from time import time
import warnings
import urllib3

from config.config import config

# 配置日志，只显示我们自己的日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 禁用或降低Elasticsearch客户端库的日志级别
for logger_name in ['elasticsearch', 'elastic_transport']:
    es_logger = logging.getLogger(logger_name)
    es_logger.setLevel(logging.WARNING)

# 降低client.es_client的日志级别，减少连接和搜索信息的输出
logger.setLevel(logging.WARNING)


class ElasticsearchClient:
    def __init__(self):
        self.client = None
        self.index_name = config.ES_INDEX
        self.connect()

    def connect(self):
        try:
            es_url = f"{'https' if config.ES_USE_SSL else 'http'}://{config.ES_HOST}:{config.ES_PORT}"

            client_kwargs = {
                'hosts': [es_url],
                'request_timeout': config.ES_REQUEST_TIMEOUT,
                'max_retries': 3,
                'retry_on_timeout': config.ES_RETRY_ON_TIMEOUT,
            }

            if config.ES_USERNAME and config.ES_PASSWORD:
                client_kwargs['basic_auth'] = (config.ES_USERNAME, config.ES_PASSWORD)

            if config.ES_USE_SSL:
                if not config.ES_VERIFY_CERTS:
                    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                    client_kwargs['verify_certs'] = False
                else:
                    client_kwargs['ca_certs'] = config.ES_CA_CERTS if hasattr(config, 'ES_CA_CERTS') else None

            warnings.filterwarnings("ignore", category=DeprecationWarning, module="elasticsearch")
            self.client = Elasticsearch(**client_kwargs)

            try:
                info = self.client.info()
                if info.get('cluster_name'):
                    logger.info(f"连接到 Elasticsearch: {es_url}")
                    logger.info(f"集群: {info.get('cluster_name')}")
                    logger.info(f"版本: {info.get('version', {}).get('number', '未知')}")
                else:
                    logger.error("无法获取集群信息")
            except Exception as e:
                logger.error(f"连接测试失败: {e}")
                self.client = None

        except Exception as e:
            logger.error(f"初始化失败: {e}")
            self._try_simple_connection()

    def _try_simple_connection(self):
        try:
            es_url = f"{'https' if config.ES_USE_SSL else 'http'}://{config.ES_HOST}:{config.ES_PORT}"

            simple_kwargs = {
                'hosts': [es_url],
                'request_timeout': 10,
            }

            if config.ES_USERNAME and config.ES_PASSWORD:
                simple_kwargs['basic_auth'] = (config.ES_USERNAME, config.ES_PASSWORD)

            if config.ES_USE_SSL and not config.ES_VERIFY_CERTS:
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                simple_kwargs['verify_certs'] = False

            warnings.filterwarnings("ignore", category=DeprecationWarning, module="elasticsearch")
            self.client = Elasticsearch(**simple_kwargs)

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
        try:
            if not self.client:
                return False
            return self.client.ping()
        except Exception as e:
            logger.error(f"连接检查失败: {e}")
            return False

    def index_exists(self) -> bool:
        """检查索引是否存在"""
        if not self.check_connection():
            logger.error("Elasticsearch 连接不可用")
            return False

        try:
            return self.client.indices.exists(index=self.index_name)
        except Exception as e:
            logger.error(f"检查索引存在性失败: {e}")
            return False

    def create_index(self, embedding_dim: int = 1024) -> bool:
        if not self.check_connection():
            logger.error("Elasticsearch 连接不可用")
            return False

        try:
            exists = self.client.indices.exists(index=self.index_name)
            if exists:
                logger.info(f"索引 {self.index_name} 已存在")
                return True

            # 【修改点 1】：更新 mapping 以适应新的数据结构
            mappings = {
                "properties": {
                    "content": {
                        "type": "text",
                        "analyzer": "ik_max_word",
                        "search_analyzer": "ik_smart"
                    },
                    "chunkId": {
                        "type": "keyword"
                    },
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "keyword"},
                            "takeEffect": {"type": "keyword"},
                            "lawType": {"type": "keyword"},
                            "whoMake": {"type": "keyword"},
                            "status": {"type": "keyword"},
                            "chapter": {"type": "keyword"},
                            "section": {"type": "keyword"},
                            "articleNumber": {"type": "keyword"}
                        }
                    },
                    "embedding": {
                        "type": "dense_vector",
                        "dims": embedding_dim,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "created_at": {
                        "type": "date",
                        "format": "strict_date_optional_time||epoch_millis"
                    }
                }
            }

            settings = {
                "index": {
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
            }

            response = self.client.indices.create(
                index=self.index_name,
                body={
                    "mappings": mappings,
                    "settings": settings
                }
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

    def index_documents(self, documents, batch_size: int = 100) -> Tuple[int, int]:
        if not self.check_connection():
            logger.error("Elasticsearch 连接不可用")
            return 0, len(documents)

        if not documents:
            logger.warning("没有文档需要索引")
            return 0, 0

        try:
            start_time = time()
            es_docs = []

            for i, doc in enumerate(documents):
                if isinstance(doc, dict):
                    es_doc = doc.copy()
                    if 'created_at' not in es_doc:
                        es_doc['created_at'] = int(time() * 1000)
                else:
                    logger.warning(f"文档 {i} 格式不支持: {type(doc)}")
                    continue

                es_docs.append(es_doc)

            if not es_docs:
                logger.warning("没有有效的文档可以索引")
                return 0, 0

            total_docs = len(es_docs)

            def generate_actions():
                for i, doc in enumerate(es_docs):
                    # 【修改点 2】：适应新的 chunkId 位置和元数据来源
                    source = doc.get('metadata', {}).get('source', 'unknown_source')
                    chunk_id = doc.get('chunkId', str(i))

                    # 为了保证 _id 的唯一性和可读性
                    doc_id = f"{source}_chunk_{chunk_id}"

                    action = {
                        "_index": self.index_name,
                        "_id": doc_id,
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

            self.client.indices.refresh(index=self.index_name)

            elapsed_time = time() - start_time
            logger.info(f"索引完成: 成功 {success}/{total_docs}, 耗时 {elapsed_time:.2f}秒")

            return success, failed

        except Exception as e:
            logger.error(f"批量索引失败: {e}")
            import traceback
            traceback.print_exc()
            return 0, len(documents)

    def hybrid_search(self, query: str, query_vector: List[float], top_k: int = 5,
                      return_raw_scores: bool = False) -> List[Dict]:
        if not self.check_connection():
            logger.error("Elasticsearch 连接不可用")
            return []

        try:
            return self._hybrid_search_with_knn(query, query_vector, top_k, return_raw_scores)
        except Exception as e:
            logger.info(f"KNN混合检索失败: {e}")
            return []

    def _hybrid_search_with_knn(self, query: str, query_vector: List[float],
                                top_k: int = 5, return_raw_scores: bool = False) -> List[Dict]:
        num_candidates = top_k * 10

        search_body = {
            "size": top_k,
            # 【修改点 3】：在 _source 中增加 chunkId 返回
            "_source": ["content", "metadata", "chunkId"],
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
                        }
                    ],
                    "minimum_should_match": 1
                }
            },
            "knn": {
                "field": "embedding",
                "query_vector": query_vector,
                "k": top_k * 2,
                "num_candidates": num_candidates,
                "boost": config.VECTOR_WEIGHT
            }
        }

        if return_raw_scores:
            search_body["ext"] = {
                "knn": {
                    "field": "embedding",
                    "query_vector": query_vector,
                    "k": num_candidates
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
                'metadata': hit['_source'].get('metadata', {}),
                'chunkId': hit['_source'].get('chunkId', ''),  # 提取 chunkId
                'score': hit['_score'],
                'id': hit['_id']
            }

            if return_raw_scores and hit.get('_score') is not None:
                bm25_score = 0.0
                vector_score = 0.0
                final_score = hit['_score']

                if 'matched_queries' in hit:
                    result['matched_queries'] = hit['matched_queries']

                result['bm25_score'] = bm25_score
                result['vector_score'] = vector_score
                result['bm25_weight'] = config.BM25_WEIGHT
                result['vector_weight'] = config.VECTOR_WEIGHT
                result['weighted_bm25'] = bm25_score * config.BM25_WEIGHT
                result['weighted_vector'] = vector_score * config.VECTOR_WEIGHT
                result['final_score'] = final_score

            results.append(result)

        logger.info(f"混合搜索(KNN)完成，返回 {len(results)} 个结果")
        return results

    def pure_vector_search(self, query_vector: List[float], top_k: int = 5) -> List[Dict]:
        if not self.check_connection():
            return []
        try:
            return self._knn_search(query_vector, top_k)
        except Exception as e:
            logger.info(f"KNN搜索失败 {e}")
            return []

    def _knn_search(self, query_vector: List[float], top_k: int = 5) -> List[Dict]:
        search_body = {
            "size": top_k,
            "_source": ["content", "metadata", "chunkId"],  # 加入 chunkId
            "knn": {
                "field": "embedding",
                "query_vector": query_vector,
                "k": top_k * 2,
                "num_candidates": top_k * 10
            }
        }

        response = self.client.search(
            index=self.index_name,
            body=search_body
        )

        return [
            {
                'content': hit['_source']['content'],
                'metadata': hit['_source'].get('metadata', {}),
                'chunkId': hit['_source'].get('chunkId', ''),
                'score': hit['_score'],
                'id': hit['_id']
            }
            for hit in response['hits']['hits'][:top_k]
        ]

    def pure_bm25_search(self, query: str, top_k: int = 5) -> Dict:
        if not self.check_connection():
            return {'hits': {'hits': [], 'total': {'value': 0}}}

        try:
            search_body = {
                "size": top_k,
                "_source": ["content", "metadata", "chunkId"],  # 加入 chunkId
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
        if not self.check_connection():
            return None
        try:
            stats = self.client.indices.stats(index=self.index_name)
            return stats
        except exceptions.NotFoundError:
            logger.warning(f"索引 {self.index_name} 不存在")
            return None
        except Exception as e:
            logger.error(f"获取索引统计失败: {e}")
            return None

    def get_document_count(self) -> int:
        if not self.check_connection():
            return 0
        try:
            count_response = self.client.count(index=self.index_name)
            return count_response.get('count', 0)
        except exceptions.NotFoundError:
            logger.warning(f"索引 {self.index_name} 不存在")
            return 0
        except Exception as e:
            logger.error(f"获取文档数量失败: {e}")
            return 0

    def get_cluster_health(self) -> Optional[Dict]:
        if not self.check_connection():
            return None
        try:
            health = self.client.cluster.health()
            return health
        except Exception as e:
            logger.error(f"获取集群健康状态失败: {e}")
            return None

    def create_index_if_not_exists(self, embedding_dim: int = 1024) -> bool:
        try:
            if not self.check_connection():
                return False

            exists = self.client.indices.exists(index=self.index_name)
            if not exists:
                return self.create_index(embedding_dim)
            else:
                logger.info(f"索引 {self.index_name} 已存在")
                return True
        except Exception as e:
            logger.error(f"检查索引存在性失败: {e}")
            return False

    def analyze_text(self, text: str, analyzer: str = "ik_smart") -> List[str]:
        if not self.check_connection():
            return []
        try:
            response = self.client.indices.analyze(
                body={
                    "analyzer": analyzer,
                    "text": text
                }
            )
            tokens = [token.get('token', '') for token in response.get('tokens', [])]
            return tokens
        except Exception as e:
            logger.warning(f"分词分析失败: {e}")
            return []

    def get_separate_scores(self, query: str, query_vector: List[float], top_k: int = 5) -> Dict:
        if not self.check_connection():
            return {"hybrid_results": [], "bm25_scores": {}, "vector_scores": {}}

        try:
            search_body = {
                "size": top_k * 5,
                "_source": ["content", "metadata", "chunkId"],  # 加入 chunkId
                "query": {
                    "match": {
                        "content": {
                            "query": query,
                            "operator": "or",
                            "minimum_should_match": "1"
                        }
                    }
                }
            }

            try:
                response = self.client.search(
                    index=self.index_name,
                    body=search_body
                )
            except Exception as e:
                logger.error(f"BM25搜索失败: {e}")
                response = {'hits': {'hits': [], 'total': {'value': 0}}}

            bm25_scores = {}
            for hit in response.get('hits', {}).get('hits', []):
                bm25_scores[hit['_id']] = {
                    'bm25_score': hit['_score'],
                    'content': hit['_source']['content'],
                    'metadata': hit['_source'].get('metadata', {}),
                    'chunkId': hit['_source'].get('chunkId', '')
                }

            if not bm25_scores:
                import re
                clean_query = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', query)
                clean_query = ' '.join(clean_query.split())

                if clean_query and clean_query != query:
                    search_body['query']['match']['content']['query'] = clean_query
                    try:
                        response = self.client.search(
                            index=self.index_name,
                            body=search_body
                        )
                    except Exception as e:
                        logger.error(f"清理后BM25搜索失败: {e}")
                    else:
                        for hit in response.get('hits', {}).get('hits', []):
                            bm25_scores[hit['_id']] = {
                                'bm25_score': hit['_score'],
                                'content': hit['_source']['content'],
                                'metadata': hit['_source'].get('metadata', {}),
                                'chunkId': hit['_source'].get('chunkId', '')
                            }

            vector_results = self.pure_vector_search(query_vector, top_k=top_k * 5)
            vector_scores = {}

            for result in vector_results:
                vector_scores[result['id']] = {
                    'vector_score': result['score'],
                    'content': result['content'],
                    'metadata': result.get('metadata', {}),
                    'chunkId': result.get('chunkId', '')
                }

            hybrid_results = self.hybrid_search(query, query_vector, top_k)

            detailed_results = []
            for result in hybrid_results:
                doc_id = result['id']
                bm25_info = bm25_scores.get(doc_id, {'bm25_score': 0.0})
                vector_info = vector_scores.get(doc_id, {'vector_score': 0.0})

                detailed_result = {
                    'id': doc_id,
                    'content': result['content'],
                    'metadata': result.get('metadata', {}),
                    'chunkId': result.get('chunkId', ''),
                    'bm25_score': bm25_info.get('bm25_score', 0.0),
                    'vector_score': vector_info.get('vector_score', 0.0),
                    'hybrid_score': result.get('score', 0.0)
                }
                detailed_results.append(detailed_result)

            return {
                'hybrid_results': detailed_results,
                'bm25_scores': bm25_scores,
                'vector_scores': vector_scores
            }

        except Exception as e:
            logger.error(f"获取详细搜索得分失败: {e}")
            import traceback
            traceback.print_exc()
            return {"hybrid_results": [], "bm25_scores": {}, "vector_scores": {}}

    def improved_bm25_search(self, query: str, top_k: int = 5, operator: str = "or",
                             minimum_should_match: str = "1") -> Dict:
        if not self.check_connection():
            return {'hits': {'hits': [], 'total': {'value': 0}}}

        try:
            search_body = {
                "size": top_k,
                "_source": ["content", "metadata", "chunkId"],  # 加入 chunkId
                "query": {
                    "match": {
                        "content": {
                            "query": query,
                            "operator": operator,
                            "minimum_should_match": minimum_should_match
                        }
                    }
                }
            }

            response = self.client.search(
                index=self.index_name,
                body=search_body
            )

            total = response.get('hits', {}).get('total', {}).get('value', 0)
            logger.info(f"BM25搜索: 查询='{query}', 匹配到{total}个文档")

            return response

        except Exception as e:
            logger.error(f"BM25搜索失败: {e}")
            return {'hits': {'hits': [], 'total': {'value': 0}}}