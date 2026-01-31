get start...

重要备注
- es的ik中文分词器插件下载路径https://release.infinilabs.com/analysis-ik/stable/ ————注意版本对齐
- 目前es的存储数据格式为
        {
          "content": "法条/法规具体内容",
          "metadata": {
            "source": "法条/法规出处",
            "page": 页码,
            "chunk_index": chunk的排序
          },
          "embedding": [0.1, 0.2, ...],  # 1024位向量
          "created_at": 1700000000000
        }