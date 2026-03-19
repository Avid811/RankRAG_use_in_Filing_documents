get start...

重要备注
- es的ik中文分词器插件下载路径https://release.infinilabs.com/analysis-ik/stable/ ————注意版本对齐
- 目前es的存储数据格式为
 {
        "content": "法条/法规具体内容",
        "metadata": {
                "source": "eg.中华人民共和国个人信息保护法",
                "takeEffect": "法条生效日期",
                "lawType": "eg.人身权、侵犯公民人身权利民主权利罪",
                "whoMake": "eg.全国人大常委会",
                "chapter": "第几章",
                "section": "第几节",
                "articleNumber": "第几条",
                "status": "现行状态，失效、现行有效等"
        },
        "chunkId": "当前块的索引",
        "embedding": "[0.1, 0.2, ...]"
}
