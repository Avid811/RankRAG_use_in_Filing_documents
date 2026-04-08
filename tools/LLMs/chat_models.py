"""
调用qwen系列chat模型

call方法返回数据结构
{
    "status_code": 200,
    "request_id": "5076e9e7-4fc7-9af1-a30b-7b911804e8d8",
    "code": "",
    "message": "",
    "output": {
        "text": null,
        "finish_reason": null,
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "\u4f60\u597d\uff01\u6211\u662f\u901a\u4e49\u5343\u95ee\uff08Qwen\uff09\uff0c\u963f\u91cc\u5df4\u5df4\u96c6\u56e2\u65d7\u4e0b\u7684\u8d85\u5927\u89c4\u6a21\u8bed\u8a00\u6a21\u578b\u3002\u6211\u80fd\u591f\u56de\u7b54\u95ee\u9898\u3001\u521b\u4f5c\u6587\u5b57\uff0c\u6bd4\u5982\u5199\u6545\u4e8b\u3001\u5199\u516c\u6587\u3001\u5199\u90ae\u4ef6\u3001\u5199\u5267\u672c\u3001\u903b\u8f91\u63a8\u7406\u3001\u7f16\u7a0b\u7b49\u7b49\uff0c\u8fd8\u80fd\u8868\u8fbe\u89c2\u70b9\uff0c\u73a9\u6e38\u620f\u7b49\u3002\u5982\u679c\u4f60\u6709\u4efb\u4f55\u95ee\u9898\u6216\u9700\u8981\u5e2e\u52a9\uff0c\u6b22\u8fce\u968f\u65f6\u544a\u8bc9\u6211\uff01\ud83d\ude0a"
                }
            }
        ]
    },
    "usage": {
        "input_tokens": 22,
        "output_tokens": 66,
        "total_tokens": 88,
        "prompt_tokens_details": {
            "cached_tokens": 0
        }
    }
}
"""

import json
import os
import time
import random
import re
import dashscope


def chat(model, prompt, asRerank=False) -> str:
    """
    调用qwen系列chat模型
    
    Args:
        model: 模型名称，如 'qwen3-max', 'qwen3.6-plus', 'qwen3.5-flash'
        prompt: 提示词
        asRerank: 是否为重排序模式，True时会进行JSON格式检查和兜底处理
    
    Returns:
        str: 模型返回的内容
    """
    messages = [
        {"role": "system", "content": "你是一个专业的法律顾问"},
        {"role": "user", "content": prompt},
    ]
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # 根据模型名称选择不同的调用方法
            if model in ['qwen3.6-plus', 'qwen3.5-flash']:
                # qwen3.6-plus 和 qwen3.5-flash 使用 MultiModalConversation
                response = dashscope.MultiModalConversation.call(
                    api_key=os.getenv('DASHSCOPE_API_KEY'),
                    model=model,
                    messages=messages,
                    result_format='message'
                )
            elif model == 'qwen3-max':
                # qwen3-max 使用 Generation
                response = dashscope.Generation.call(
                    api_key=os.getenv('DASHSCOPE_API_KEY'),
                    model=model,
                    messages=messages,
                    result_format='message'
                )
            else:
                # 默认使用 Generation
                response = dashscope.Generation.call(
                    api_key=os.getenv('DASHSCOPE_API_KEY'),
                    model=model,
                    messages=messages,
                    result_format='message'
                )
            
            # 检查响应状态
            if response.status_code == 200:
                content = response.output.choices[0].message.content
                
                # 如果不是asRerank模式，直接返回结果
                if not asRerank:
                    return content
                
                # asRerank模式：检查是否为有效的JSON格式
                try:
                    # 尝试解析JSON
                    json.loads(content)
                    return content
                except json.JSONDecodeError:
                    # 兜底机制：使用正则表达式提取信息
                    print("警告：返回结果不是有效JSON，使用兜底机制处理")
                    # 提取文段
                    text_match = re.search(r'"文段":\s*"([^"]+)"', content)
                    text = text_match.group(1) if text_match else ""
                    
                    # 提取违规记录
                    violations = []
                    # 匹配违规记录块
                    violation_pattern = re.compile(r'"违规记录":\s*\[(.*?)\]', re.DOTALL)
                    violation_match = violation_pattern.search(content)
                    if violation_match:
                        violation_content = violation_match.group(1)
                        # 匹配每个违规条目
                        item_pattern = re.compile(r'\{[^\}]*"法律法规":\s*"([^"]+)",[^\}]*"违规原因":\s*"([^"]+)",[^\}]*"程度":\s*"([^"]+)"[^\}]*\}', re.DOTALL)
                        for match in item_pattern.finditer(violation_content):
                            law, reason, level = match.groups()
                            violations.append({
                                "法律法规": law,
                                "违规原因": reason,
                                "程度": level
                            })
                    
                    # 构建标准JSON格式
                    if violations:
                        standard_json = [{
                            "文段": text,
                            "违规记录": violations
                        }]
                        return json.dumps(standard_json, ensure_ascii=False)
                    else:
                        return "当前chunk未触犯任何法律法规"
            else:
                print(f"HTTP返回码：{response.status_code}")
                print(f"错误码：{response.code}")
                print(f"错误信息：{response.message}")
                if attempt < max_retries - 1:
                    print(f"重试中... ({attempt + 1}/{max_retries})")
                    time.sleep(2 + random.random() * 3)
                    continue
                else:
                    return "当前chunk未触犯任何法律法规"
                    
        except Exception as e:
            print(f"错误：{e}")
            if attempt < max_retries - 1:
                print(f"重试中... ({attempt + 1}/{max_retries})")
                time.sleep(2 + random.random() * 3)
                continue
            else:
                return "当前chunk未触犯任何法律法规"
