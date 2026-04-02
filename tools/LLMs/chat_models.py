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
from dashscope import Generation
import dashscope


def chat(model, prompt) -> str:
    messages = [
        {"role": "system", "content": "你是一个专业的法律顾问"},
        {"role": "user", "content": prompt},
    ]
    response = Generation.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        # Qwen3.5系列需要使用多模态接口，直接替换模型会导致报错
        model=model,
        messages=messages,
        result_format="message",
    )

    if response.status_code == 200:
        return response.output.choices[0].message.content
        # 如需查看完整响应，取消下列注释
        # print(json.dumps(response, default=lambda o: o.__dict__, indent=4))
    else:
        print(f"HTTP返回码：{response.status_code}")
        print(f"错误码：{response.code}")
        print(f"错误信息：{response.message}")
        print("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")
