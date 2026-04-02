import os
import time
from functools import wraps
from typing import Dict

import dashscope

class CallCostTracker:
    """百炼单次调用成本追踪器（精简打印版）"""

    def __init__(self):
        # 阿里云百炼计费标准 (单位: 元/千Token)
        self.pricing = {
            "qwen-max": {"input": 0.02, "output": 0.08},
            "qwen-plus": {"input": 0.01, "output": 0.04},
            "qwen-turbo": {"input": 0.003, "output": 0.012},
            "text-embedding-v3": {"input": 0.0005, "output": 0.0},
        }

    def track_call(self, func):
        """追踪单次函数调用的装饰器，直接在控制台打印账单"""

        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                # 执行原始函数
                response = func(*args, **kwargs)
                end_time = time.time()
                duration = end_time - start_time

                # 1. 安全提取使用的模型名称
                model = kwargs.get('model')
                if not model:
                    try:
                        # 优先尝试使用字典的 get 方法提取
                        if hasattr(response, 'get'):
                            model = response.get('model')
                        # 如果没找到，再尝试属性访问，并捕获可能的 KeyError
                        if not model:
                            model = getattr(response, 'model', None)
                    except Exception:  # 这里使用 Exception 兜底，防止 KeyError 穿透
                        pass
                model = model or "unknown"

                # 2. 安全提取 token 使用情况
                prompt_tokens = completion_tokens = total_tokens = 0
                usage = None

                try:
                    # 同样防范 KeyError 穿透
                    usage = getattr(response, 'usage', None)
                except Exception:
                    pass

                if usage is None and hasattr(response, 'get'):
                    usage = response.get('usage')

                if usage:
                    if isinstance(usage, dict):
                        total_tokens = usage.get('total_tokens', 0)
                        prompt_tokens = usage.get('input_tokens', usage.get('prompt_tokens', 0))
                        completion_tokens = usage.get('output_tokens', usage.get('completion_tokens', 0))
                    else:
                        total_tokens = getattr(usage, 'total_tokens', 0)
                        prompt_tokens = getattr(usage, 'input_tokens', getattr(usage, 'prompt_tokens', 0))
                        completion_tokens = getattr(usage, 'output_tokens', getattr(usage, 'completion_tokens', 0))

                    # 🌟 核心修复点：针对 Embedding 这种只返回 total_tokens 的情况
                    # 将所有的 token 消耗全部算作 input (prompt_tokens) 计费
                    if total_tokens > 0 and prompt_tokens == 0 and completion_tokens == 0:
                        prompt_tokens = total_tokens

                # 3. 计算成本
                cost = self._calculate_cost(model, prompt_tokens, completion_tokens)

                # 4. ====== 直接打印单次调用的详细账单 ======
                print("\n" + "=" * 45)
                print(f"[{func.__name__}] 调用完成")
                print(f"模型名称: {model}")
                print(f"耗时:     {duration:.2f} 秒")
                print(f"Token消耗: {prompt_tokens} (入) + {completion_tokens} (出) = {total_tokens} 总计")
                print(f"本次费用: ¥{cost:.6f} 元")
                print("=" * 45 + "\n")

                return response

            except Exception as e:
                print(f"\n❌ [{func.__name__}] 调用失败 | 耗时: {time.time() - start_time:.2f} 秒 | 错误: {str(e)}\n")
                raise

        return wrapper

    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """计算调用成本逻辑"""
        if model not in self.pricing:
            return 0.0

        prices = self.pricing[model]
        input_cost = (prompt_tokens / 1000) * prices["input"]
        output_cost = (completion_tokens / 1000) * prices["output"]
        return input_cost + output_cost

# ==========================================
# 使用示例
# ==========================================
tracker = CallCostTracker()

@tracker.track_call
def call_embedding(text: str, model: str = "text-embedding-v3"):
    """调用百炼 Embedding 模型"""
    # 模拟调用
    response = dashscope.TextEmbedding.call(
        model=model,
        input=text,
        api_key=os.getenv("DASHSCOPE_API_KEY", os.getenv("OPENAI_API_KEY"))
    )
    return response

if __name__ == "__main__":
    # 测试 Embedding
    print(call_embedding("飒飒大苏打撒旦重新注册", model="text-embedding-v3"))