好的,让我们开始第一天的学习,专注于ChatGPT与LangChain的基础内容。根据计划,我们需要先掌握OpenAI API的核心功能,然后开始构建基础设施。
让我们从创建一个基础的OpenAIService类开始,这将是我们与OpenAI API交互的核心组件。

```python
from openai import OpenAI
import time
from typing import Optional, Dict, Any
import logging

class RetryWithExponentialBackoff:
    def __init__(self, initial_delay: float = 1, max_delay: float = 60, max_retries: int = 5):
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.max_retries = max_retries

    def execute(self, func, *args, **kwargs):
        delay = self.initial_delay
        retries = 0

        while retries < self.max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                retries += 1
                if retries == self.max_retries:
                    raise e

                # Calculate next delay with exponential backoff
                delay = min(delay * 2, self.max_delay)
                logging.warning(f"Retry {retries} after {delay} seconds due to {str(e)}")
                time.sleep(delay)

class TokenCounter:
    def __init__(self):
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def update(self, usage: Dict[str, int]):
        self.total_tokens += usage.get('total_tokens', 0)
        self.prompt_tokens += usage.get('prompt_tokens', 0)
        self.completion_tokens += usage.get('completion_tokens', 0)

    def get_stats(self) -> Dict[str, int]:
        return {
            'total_tokens': self.total_tokens,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens
        }

class OpenAIService:
    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = "gpt-3.5-turbo"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.retry_strategy = RetryWithExponentialBackoff()
        self.token_counter = TokenCounter()
        self.logger = logging.getLogger(__name__)

    def chat_completion(self,
                       messages: list[Dict[str, str]],
                       temperature: float = 0.7,
                       max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Send a chat completion request to OpenAI API with retry logic

        Args:
            messages: List of message dictionaries
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Dict containing the API response
        """
        try:
            response = self.retry_strategy.execute(
                self.client.chat.completions.create,
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Update token usage statistics
            if hasattr(response, 'usage'):
                self.token_counter.update(response.usage)

            return {
                'content': response.choices[0].message.content,
                'role': response.choices[0].message.role,
                'usage': response.usage if hasattr(response, 'usage') else None
            }

        except Exception as e:
            self.logger.error(f"Error in chat completion: {str(e)}")
            raise

    def get_token_usage(self) -> Dict[str, int]:
        """Get current token usage statistics"""
        return self.token_counter.get_stats()

# Usage Example
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Initialize service
    service = OpenAIService()

    # Example chat completion request
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]

    try:
        response = service.chat_completion(messages)
        print(f"Response: {response['content']}")
        print(f"Token usage: {service.get_token_usage()}")
    except Exception as e:
        print(f"Error: {str(e)}")

```

这个实现包含了几个关键组件:

1. **RetryWithExponentialBackoff 类**:
   - 实现了指数退避重试策略
   - 可配置初始延迟、最大延迟和最大重试次数
   - 处理各种API错误和网络问题

2. **TokenCounter 类**:
   - 追踪API调用的token使用情况
   - 分别统计提示词和完成词的token数量
   - 提供使用统计信息的访问方法

3. **OpenAIService 类**:
   - 封装了OpenAI API的核心功能
   - 集成了重试策略和token计数器
   - 提供了易用的chat_completion方法
   - 包含完整的错误处理和日志记录

要开始使用这个服务,你需要:

1. 安装必要的依赖:
```bash
pip install openai python-dotenv
```

2. 设置OpenAI API密钥:
```bash
export OPENAI_API_KEY='your-api-key'
```

接下来我们可以:
1. 添加更多的API方法(如embeddings等)
2. 增强错误处理机制
3. 添加更多的配置选项

你想先重点关注哪个方面?