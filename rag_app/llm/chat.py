from dotenv import load_dotenv
load_dotenv()

import os
from langchain_openai import ChatOpenAI


def get_chat_model(temperature: float = 0.1):
    api_key = os.getenv('DEEPSEEK_API_KEY', '')
    base_url = os.getenv('DEEPSEEK_BASE_URL', 'https://api.siliconflow.cn/v1')
    model_name = os.getenv('DEEPSEEK_MODEL', 'deepseek-ai/DeepSeek-V4-Flash')
    
    if not api_key:
        raise ValueError(
            'DEEPSEEK_API_KEY is not configured.\n'
            'Please set it in .env file or as an environment variable.'
        )
    
    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )
    
    return llm
