from abc import ABC, abstractmethod
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

log = logging.getLogger(__name__)

class LLMProvider(ABC):
    @abstractmethod
    async def generate_response(self, prompt: str) -> str:
        pass

class GroqProvider(LLMProvider):
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model_name = os.getenv("GROQ_MODEL_NAME")
        log.info(f"Initializing Groq provider with model: {self.model_name}")
        
        if not self.api_key:
            log.error("GROQ_API_KEY is not set in environment variables")
            raise ValueError("GROQ_API_KEY is not set in environment variables.")
        if not self.model_name:
            log.error("GROQ_MODEL_NAME is not set in environment variables")
            raise ValueError("GROQ_MODEL_NAME is not set in environment variables.")
        
        self.llm = ChatGroq(
            model_name=self.model_name,
            api_key=self.api_key
        )
        log.info(f"Successfully initialized Groq provider with model: {self.model_name}")

    async def generate_response(self, prompt: str) -> str:
        messages = [HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)
        return response.content

class OpenAIProvider(LLMProvider):
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
        log.info(f"Initializing OpenAI provider with model: {self.model_name}")
        
        if not self.api_key:
            log.error("OPENAI_API_KEY is not set in environment variables")
            raise ValueError("OPENAI_API_KEY is not set in environment variables.")
        
        self.llm = ChatOpenAI(
            model_name=self.model_name,
            api_key=self.api_key
        )
        log.info(f"Successfully initialized OpenAI provider with model: {self.model_name}")

    async def generate_response(self, prompt: str) -> str:
        messages = [HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)
        return response.content

def get_llm_provider(provider_name: str) -> LLMProvider:
    provider_name = provider_name.lower()
    log.info(f"Getting LLM provider: {provider_name}")
    
    if provider_name == 'groq':
        return GroqProvider()
    elif provider_name == 'openai':
        return OpenAIProvider()
    else:
        log.error(f"Unsupported LLM provider: {provider_name}")
        raise ValueError(f"Unsupported LLM provider: {provider_name}") 