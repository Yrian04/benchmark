from qdrant_client import QdrantClient
from langchain_openai import ChatOpenAI, OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

from organisation_utils.logging_config import logger_factory

from src.config import settings


def get_logger():
    return logger_factory.get_logger("BenchmarkService")


def get_qdrant_client():
    return QdrantClient(url=settings.get_qdrant_url())


def get_llm() -> ChatOpenAI:
    open_ai_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.get_model_key()
    )
    return ChatOpenAI(
        client=open_ai_client,
        openai_api_key=settings.get_model_key(),
        openai_api_base="https://openrouter.ai/api/v1",
        model=settings.get_model_name(),
        temperature=0.0,
    )


def get_embeddings() -> LangchainEmbeddingsWrapper: 
    base_embeddings = HuggingFaceEmbeddings(
        model_name=settings.get_embeddings_model_name()
    )
    return LangchainEmbeddingsWrapper(base_embeddings)