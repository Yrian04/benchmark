import requests
from typing import(
    Tuple,
    List,
    TypedDict,
    NotRequired
)

from fastapi import Depends
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Filter, 
    FieldCondition, 
    MatchValue
)
from langchain_openai import ChatOpenAI
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import evaluate
from ragas.metrics import (
    faithfulness, 
    answer_relevancy, 
    context_recall, 
    context_precision,
)
from datasets import Dataset

from src.config import settings

from .dependencies import *
from .schemas import *


type URL = str
type Response = str
type Context = str

class Test(TypedDict):
    user_input: str
    response: NotRequired[str]
    reference: str
    retrieved_contexts: NotRequired[List[str]]
    

class BenchmarkService:
    def __init__(
        self,
        qdrant_client: QdrantClient = Depends(get_qdrant_client),
        llm: ChatOpenAI = Depends(get_llm),
        embeddings: LangchainEmbeddingsWrapper = Depends(get_embeddings),
        logger = Depends(get_logger)
    ):
        self.qdrant_client = qdrant_client
        self.llm = llm
        self.embeddings = embeddings
        self.logger = logger
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_recall,
            # context_precision,
        ]

    def _call_rag_service(self, query: str) -> Tuple[Response, List[URL]]:
        """Вызвать RAG-микросервис через REST API"""
        try:
            self.logger.info("Request to rag service: %s", query)
            rag_response = requests.post(
                settings.get_rag_service_url(),
                data={"message": query},
                timeout=1000
            )
            if rag_response.status_code == 200:
                response = rag_response.json().get("response", "")
                sources_urls = rag_response.json().get("source_urls", [])
                self.logger.info(
                    "Response is gotten. Response: %s Sources: %s",
                    response,
                    sources_urls 
                )
                return (response, sources_urls)
            else:
                self.logger.error(
                    "Ошибка вызова RAG-сервиса: %s %s",
                    rag_response.status_code,
                    rag_response.reason
                )
                return "", []
        except Exception as e:
            self.logger.exception("Исключение при вызове RAG-сервиса: %s", e)
            return "", []

    def _get_retrieved_contexts(self, source_urls: List[URL]) -> List[Context]:
        retrieved_contexts = []
        for source_url in source_urls:
            self.logger.info("Getting retrieved context %s", source_url)
            point = self.qdrant_client.scroll(
                collection_name=settings.get_qdrant_collection(),
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="source_url",
                            match=MatchValue(value=source_url)
                        )
                    ]
                ),
                limit=1
            )[0][0]
            content = point.payload.get('content', '')
            self.logger.info("Context was gotte %s", content)
            retrieved_contexts.append(content)
        return retrieved_contexts

    def _generate_dataset(self, tests: List[Test]) -> List[Test]:
        """Сгенерировать набор тестовых данных"""
        self.logger.info("Start generating dataset %s", tests)
        for test in tests:
            response, source_urls = self._call_rag_service(test['user_input'])
            test['response'] = response 
            test['retrieved_contexts'] = self._get_retrieved_contexts(source_urls)
            test['reference_contexts'] = self._get_retrieved_contexts(source_urls)
        self.logger.info("Finish generating dataset %s", tests)
        return tests

    def run_benchmark(self, tests: List[TestModel]) -> List[TestResultModel]:
        """Запустить оценку метрик с помощью Ragas"""
        self.logger.info("Start benchmark")
        test_dataset = self._generate_dataset([
            Test(
                user_input=t.user_input,
                reference=t.reference
            )
            for t in tests
        ])

        self.logger.info("Start evaluating")
        result = evaluate(
            Dataset.from_list(test_dataset),
            metrics=self.metrics,
            llm=self.llm,
            embeddings=self.embeddings
        )
        self.logger.info("Finish evaluating %s", result.to_pandas())
        self.logger.info("Finish benchmark")

        return [
            TestResultModel(
                user_input=d.user_input,
                reference=d.reference,
                response=d.response,
                retrieved_contexts=d.retrieved_contexts,
                metrics=t
            )
            for d, t in zip(result.dataset, result.scores)
        ]
