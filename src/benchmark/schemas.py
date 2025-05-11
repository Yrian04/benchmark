from typing import List, Dict, Any

from pydantic import BaseModel


type MetricName = str


class CustomModel(BaseModel):
    pass


class TestModel(CustomModel):
    user_input: str
    reference: str


class TestResultModel(CustomModel):
    user_input: str
    reference: str
    response: str
    retrieved_contexts: List[str]
    metrics: Dict[MetricName, Any]
