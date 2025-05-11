from typing import List 

from fastapi import APIRouter, Depends 

from .services import BenchmarkService
from .schemas import *

router = APIRouter(prefix='/benchmark', tags=['Benchmark'])

@router.post('/')
async def get_texts(
    tests: List[TestModel],
    benchmark_service: BenchmarkService = Depends(BenchmarkService)
) -> List[TestResultModel]:
    return benchmark_service.run_benchmark(tests)
