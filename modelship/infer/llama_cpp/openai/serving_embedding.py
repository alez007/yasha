import asyncio
import inspect

from llama_cpp import Llama

from modelship.infer.base_serving import OpenAIServing
from modelship.infer.infer_config import RawRequestProxy
from modelship.logging import get_logger
from modelship.openai.protocol import (
    EmbeddingRequest,
    EmbeddingResponse,
    ErrorResponse,
    create_error_response,
)
from modelship.utils import base_request_id

logger = get_logger("infer.llama_cpp.embedding")


class OpenAIServingEmbedding(OpenAIServing):
    request_id_prefix = "embd"

    def __init__(self, llama: Llama, model_name: str):
        self._llama = llama
        self.model_name = model_name
        self._lock = asyncio.Lock()
        self._accepted_params = set(inspect.signature(llama.create_embedding).parameters)

    async def warmup(self) -> None:
        logger.info("Warming up llama.cpp embedding model: %s", self.model_name)
        request = EmbeddingRequest(model=self.model_name, input="warmup")
        await self.create_embedding(request, RawRequestProxy(None, {}))
        logger.info("Warmup embedding done for %s", self.model_name)

    async def create_embedding(
        self, request: EmbeddingRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | EmbeddingResponse:
        request_id = f"{self.request_id_prefix}-{base_request_id(raw_request)}"
        logger.info("embedding request %s", request_id)

        params = request.model_dump(exclude_none=True)
        kwargs: dict = {}
        dropped: list[str] = []
        for k, v in params.items():
            if k in self._accepted_params:
                kwargs[k] = v
            else:
                dropped.append(k)
        if dropped:
            logger.warning(
                "llama_cpp: dropping request params not supported by create_embedding: %s",
                dropped,
            )

        loop = asyncio.get_event_loop()
        llama = self._llama
        async with self._lock:
            try:
                result = await loop.run_in_executor(None, lambda: llama.create_embedding(**kwargs))
                return EmbeddingResponse.model_validate(result)
            except Exception as e:
                logger.warning("llama_cpp embedding failed: %s", e)
                return create_error_response(e)
