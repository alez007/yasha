import time

from sentence_transformers import SentenceTransformer
from vllm.entrypoints.openai.engine.protocol import UsageInfo
from vllm.entrypoints.pooling.embed.protocol import EmbeddingResponseData

from modelship.infer.base_serving import OpenAIServing
from modelship.infer.infer_config import RawRequestProxy
from modelship.logging import TRACE, get_logger
from modelship.openai.protocol import EmbeddingRequest, EmbeddingResponse, ErrorResponse, create_error_response
from modelship.utils import base_request_id

logger = get_logger("infer.transformers.embedding")


class OpenAIServingEmbedding(OpenAIServing):
    request_id_prefix = "embd"

    def __init__(self, model: SentenceTransformer, model_name: str):
        self.model = model
        self.model_name = model_name

    async def warmup(self) -> None:
        logger.info("Warming up embedding model: %s", self.model_name)
        await self.run_in_executor(self._run, ["warmup"])
        logger.info("Warmup embedding done for %s", self.model_name)

    def _run(self, inputs: list[str]):
        return self.model.encode(inputs)

    async def create_embedding(
        self, request: EmbeddingRequest, raw_request: RawRequestProxy
    ) -> EmbeddingResponse | ErrorResponse:
        request_id = f"{self.request_id_prefix}-{base_request_id(raw_request)}"
        logger.info("embedding request %s", request_id)
        logger.log(TRACE, "embedding request %s: input=%s", request_id, request.input)

        inputs = request.input
        if isinstance(inputs, str):
            inputs = [inputs]

        try:
            embeddings = await self.run_in_executor(self._run, inputs)
        except Exception:
            logger.exception("embedding inference failed for %s", request_id)
            return create_error_response("embedding inference failed")

        tokenized = self.model.tokenize(inputs)
        prompt_tokens = sum(len(ids) for ids in tokenized["input_ids"])

        data = [EmbeddingResponseData(index=i, embedding=emb.tolist()) for i, emb in enumerate(embeddings)]

        response = EmbeddingResponse(
            model=self.model_name,
            data=data,
            usage=UsageInfo(prompt_tokens=prompt_tokens, total_tokens=prompt_tokens),
            created=int(time.time()),
        )
        logger.log(
            TRACE, "embedding response %s: num_embeddings=%d, prompt_tokens=%d", request_id, len(data), prompt_tokens
        )
        return response
