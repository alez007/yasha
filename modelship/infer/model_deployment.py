import time
from collections.abc import AsyncGenerator
from typing import Any

from ray import serve

from modelship.infer.base_infer import BaseInfer
from modelship.infer.infer_config import ModelLoader, ModelshipModelConfig, RawRequestProxy
from modelship.logging import configure_logging, get_logger
from modelship.metrics import (
    EMBEDDING_DURATION_SECONDS,
    GENERATION_DURATION_SECONDS,
    IMAGE_GENERATION_DURATION_SECONDS,
    MODEL_LOAD_DURATION_SECONDS,
    MODEL_LOAD_FAILURES_TOTAL,
    TRANSCRIPTION_DURATION_SECONDS,
    TTS_GENERATION_DURATION_SECONDS,
)
from modelship.openai.protocol import (
    ChatCompletionRequest,
    EmbeddingRequest,
    ImageGenerationRequest,
    SpeechRequest,
    TranscriptionRequest,
    TranslationRequest,
)

logger = get_logger("infer.deployment")


def _reap_child_processes() -> None:
    """Kill any subprocesses still alive in this actor process.

    vLLM with distributed_executor_backend='mp' forks Worker_TP* subprocesses
    before the AsyncLLM constructor returns. If init then raises (e.g. CUDA
    OOM during graph capture), those workers never get reaped — they reparent
    to PID 1 and hold their full GPU allocation until manually killed.
    """
    import contextlib

    try:
        import psutil

        children = psutil.Process().children(recursive=True)
        if not children:
            return
        logger.warning(
            "Reaping %d orphan subprocess(es): %s",
            len(children),
            [c.pid for c in children],
        )
        for c in children:
            with contextlib.suppress(psutil.NoSuchProcess):
                c.terminate()
        _, alive = psutil.wait_procs(children, timeout=5)
        for c in alive:
            with contextlib.suppress(psutil.NoSuchProcess):
                c.kill()
    except Exception:
        logger.exception("Failed to reap child subprocesses")


@serve.deployment
class ModelDeployment:
    async def __init__(self, config: ModelshipModelConfig):
        configure_logging()
        self.config = config
        start = time.monotonic()
        self.infer: BaseInfer
        try:
            if config.loader == ModelLoader.vllm:
                from modelship.infer.vllm.vllm_infer import VllmInfer

                self.infer = VllmInfer(config)
            elif config.loader == ModelLoader.transformers:
                from modelship.infer.transformers.transformers_infer import TransformersInfer

                self.infer = TransformersInfer(config)
            elif config.loader == ModelLoader.diffusers:
                from modelship.infer.diffusers.diffusers_infer import DiffusersInfer

                self.infer = DiffusersInfer(config)
            elif config.loader == ModelLoader.llama_cpp:
                from modelship.infer.llama_cpp.llama_cpp_infer import LlamaCppInfer

                self.infer = LlamaCppInfer(config)
            else:
                from modelship.infer.custom.custom_infer import CustomInfer

                self.infer = CustomInfer(config)

            await self.infer.start()
            await self.infer.warmup()
        except Exception as e:
            MODEL_LOAD_FAILURES_TOTAL.inc(tags={"model": config.name, "loader": config.loader.value})
            if infer := getattr(self, "infer", None):
                try:
                    infer.shutdown()
                except Exception:
                    logger.exception("infer.shutdown() failed during init cleanup for %s", config.name)
            _reap_child_processes()

            err_msg = f"{config.loader.value} engine init failed for '{config.name}': {e}"
            try:
                from modelship.infer.deploy_coordinator import get_or_create_coordinator

                coordinator = get_or_create_coordinator()
                app_name = serve.get_replica_context().app_name
                await coordinator.report_fatal_error.remote(app_name, err_msg)
            except Exception:
                logger.exception("Failed to report fatal error to coordinator for %s", config.name)

            raise RuntimeError(err_msg) from e
        finally:
            MODEL_LOAD_DURATION_SECONDS.observe(
                time.monotonic() - start, tags={"model": config.name, "loader": config.loader.value}
            )

    def __del__(self):
        if infer := getattr(self, "infer", None):
            try:
                infer.shutdown()
            except Exception:
                logger.exception("Failed to shutdown infer for %s", self.config.name)

    @staticmethod
    def _set_request_id(request_id: str | None) -> None:
        from modelship.logging import request_id_var

        request_id_var.set(request_id)

    async def generate(
        self,
        request: ChatCompletionRequest,
        request_headers: dict[str, str],
        disconnect_event: Any,
        request_id: str | None = None,
    ):
        self._set_request_id(request_id)
        proxy = RawRequestProxy(disconnect_event, request_headers, request_id)
        start = time.monotonic()
        result = await self.infer.create_chat_completion(request, proxy)
        GENERATION_DURATION_SECONDS.observe(time.monotonic() - start, tags={"model": self.config.name})
        if isinstance(result, AsyncGenerator):
            async for chunk in result:
                yield chunk
        else:
            yield result

    async def embed(
        self,
        request: EmbeddingRequest,
        request_headers: dict[str, str],
        disconnect_event: Any,
        request_id: str | None = None,
    ):
        self._set_request_id(request_id)
        proxy = RawRequestProxy(disconnect_event, request_headers, request_id)
        start = time.monotonic()
        result = await self.infer.create_embedding(request, proxy)
        EMBEDDING_DURATION_SECONDS.observe(time.monotonic() - start, tags={"model": self.config.name})
        if isinstance(result, AsyncGenerator):
            async for chunk in result:
                yield chunk
        else:
            yield result

    async def transcribe(
        self,
        audio_data: bytes,
        request: TranscriptionRequest,
        request_headers: dict[str, str],
        disconnect_event: Any,
        request_id: str | None = None,
    ):
        self._set_request_id(request_id)
        proxy = RawRequestProxy(disconnect_event, request_headers, request_id)
        start = time.monotonic()
        result = await self.infer.create_transcription(audio_data, request, proxy)
        TRANSCRIPTION_DURATION_SECONDS.observe(time.monotonic() - start, tags={"model": self.config.name})
        if isinstance(result, AsyncGenerator):
            async for chunk in result:
                yield chunk
        else:
            yield result

    async def translate(
        self,
        audio_data: bytes,
        request: TranslationRequest,
        request_headers: dict[str, str],
        disconnect_event: Any,
        request_id: str | None = None,
    ):
        self._set_request_id(request_id)
        proxy = RawRequestProxy(disconnect_event, request_headers, request_id)
        start = time.monotonic()
        result = await self.infer.create_translation(audio_data, request, proxy)
        TRANSCRIPTION_DURATION_SECONDS.observe(time.monotonic() - start, tags={"model": self.config.name})
        if isinstance(result, AsyncGenerator):
            async for chunk in result:
                yield chunk
        else:
            yield result

    async def speak(
        self,
        request: SpeechRequest,
        request_headers: dict[str, str],
        disconnect_event: Any,
        request_id: str | None = None,
    ):
        self._set_request_id(request_id)
        proxy = RawRequestProxy(disconnect_event, request_headers, request_id)
        start = time.monotonic()
        result = await self.infer.create_speech(request, proxy)
        TTS_GENERATION_DURATION_SECONDS.observe(time.monotonic() - start, tags={"model": self.config.name})
        if isinstance(result, AsyncGenerator):
            async for chunk in result:
                yield chunk
        else:
            yield result

    async def imagine(
        self,
        request: ImageGenerationRequest,
        request_headers: dict[str, str],
        disconnect_event: Any,
        request_id: str | None = None,
    ):
        self._set_request_id(request_id)
        proxy = RawRequestProxy(disconnect_event, request_headers, request_id)
        start = time.monotonic()
        result = await self.infer.create_image_generation(request, proxy)
        IMAGE_GENERATION_DURATION_SECONDS.observe(time.monotonic() - start, tags={"model": self.config.name})
        if isinstance(result, AsyncGenerator):
            async for chunk in result:
                yield chunk
        else:
            yield result
