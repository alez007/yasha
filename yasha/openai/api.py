import logging
import time
from typing import Annotated

from pydantic import BaseModel, Field
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest, ChatCompletionResponse
from vllm.entrypoints.openai.speech_to_text.protocol import TranscriptionRequest, TranslationRequest, TranscriptionResponse, TranslationResponse
from vllm.entrypoints.pooling.embed.protocol import EmbeddingRequest, EmbeddingResponse
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from http import HTTPStatus
from ray import serve
from ray.serve.handle import DeploymentHandle

from yasha.infer.infer_config import ModelUsecase, RawSpeechResponse, RequestWatcher, SpeechRequest


logger = logging.getLogger("ray.serve")


def build_app():
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(HTTPException)
    async def log_http_exception(request: Request, exc: HTTPException):
        logger.warning("%s %s -> %s: %s", request.method, request.url.path, exc.status_code, exc.detail)
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.exception_handler(Exception)
    async def log_unhandled_exception(request: Request, exc: Exception):
        logger.exception("%s %s -> 500: %s", request.method, request.url.path, exc)
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    return app


app = build_app()


class OpenAiModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "yasha"


class OpenaiModelList(BaseModel):
    object: str = "list"
    data: list[OpenAiModelCard] = []


def _error_response(result: ErrorResponse) -> JSONResponse:
    return JSONResponse(content=result.model_dump(), status_code=result.error.code if result.error else 500)


@serve.deployment
@serve.ingress(app)
class YashaAPI:
    def __init__(self, model_handles: dict[str, tuple[DeploymentHandle, ModelUsecase]]):
        self.models = {name: handle for name, (handle, _) in model_handles.items()}
        self.model_list = [
            OpenAiModelCard(id=name)
            for name, (_, usecase) in model_handles.items()
            if usecase is ModelUsecase.generate
        ]

    def _get_handle(self, model_name: str | None) -> DeploymentHandle:
        if model_name is None or model_name not in self.models:
            raise HTTPException(status_code=HTTPStatus.NOT_FOUND.value, detail="model not found")
        return self.models[model_name]

    async def _handle_response(self, response_gen, stream_media_type: str = "text/event-stream"):
        first = await response_gen.__anext__()

        if isinstance(first, ErrorResponse):
            return _error_response(first)

        if isinstance(first, RawSpeechResponse):
            return Response(content=first.audio, media_type=first.media_type)

        if isinstance(first, (ChatCompletionResponse, EmbeddingResponse, TranscriptionResponse, TranslationResponse)):
            return JSONResponse(content=first.model_dump())

        # streaming — first chunk already consumed, chain it back
        async def _stream():
            yield first
            async for chunk in response_gen:
                yield chunk

        return StreamingResponse(content=_stream(), media_type=stream_media_type)

    @app.get("/v1/models", response_model=OpenaiModelList)
    async def list_models(self):
        return OpenaiModelList(data=self.model_list)

    @app.get("/v1/models/{model}", response_model=OpenAiModelCard)
    async def model_info(self, model: str) -> OpenAiModelCard:
        for card in self.model_list:
            if card.id == model:
                return card
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND.value, detail="model not found")

    @app.post("/v1/chat/completions")
    async def create_chat_completion(self, request: ChatCompletionRequest, raw_request: Request):
        handle = self._get_handle(request.model)
        watcher = RequestWatcher(raw_request)
        headers = dict(raw_request.headers)
        # Force evaluation of any lazy pydantic validators (e.g. ValidatorIterator in
        # tool_calls fields inside raw message dicts) before crossing the Ray process
        # boundary, where pickling fails. model_dump_json uses pydantic's Rust serializer
        # which recurses into raw dict values and consumes iterators; model_dump does not.
        request = ChatCompletionRequest.model_validate_json(request.model_dump_json())
        response_gen = handle.generate.options(stream=True).remote(request, headers, watcher.event)
        return await self._handle_response(response_gen)

    @app.post("/v1/embeddings")
    async def create_embeddings(self, request: EmbeddingRequest, raw_request: Request):
        handle = self._get_handle(request.model)
        watcher = RequestWatcher(raw_request)
        headers = dict(raw_request.headers)
        # EmbeddingRequest is a UnionType — force resolution before Ray pickle boundary.
        request = type(request).model_validate_json(request.model_dump_json())
        response_gen = handle.embed.options(stream=True).remote(request, headers, watcher.event)
        return await self._handle_response(response_gen)

    @app.post("/v1/audio/transcriptions")
    async def create_transcriptions(
        self, request: Annotated[TranscriptionRequest, Form()], raw_request: Request
    ):
        handle = self._get_handle(request.model)
        watcher = RequestWatcher(raw_request)
        headers = dict(raw_request.headers)
        # Read audio bytes before crossing process boundary — UploadFile is not serializable.
        # The bytes are passed separately; the request is reconstructed without the file field.
        audio_data = await request.file.read()
        request_no_file = TranscriptionRequest.model_construct(**request.model_dump(exclude={"file"}))
        response_gen = handle.transcribe.options(stream=True).remote(audio_data, request_no_file, headers, watcher.event)
        return await self._handle_response(response_gen)

    @app.post("/v1/audio/translations")
    async def create_translations(
        self, request: Annotated[TranslationRequest, Form()], raw_request: Request
    ):
        handle = self._get_handle(request.model)
        watcher = RequestWatcher(raw_request)
        headers = dict(raw_request.headers)
        # Read audio bytes before crossing process boundary — UploadFile is not serializable.
        # The bytes are passed separately; the request is reconstructed without the file field.
        audio_data = await request.file.read()
        request_no_file = TranslationRequest.model_construct(**request.model_dump(exclude={"file"}))
        response_gen = handle.translate.options(stream=True).remote(audio_data, request_no_file, headers, watcher.event)
        return await self._handle_response(response_gen)

    @app.post("/v1/audio/speech")
    async def create_speech(self, request: SpeechRequest, raw_request: Request):
        handle = self._get_handle(request.model)
        watcher = RequestWatcher(raw_request)
        headers = dict(raw_request.headers)
        response_gen = handle.speak.options(stream=True).remote(request, headers, watcher.event)
        return await self._handle_response(response_gen)
