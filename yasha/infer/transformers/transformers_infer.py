from typing import Annotated
import sys
from transformers import pipeline, AutomaticSpeechRecognitionPipeline
import torch

from yasha.config.infer_config import ModelUsecase, YashaModelConfig
from fastapi import FastAPI, Form, HTTPException, Request
from http import HTTPStatus
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, EmbeddingRequest, TranscriptionRequest, TranscriptionResponse


class TransformersInfer():
    @staticmethod
    def check_transformers_support(model_config: YashaModelConfig) -> Exception|None:
        if model_config.use_vllm is False and model_config.usecase in [ModelUsecase.generate, ModelUsecase.embed]:
            raise Exception("transformers support is not implemented for embed and generate models")
    
    def create_pipeline(self) -> Exception|AutomaticSpeechRecognitionPipeline:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model_usecase = self.model_config.usecase
        model = self.model_config.model

        match model_usecase:
            case ModelUsecase.transcription:
                return pipeline(
                    task="automatic-speech-recognition",
                    model=model,
                    device=device,
                    dtype=torch.float16
                )
            case _:
                raise Exception("model usecase not supported with transformers")

    def __init__(self, model_config: YashaModelConfig):
        self.check_transformers_support(model_config)

        self.model_config = model_config

        self.pipeline = self.create_pipeline()
    
    async def start(self):
        await self.init_serving_chat()
    
    async def init_serving_chat(self):
        self.serving_chat = None

    async def create_chat_completion(self, request: ChatCompletionRequest, raw_request: Request):
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND.value,
            detail="model does not support this action")

    async def create_embedding(self, request: EmbeddingRequest, raw_request: Request):
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND.value,
            detail="model does not support this action")

    async def create_transcription(self, request: Annotated[TranscriptionRequest, Form], raw_request: Request):
        if not isinstance(self.pipeline, AutomaticSpeechRecognitionPipeline):
            raise HTTPException(status_code=HTTPStatus.NOT_FOUND.value,
                detail="model does not support this action")

        audio_data = await request.file.read()
        
        for item in self.pipeline(audio_data):
            sys.stdout.write("\033[K")
            print(item["text"], end="\r")
            if not item["partial"][0]:
                break
        
        return TranscriptionResponse(text="bla bla blablabla")


        
