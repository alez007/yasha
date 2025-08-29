import os
import asyncio
import time
from typing import AsyncGenerator
import ray
import requests
from fastapi import FastAPI
from ray import serve
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.yasha.agents.main import MainAgentChat
from src.yasha.agents.translator import TranslatorAgent
from src.yasha.agents.wake_word_detector import WakeWordDetectorAgent

from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor

# Import placement group APIs.
from ray.util.placement_group import (
    placement_group,
    placement_group_table,
    remove_placement_group,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.llm._internal.serve.configs.openai_api_models import ChatCompletionRequest

if ray.is_initialized():
    ray.shutdown()

ray.init(
    address="ray://0.0.0.0:10001"
)

class Instruct(BaseModel):
    input: str

app = FastAPI()

# pg = placement_group([{"CPU": 2, "GPU": 1}])

# try:
#     ray.get(pg.ready(), timeout=10)
# except Exception as e:
#     print(
#         "Cannot create a placement group for API actor"
#     )
#     print(e)

@serve.deployment(
    num_replicas=1,
    # placement_group_bundles=[{"CPU": 2, "GPU": 1}]
)
@serve.ingress(app)
class YashaAPI:
    # @staticmethod
    # async def generate_numbers(max_num: int) -> AsyncGenerator[str, None]:
    #     try:
    #         for i in range(max_num):
    #             yield str(i)
    #             await asyncio.sleep(0.1)
    #     except asyncio.CancelledError:
    #         print("Cancelled! Exiting.")
    #
    # @app.get("/")
    # async def root(self):
    #     return StreamingResponse(self.generate_numbers(10), media_type="text/plain", status_code=200)

    def __init__(self, main_agent = None, translator_agent = None, wake_word_detector_agent = None):
        self.main_agent = main_agent
        self.translator_agent = translator_agent
        # self.wake_word_detector_agent = wake_word_detector_agent

    @app.post("/instruct")
    async def instruct(self, instruct: Instruct):
        # response = await self.simple_agent.remote()
        response = await self.translator_agent.remote(instruct.input)
        # await self.wake_word_detector_agent.remote()
        return response

    @app.get("/healthy")
    async def healthy(self):
        return os.getenv('HF_TOKEN')

    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        self, request: ChatCompletionRequest
    ):
        print("===============================", request)
        response = await self.main_agent.remote(request)
        return response


app = YashaAPI.bind(
    main_agent=MainAgentChat.bind(),
    # translator_agent=TranslatorAgent.bind(),
    # wake_word_detector_agent=WakeWordDetectorAgent.bind()
)

serve.run(app, route_prefix="/api", name="api")




