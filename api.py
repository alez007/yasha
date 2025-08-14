import asyncio
import time
from typing import AsyncGenerator
import ray
import requests
from fastapi import FastAPI
from ray import serve
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.yasha.agents.simple import SimpleAgent
from src.yasha.agents.translator import TranslatorAgent

from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor

# ray.init(num_cpus=6, num_gpus=0, dashboard_host="0.0.0.0")

class Instruct(BaseModel):
    input: str

app = FastAPI()

@serve.deployment(num_replicas=1, name="api")
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

    def __init__(self, simple_agent = None, translator_agent = None):
        self.simple_agent = simple_agent
        self.translator_agent = translator_agent

    @app.post("/instruct")
    async def instruct(self, instruct: Instruct):
        # response = await self.simple_agent(self.processor)
        response = await self.translator_agent.remote(instruct.input)
        return response


app = YashaAPI.bind(
    # simple_agent=SimpleAgent.bind(),
    translator_agent=TranslatorAgent.bind(),
)
# serve.run(app, route_prefix="/v1", blocking=True)


