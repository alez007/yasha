from pydantic import BaseModel

import ray
from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor
from ray import serve

from transformers import pipeline

@serve.deployment(num_replicas=1)
class TranslatorAgent:
    def __init__(self):
        self.model = pipeline("translation_en_to_fr", model="t5-small")

    def __call__(self, text):
        model_output = self.model(text)

        translation = model_output[0]["translation_text"]

        return translation
