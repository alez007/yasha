from pydantic import BaseModel

import ray
from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor
from ray import serve

from transformers import pipeline

@serve.deployment(num_replicas=1)
class WakeWordDetectorAgent:
    def __init__(self):
        self.model = pipeline(
            "audio-classification", model="MIT/ast-finetuned-speech-commands-v2"
        )

    def __call__(self):
        print(self.model.model.config.id2label)

        return "classifier"
