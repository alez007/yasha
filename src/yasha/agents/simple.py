from pydantic import BaseModel

import ray
from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor
from ray import serve

class SimpleAgent:
    def __init__(self):
        class AnswerWithExplain(BaseModel):
            problem: str
            answer: int
            explain: str

        json_schema = AnswerWithExplain.model_json_schema()

        # 2. construct a vLLM processor config.
        processor_config = vLLMEngineProcessorConfig(
            # The base model.
            model_source="unsloth/Llama-3.2-1B-Instruct",
            # vLLM engine config.
            engine_kwargs=dict(
                # Specify the guided decoding library to use. The default is "xgrammar".
                # See https://docs.vllm.ai/en/latest/serving/engine_args.html
                # for other available libraries.
                guided_decoding_backend="xgrammar",
                # Older GPUs (e.g. T4) don't support bfloat16. You should remove
                # this line if you're using later GPUs.
                dtype="half",
                # Reduce the model length to fit small GPUs. You should remove
                # this line if you're using large GPUs.
                max_model_len=1024,
            ),
            # The batch size used in Ray Data.
            batch_size=16,
            # Use one GPU in this example.
            concurrency=1,
        )

        # 3. construct a processor using the processor config.
        self.processor = build_llm_processor(
            processor_config,
            # Convert the input data to the OpenAI chat form.
            preprocess=lambda row: dict(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a math teacher. Give the answer to "
                        "the equation and explain it. Output the problem, answer and "
                        "explanation in JSON",
                    },
                    {
                        "role": "user",
                        "content": f"3 * {row['id']} + 5 = ?",
                    },
                ],
                sampling_params=dict(
                    temperature=0.3,
                    max_tokens=150,
                    detokenize=False,
                    # Specify the guided decoding schema.
                    guided_decoding=dict(json=json_schema),
                ),
            ),
            # Only keep the generated text in the output dataset.
            postprocess=lambda row: {
                "resp": row["generated_text"],
            },
        )

    async def __call__(self, processor):
        ds = ray.data.range(30)
        ds = self.processor(ds)
        ds = ds.materialize()

        for out in ds.take_all():
            print(out["resp"])
            print("==========")

        return "hello from SimpleAgent"
