This is a proof of concept of how to deploy 2 or more models on the same GPU using [VLLm](https://github.com/vllm-project/vllm) and [Ray](https://github.com/ray-project/ray).

# requirements:
 - Nvidia GPU
 - patience to play with the gpu memory percentages to maximise the context length for the available memory

# build project
 - uv sync
 - docker compose build

# start project
docker compose up

# deploy the openai api
uv run -- api.py

