This is a proof of concept of how to deploy 2 or more models on the same GPU using [VLLm](https://github.com/vllm-project/vllm) and [Ray](https://github.com/ray-project/ray).

# requirements:
 - Nvidia GPU
 - patience to play with the gpu memory percentages to maximise the context length for the available memory

# build project for development
 - clone the project
 - copy .env.example to .env and update the env vars if necessary (huggingface token for example)
 - build the docker image: `docker build -t yasha_dev -f Dockerfile.dev .`

# start project for development
 - run the docker image: `docker run -it --rm --shm-size=8g --env-file .env --gpus all --mount type=bind,src=./,dst=/yasha  -p 8265:8265 -p 8000:8000 yasha_dev`
 - inside the container run: `serve run --route-prefix=/ --name='yasha api' --working-dir=/yasha --address=0.0.0.0:${RAY_REDIS_PORT} start:app`
 - use vscode dev containers extension (or similar for other ides) to connect to container and start developing
 - to check the ray dashboard, open a browser window to `http://localhost:8265`
 - the OpenAI API is exposed on `http://localhost:8000`
 - wioming protocol openai docker: `docker run -it -p 10300:10300 -e WYOMING_URI="tcp://0.0.0.0:10300" -e WYOMING_LOG_LEVEL="INFO" -e WYOMING_LANGUAGES="en" -e STT_OPENAI_URL="http://0.0.0.0:8000/v1" -e STT_MODELS="whisper" -e STT_STREAMING_MODELS="whisper" -e STT_BACKEND="OPENAI" -e TTS_OPENAI_URL="http://0.0.0.0:8000/v1" -e TTS_MODELS="kokoro" -e TTS_STREAMING_MODELS="kokoro" -e TTS_VOICES="af_heart" -e TTS_SPEED="1.0" -e TTS_BACKEND="OPENAI" ghcr.io/roryeckel/wyoming_openai:latest`
