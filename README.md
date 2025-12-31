This is a proof of concept of how to deploy 2 or more models on the same GPU using [VLLm](https://github.com/vllm-project/vllm) and [Ray](https://github.com/ray-project/ray).

# requirements:
 - Nvidia GPU
 - patience to play with the gpu memory percentages to maximise the context length for the available memory

# build project for development
 - clone the project
 - copy .env.example to .env and update the env vars if necessary (huggingface token for example)
 - build the docker image: `DOCKER_BUILDKIT=1 docker build -t yasha_dev -f Dockerfile.dev .`

# start project for development
 - run the docker image: `docker run -it --rm --shm-size=8g --env-file .env --gpus all --mount type=bind,src=./,dst=/yasha --mount type=volume,dst=/yasha/.venv  -p 8265:8265 -p 8000:8000 -p 8081:8080 yasha_dev`
 - open a browser window to `http://localhost:8081` where code-server runs
 - to check the ray dashboard, open a browser window to `http://localhost:8265`
 - the OpenAI API is exposed on `http://localhost:8000`
