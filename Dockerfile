FROM rayproject/ray:2.48.0

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

ADD . /yasha

WORKDIR /yasha

RUN uv sync --locked

ENV PATH="/yasha/.venv/bin:$PATH"

ENTRYPOINT []

RUN uv run --env-file .env -- ray start --head --dashboard-host="0.0.0.0" --port=6380

CMD ["uv run", "--env-file .env", "--", "serve run", "--route-prefix=/api", "--name=api", "api:app"]
