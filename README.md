# start cluster(s)
uv run --env-file .env -- ray start --head --dashboard-host="0.0.0.0" --port=6379
uv run --env-file .env -- ray start --head --dashboard-host="0.0.0.0" --port=6380

# serve apps
uv run --env-file .env -- serve run --route-prefix=/api --name=api api:app

