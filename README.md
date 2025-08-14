# start cluster
uv run ray start --head --dashboard-host="0.0.0.0"

# serve apps
uv run --env-file=.env serve build api:app instruct:app -o config.yaml
uv run serve run config.yaml 

