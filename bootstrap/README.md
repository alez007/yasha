# mship

Installer for [modelship](https://github.com/modelship-ai/modelship) — the
production backend for self-hosted agents.

```bash
pipx install mship          # or: uv tool install mship / pip install mship
```

Then pick the node's role — the install command is the same on every machine:

```bash
mship deploy --cuda  --config models.yaml   # NVIDIA GPU node
mship deploy --cpu   --config models.yaml   # CPU node (includes vLLM CPU)
mship deploy --metal --config models.yaml   # Apple Silicon
mship deploy --thin  --config models.yaml   # coordinator/head only, no capacity
```

Nothing to pin and nothing to match across machines. The first run of a variant
sets itself up; after that it starts straight through. Every node lands on the
same footing, so a new one joins the cluster by running the same command.

Alpine and other musl distros are unsupported — use a glibc distro or the Docker
images.

See the [installation docs](https://docs.model-ship.ai/installation/) for
platform prerequisites and the [documentation](https://docs.model-ship.ai/) for
everything else.
