# mship

Installer for [modelship](https://github.com/modelship-ai/modelship) — the
production backend for self-hosted agents.

```bash
pipx install mship          # or: uv tool install mship / pip install mship
```

Then pick the node's role once — the install command is the same on every machine:

```bash
mship bootstrap --cuda      # NVIDIA GPU node
mship bootstrap --cpu       # CPU node (includes vLLM CPU)
mship bootstrap --metal     # Apple Silicon
mship bootstrap --thin      # coordinator/head only, no capacity

mship deploy --config models.yaml
```

`bootstrap` installs a pinned Python 3.12.10 environment for that role and records
it, so `deploy` needs no variant flag afterwards and installs nothing. Nothing to
pin and nothing to match across machines: every node lands on the same footing, so
a new one joins the cluster by running the same two commands. After upgrading
`mship`, re-run `mship bootstrap` to move the environment with it.

Alpine and other musl distros are unsupported — use a glibc distro or the Docker
images.

See the [installation docs](https://docs.model-ship.ai/installation/) for
platform prerequisites and the [documentation](https://docs.model-ship.ai/) for
everything else.
