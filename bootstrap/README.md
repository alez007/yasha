# mship

Installer for [modelship](https://github.com/modelship-ai/modelship) — the
production backend for self-hosted agents.

```bash
pipx install mship          # or: uv tool install mship / pip install mship
```

Then pick the node's role. The variant is chosen at run time, not at install
time, so the install command is the same on every machine:

```bash
mship deploy --cuda  --config models.yaml   # NVIDIA GPU node
mship deploy --cpu   --config models.yaml   # CPU node (includes vLLM CPU)
mship deploy --metal --config models.yaml   # Apple Silicon
mship deploy --thin  --config models.yaml   # coordinator/head only, no capacity
```

On first use of a variant this provisions a CPython 3.12.10 environment under
`~/.modelship/envs/<variant>/` from hash-pinned dependency lists shipped in this
package, then runs `mship-engine` inside it. Every node in a cluster therefore
runs an identical interpreter and dependency set, which Ray requires.

Subsequent runs re-verify that environment in milliseconds and exec straight
through.

Requires glibc — `ray` publishes no musllinux wheels, so Alpine is unsupported.
Use a glibc distro or the Docker images.

See the [installation docs](https://docs.model-ship.ai/installation/) for
platform prerequisites and the [documentation](https://docs.model-ship.ai/) for
everything else.
