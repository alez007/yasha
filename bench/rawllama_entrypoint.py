"""Runs `llama-server` directly against the same models.yaml modelship reads,
bypassing Ray, to A/B against the llama_server loader with an identical
launch command. Mirrors `LlamaServerInfer._launch` — keep both in sync."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

from modelship.infer.infer_config import LlamaServerConfig, ModelLoader, ModelshipConfig, ModelUsecase
from modelship.infer.model_resolver import resolve_model_source

CONFIG_PATH = Path(os.environ.get("MSHIP_CONFIG", "/modelship/config/models.yaml"))


def main() -> int:
    binary = os.environ.get("MSHIP_LLAMA_SERVER_BIN")
    if not binary or not os.path.isfile(binary):
        print(f"MSHIP_LLAMA_SERVER_BIN must point at a llama-server executable; got {binary!r}", file=sys.stderr)
        return 2

    raw = yaml.safe_load(CONFIG_PATH.read_text())
    cfg = ModelshipConfig.model_validate(raw)
    llama_models = [m for m in cfg.models if m.loader == ModelLoader.llama_server]
    if len(llama_models) != 1:
        print(
            f"bench expects exactly one llama_server model in {CONFIG_PATH}, got {len(llama_models)}", file=sys.stderr
        )
        return 2

    m = llama_models[0]
    k = m.llama_server_config or LlamaServerConfig()

    model_path = resolve_model_source(m.model)
    print(f"rawllama resolved model -> {model_path}", flush=True)

    args = [
        binary,
        "serve",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "-m",
        model_path,
        "-c",
        str(k.n_ctx * k.parallel),
        "-b",
        str(k.n_batch),
        "-ub",
        str(k.ubatch_size),
        "-fa",
        k.flash_attn,
        "-ctk",
        k.cache_type_k,
        "-ctv",
        k.cache_type_v,
        "--parallel",
        str(k.parallel),
        "--jinja",
        "--reasoning-format",
        "auto",
        "--no-webui",
        # /v1/models reports this as "id"; no --api-key, matching vanilla llama-server.
        "--alias",
        m.name,
    ]
    # Ray only sets CUDA_VISIBLE_DEVICES for GPU-reserving actors, so a
    # num_gpus=0 deploy may still see every GPU — force no offload.
    if m.num_gpus > 0:
        args += ["-ngl", str(k.n_gpu_layers)]
        if k.tensor_split:
            args += ["-ts", ",".join(str(v) for v in k.tensor_split)]
        # Bypasses Ray's own CUDA_VISIBLE_DEVICES restriction — set it explicitly.
        # This script only benchmarks whole-GPU configs, not the loader's own
        # fractional num_gpus (shared-GPU) support.
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(int(m.num_gpus)))
    else:
        args += ["-ngl", "0"]
    if k.threads is not None:
        args += ["--threads", str(k.threads)]
    if k.chat_template:
        flag = "--chat-template-file" if os.path.isfile(k.chat_template) else "--chat-template"
        args += [flag, k.chat_template]
    if k.mmproj:
        mmproj_path = resolve_model_source(k.mmproj)
        args += ["--mmproj", mmproj_path]
    if m.usecase == ModelUsecase.embed:
        args += ["--embedding"]

    print("rawllama exec:", " ".join(args), flush=True)
    os.execvp(args[0], args)


if __name__ == "__main__":
    sys.exit(main())
