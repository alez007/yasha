import os
import logging
import ray

from ray import serve
from ray.serve.config import HTTPOptions
from pydantic_yaml import parse_yaml_raw_as

from yasha.infer.infer_config import YashaConfig
from yasha.openai.api import YashaAPI, app

logger = logging.getLogger("ray")


serve.shutdown()
serve.start(
    http_options=HTTPOptions(
        host="0.0.0.0"
    )
)

def yasha_app() -> serve.Application:
    _config_file = os.path.dirname(os.path.abspath(__file__)) + "/config/models.yaml"
    _yml_conf: YashaConfig | None = None
    with open(_config_file, "r") as f:
        _yml_conf = parse_yaml_raw_as(YashaConfig, f)

    assert _yml_conf is not None

    logger.info("Init yasha app with config: %s", _yml_conf)

    return YashaAPI.options(
        name="yasha api",
        num_replicas=1,
        ray_actor_options=dict(
            num_cpus=1,
            num_gpus=2,
        )
    ).bind(_yml_conf.models)
    
    

app = yasha_app()
