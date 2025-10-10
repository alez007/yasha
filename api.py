import os
import logging
import ray

from ray import serve
from pydantic_yaml import parse_yaml_raw_as

from yasha.infer.infer_config import YashaConfig
from yasha.openai.api import YashaAPI, app

logger = logging.getLogger("ray")

if ray.is_initialized():
    ray.shutdown()

ray.init(
    address="ray://0.0.0.0:10001",
)

def yasha_app() -> serve.Application:
    _config_file = os.path.dirname(os.path.abspath(__file__)) + "/config/models.yaml"
    _instruct_model_config = {}
    _yml_conf: YashaConfig | None = None
    with open(_config_file, "r") as f:
        _yml_conf = parse_yaml_raw_as(YashaConfig, f)

    assert _yml_conf is not None

    logger.info("Init yasha app with config: %s", _yml_conf)
    
    deployment = serve.deployment(serve.ingress(app)(YashaAPI), name="yasha api")
    
    return deployment.options(
        num_replicas=1,
        ray_actor_options=dict(
            num_cpus=3,
            num_gpus=1,
        ),
    ).bind(_yml_conf.models)



serve.run(yasha_app(), route_prefix='/', name='yasha api')
