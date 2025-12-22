import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

m = torch.jit.load('tests/assets/simple_linear_nested.pt')
sd = m.state_dict()
logger.info("type(sd): %s", type(sd))
logger.info("keys: %s", sorted(sd.keys()))
for k, v in sd.items():
    logger.info("key %s type %s", k, type(v))
    if isinstance(v, dict):
        logger.info("nested keys %s", sorted(list(v.keys())))
