import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

m = torch.jit.load('tests/assets/simple_linear_nested.pt')
sd = m.state_dict()
logger.info('repr sd: %s', repr(sd))
try:
    keys = list(sd.keys())
    logger.info('keys %s', keys)
    for k in keys:
        v = sd[k]
        logger.info('key %s type(v): %s', k, type(v))
        if isinstance(v, dict):
            logger.info('innerkeys %s', list(v.keys()))
except Exception as e:
    logger.error('Exception accessing sd keys: %s', e)
