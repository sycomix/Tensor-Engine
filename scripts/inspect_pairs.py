import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

m = torch.jit.load('tests/assets/simple_linear_pairs.pt')
sd = m.state_dict()
logger.info("type(sd): %s", type(sd))
try:
    logger.info("keys: %s", list(sd.keys()))
except Exception as e:
    logger.warning("sd is not mapping; repr: %s", repr(sd))
    # if sd is list of pairs
    try:
        logger.info("list len %d", len(sd))
        for el in sd:
            logger.info("el: %s %s", type(el), repr(el))
    except Exception as ee:
        logger.error("error inspecting sd as list: %s", ee)
