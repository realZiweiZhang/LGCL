from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

from COOP.trainers import coop

def build_coop_encoder(args)

    cfg = setup_cfg(args)
        if cfg.SEED >= 0:
            print("Setting fixed seed: {}".format(cfg.SEED))
            set_random_seed(cfg.SEED)
        setup_logger(cfg.OUTPUT_DIR)
    
    model = coop.CustomCLIP(c)