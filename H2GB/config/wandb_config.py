from H2GB.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('cfg_wandb')
def set_cfg_wandb(cfg):
    """Weights & Biases tracker configuration.
    """

    # WandB group
    cfg.wandb = CN()

    # Use wandb or not
    cfg.wandb.use = False

    # Wandb entity name, should exist beforehand
    cfg.wandb.entity = "junhonghust"

    # Wandb project name, will be created in your team if doesn't exist already
    cfg.wandb.project = "H2GB"

    # Optional run name
    cfg.wandb.name = ""
