import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import sys

import da_hri.environments.register


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
  pass
if __name__ == "__main__":
    main()
