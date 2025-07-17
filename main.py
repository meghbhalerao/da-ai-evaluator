import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import sys
from augmentation.rollouts import CollectRollouts
from policies.keyboard_agent import HumanKeyboardPolicyAgent
from environments import *

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # initialize wandb
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # print the configuration to console for debugging
    print(OmegaConf.to_yaml(cfg))

    # Make the policy that is going to collect the rolouts
    if cfg.policy.name == "keyboard_agent":
        policy = HumanKeyboardPolicyAgent(cfg)
    else:
        raise NotImplementedError(f"Policy {cfg.policy.name} is not implemented yet!")

    # collect rollouts 
    CollectRollouts(
        env_id=cfg.environment.gym_id,
        env_params=cfg.environment.env_params,
        policy=policy,
        num_episodes=cfg.rollout.num_episodes
    ).collect()


if __name__ == "__main__":
    main()
