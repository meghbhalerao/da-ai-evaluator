import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import sys
import os
from augmentation.rollouts import CollectRollouts
from policies.keyboard_agent import HumanKeyboardPolicyAgent
from environments import *
from dataloaders.dataloader import convert_data_format
from trainers.trainer import Trainer
from environments.make_env import make_vec_env

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # initialize wandb
    wandb.init(config=OmegaConf.to_container(cfg, resolve=True),)

    # print the configuration to console for debugging
    print(OmegaConf.to_yaml(cfg))

    if cfg.use_collected_traj:
        print("Using collected trajectories for training.")
        trajectory_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg.environment.env_folder_path, "trajectories")
        trajectories = convert_data_format(trajectory_path)
        print(f"Converted {len(trajectories)} trajectories from {trajectory_path}.")
    else:
        raise NotImplementedError("Only collected trajectories are supported in this version! Trajectories must be collected before training.")
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

    train_envs, eval_envs = make_vec_env(
        cfg.environment.gym_id,
        cfg.environment.env_params,
        cfg.environment.num_envs,
        cfg.seed,
        cfg.environment.wrapper_class
    )


    # IL model training algorithm is to be added here 
    Trainer(train_envs, eval_envs, trajectories, cfg.seed, cfg.algorithm.name).train()



if __name__ == "__main__":
    main()
