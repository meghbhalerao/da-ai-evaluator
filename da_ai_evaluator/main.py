import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import sys
import os
from augmentation.rollouts import CollectRollouts
from augmentation.model_augmentation import MentalModelAugmentation
from policies.keyboard_agent import HumanKeyboardPolicyAgent
from environments import *
from packages.diffusion_motion_generation.trainer_interaction_motion_diffusion import (
    run_train,
    run_validation)

from packages.diffusion_motion_generation.sample import run_sample

import torch


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # print the configuration to console for debugging
    # for transformer based diffusion model based motion modeling or generation, we branch out the code from here itself and go to the diffusion model training, for better readability, but still this main file is the entry point for the code since it is good to have a single entry point codebase for easier debugging

    OmegaConf.set_struct(
        cfg, False
    )  # this is done to allow keys that are not in the original config dict to be added, in case such a functionality is needed
    if cfg.algorithm.name == "diff_trans":

        cfg.algorithm.save_dir = os.path.join(
            cfg.algorithm.project, cfg.algorithm.exp_name)
        
        device = torch.device(
            f"cuda:{cfg.algorithm.device}" if torch.cuda.is_available() else "cpu")
        

        if cfg.algorithm.phase in ("validation", "valid"):
            run_validation(cfg.algorithm, device)
        elif cfg.algorithm.phase in ("training", "train"):
            run_train(cfg.algorithm, device)
        elif cfg.algorithm.phase in ("sample"):
            run_sample(cfg.algorithm, device)
        else:
            raise ValueError(f"Phase {cfg.algorithm.phase} does not have support yet!")

    sys.exit(0)

    from dataloaders.dataloader import convert_data_format
    from trainers.trainer import TrainerAllTypes
    from environments.make_env import make_vec_env
    from utils.viz_utils import VizPolicy
    from flappy_bird_gym.utils.group_trajs import (
        group_trajectories,
        add_pseudo_latent_state,
    )

    if cfg.use_collected_traj:
        print("Using collected trajectories for training.")
        trajectory_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            cfg.environment.env_folder_path,
            "trajectories",
        )
        trajectories = convert_data_format(
            trajectory_path, env_id=cfg.environment.gym_id
        )
        print(f"Converted {len(trajectories)} trajectories from {trajectory_path}.")
    else:
        raise NotImplementedError(
            "Only collected trajectories are supported in this version! Trajectories must be collected before training."
        )
        # Make the policy that is going to collect the rolouts
        if cfg.policy.name == "keyboard_agent":
            policy = HumanKeyboardPolicyAgent(cfg)
        else:
            raise NotImplementedError(
                f"Policy {cfg.policy.name} is not implemented yet!"
            )

        # collect rollouts
        CollectRollouts(
            env_id=cfg.environment.gym_id,
            env_params=cfg.environment.env_params,
            policy=policy,
            num_episodes=cfg.rollout.num_episodes,
        ).collect()

    train_envs, eval_envs = make_vec_env(
        cfg.environment.gym_id,
        cfg.environment.env_params,
        cfg.environment.num_envs,
        cfg.seed,
        cfg.environment.wrapper_class,
    )

    # for proof of concept reasons, we will essentially use an aritficial grouping of trajectories to simulate for different human demonstrators - for now, in the future, we will actually collected different data from different human demonstrators
    grouped_trajs = group_trajectories(
        trajectories,
        group_criteria=cfg.group_criteria,
        group_size=cfg.group_size,
        num_groups=cfg.num_groups,
    )
    grouped_trajs = add_pseudo_latent_state(grouped_trajs)
    # perform AMM learning for each of the grouped trajectories, using variational inference algroriths

    policy = TrainerAllTypes(
        train_envs, eval_envs, grouped_trajs, cfg.seed, cfg.algorithm, cfg
    ).train()
    wandb.finish()

    if cfg.policy.viz_policy_after_train:
        # visualize the trained policy
        VizPolicy(policy, cfg.environment).visualize()

    if cfg.augmentation.name == "mentalmodel":
        augmented_trajectories = MentalModelAugmentation(
            policy, eval_envs, cfg.augmentation, num_initial_trajectories=10
        ).augment()
    else:
        raise NotImplementedError(
            f"Augmentation type {cfg.augmentation.name} not implemented yet!"
        )

    # visualization mechanism of the augmented trajectories in the environment

    wandb.finish()


if __name__ == "__main__":
    main()
