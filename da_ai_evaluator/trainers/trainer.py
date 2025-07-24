import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from utils.path_utils import find_git_root
import os
import torch
from imitation.util.logger import WandbOutputFormat
POLICY_SAVE_FOLDER = os.path.join(find_git_root(),"da_ai_evaluator", "saved_stuff", "policies")
os.makedirs(POLICY_SAVE_FOLDER, exist_ok=True)

class Trainer():
    def __init__(self, env, eval_env, demonstrations, seed, algo_config):
        self.env = env
        self.eval_env = eval_env
        self.seed = seed
        self.algo_config  = algo_config
        if algo_config.name == "bc":
            self.trainer = bc.BC(observation_space=self.env.observation_space, action_space=env.action_space, demonstrations=demonstrations, rng=seed, log_wandb = True)
        else:
            raise NotImplementedError(f"Algorithm {algo_config.name} is not implemented yet!")
        self.trainer_config = algo_config.training_params
        
    def train(self):
        print("Evaluating the untrained policy.")
        reward, _ = evaluate_policy(
            self.trainer.policy,  # type: ignore[arg-type]
            self.eval_env,
            n_eval_episodes=3,
            render=True,  # comment out to speed up
        )
        print(f"Reward before training: {reward}")

        # Save the state_dict
        policy_path = os.path.join(POLICY_SAVE_FOLDER, f"{self.env.get_attr("spec")[0].id}_policy_epochs_{self.trainer_config.epochs}.pth")

        if os.path.exists(policy_path) and self.algo_config.load_pretrained_policy:
            print(f"Policy already exists at {policy_path}, hence loading pretrained policy!")
            self.trainer.policy.load_state_dict(torch.load(policy_path))
        else:
            print(f"Training a policy using Behavior Cloning for {self.trainer_config.epochs} epochs")
            self.trainer.train(n_epochs=self.trainer_config.epochs)

            print("Evaluating the trained policy.")
            reward, _ = evaluate_policy(
                self.trainer.policy,  # type: ignore[arg-type]
                self.eval_env,
                n_eval_episodes=3,
                render=True,  # comment out to speed up
            )
            print(f"Reward after training: {reward}")

            print(f"Saving the policy to {policy_path}")
            torch.save(self.trainer.policy.state_dict(), policy_path)  

        return self.trainer.policy



