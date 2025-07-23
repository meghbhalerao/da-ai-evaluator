import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env


class Trainer():
    def __init__(self, env, eval_env, demonstrations, seed, algorithm):
        self.env = env
        self.eval_env = eval_env
        self.seed = seed
        if algorithm == "bc":
            self.trainer = bc.BC(observation_space=self.env.observation_space, action_space=env.action_space, demonstrations=demonstrations, rng=seed,)
        else:
            raise NotImplementedError(f"Algorithm {algorithm} is not implemented yet!")
        
    def train(self):
        print("Evaluating the untrained policy.")
        reward, _ = evaluate_policy(
            self.trainer.policy,  # type: ignore[arg-type]
            self.eval_env,
            n_eval_episodes=3,
            render=True,  # comment out to speed up
        )

        print(f"Reward before training: {reward}")

        print("Training a policy using Behavior Cloning")
        self.trainer.train(n_epochs=1)

        print("Evaluating the trained policy.")
        reward, _ = evaluate_policy(
            self.trainer.policy,  # type: ignore[arg-type]
            self.eval_env,
            n_eval_episodes=3,
            render=True,  # comment out to speed up
        )
        print(f"Reward after training: {reward}")



