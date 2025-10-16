import numpy as np
import sys
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms import bc, dagger, density, mce_irl, sqil, preference_comparisons
from stable_baselines3 import PPO
from stable_baselines3 import PPO
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from utils.path_utils import find_git_root
import os
import torch
from imitation.util.logger import WandbOutputFormat
from omegaconf import DictConfig, OmegaConf

POLICY_SAVE_FOLDER = os.path.join(find_git_root(), "da_ai_evaluator", "saved_stuff", "policies")
os.makedirs(POLICY_SAVE_FOLDER, exist_ok=True)

class TrainerAllTypes():
    def __init__(self, env, eval_env, grouped_demos, seed, algo_config, cfg):
        self.env = env
        self.eval_env = eval_env
        self.seed = seed
        self.algo_config = algo_config
        self.cfg = cfg
        self.policy_viz_params = self.algo_config.policy_viz_params

 
        self.trainer_list = [Trainer(env, eval_env, demonstrations, seed, algo_config, cfg, self.policy_viz_params) for demonstrations in grouped_demos]


    def train(self):
        trained_policies = []
        for idx, trainer in enumerate(self.trainer_list):
            print(f"Training policy for group {idx+1}/{len(self.trainer_list)}")
            trainer.train()
            trained_policies.append(trainer.trainer.policy)
            # sys.exit()
        return trained_policies


class Trainer():
    def __init__(self, env, eval_env, demonstrations, seed, algo_config, cfg, policy_viz_params):
        self.env = env
        self.eval_env = eval_env
        self.seed = seed
        self.algo_name = algo_config.name.lower()
        self.algo_config  = algo_config
        self.policy_viz_params = policy_viz_params

        run = wandb.init(config=OmegaConf.to_container(cfg, resolve=True), sync_tensorboard=True, monitor_gym=True)
        
        if algo_config.name.lower() == "bc":
            self.trainer = bc.BC(observation_space=self.env.observation_space, action_space=env.action_space, demonstrations=demonstrations, rng=seed, log_wandb = True, batch_size = self.algo_config.training_params.batch_size)

        elif algo_config.name.lower() == "dagger":
            bc_trainer = bc.BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            rng=rng,)

            with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
                print(tmpdir)
                dagger_trainer = dagger.SimpleDAggerTrainer(
                    venv=env,
                    scratch_dir=tmpdir,
                    expert_policy=expert,
                    bc_trainer=bc_trainer,
                    rng=rng,
                )

        elif algo_config.name.lower() == "gail":
            # logger_ =  WandbOutputFormat()
            learner = PPO(env=env,
                policy=MlpPolicy,
                batch_size=self.algo_config.rl_training_params.batch_size,
                ent_coef=self.algo_config.rl_training_params.ent_coef,
                learning_rate=self.algo_config.rl_training_params.learning_rate,
                gamma=0.95,
                n_epochs=self.algo_config.rl_training_params.epochs,
                seed=self.algo_config.seed,
                tensorboard_log=f"runs/{run.id}", verbose=1, device = 'cuda')
            
            # print("Learning dummy ppo policy")
            # learner.learn(total_timesteps=10000000)  # to initialize the policy
            # sys.exit()
            reward_net = BasicRewardNet(
                observation_space=env.observation_space,
                action_space=env.action_space,
                normalize_input_layer=RunningNorm,
            )

            self.trainer = GAIL(
                demonstrations=demonstrations,
                demo_batch_size=self.algo_config.training_params.batch_size,
                gen_replay_buffer_capacity=self.algo_config.training_params.gen_replay_buffer_capacity,
                n_disc_updates_per_round=self.algo_config.training_params.n_disc_updates_per_round,
                venv = env,
                eval_venv = self.eval_env,
                gen_algo=learner,
                reward_net=reward_net,
                allow_variable_horizon=True,
                log_wandb=True,
                policy_viz_params = self.policy_viz_params,
                gen_train_timesteps = self.algo_config.rl_training_params.gen_train_timesteps)
            
        else:
            raise NotImplementedError(f"Algorithm {algo_config.name} is not implemented yet!")

        self.num_eval_episodes = algo_config.evaluation_params.num_eval_episodes

        self.trainer_config = algo_config.training_params
        
        
    def train(self):
        print("Evaluating the untrained policy.")
        mean_rew, std_rew, mean_score, std_score = evaluate_policy(
            self.trainer.policy,  # type: ignore[arg-type]
            self.eval_env,
            n_eval_episodes= self.num_eval_episodes,
            render = False,  # comment out to speed up
            policy_viz_params = self.policy_viz_params
        )
        
        print(f"Mean and stddev reward before training: {mean_rew}, {std_rew}")
        print(f"Mean and stddev score before training: {mean_score}, {std_score}")

        # Save the state_dict
        if self.algo_name == 'gail':
            policy_path = os.path.join(POLICY_SAVE_FOLDER, f"{self.env.get_attr("spec")[0].id}_{self.algo_name}_policy_timesteps_{self.trainer_config.total_timesteps}.pth")
        elif self.algo_name == 'bc':
            policy_path = os.path.join(POLICY_SAVE_FOLDER, f"{self.env.get_attr("spec")[0].id}_{self.algo_name}_policy_epochs_{self.trainer_config.epochs}.pth")
        

        if os.path.exists(policy_path) and self.algo_config.load_pretrained_policy:
            print(f"Policy already exists at {policy_path}, hence loading pretrained policy!")
            self.trainer.policy.load_state_dict(torch.load(policy_path))
        else:
            if self.algo_name == 'gail':
                print(f"Training a policy using {self.algo_name} for {self.trainer_config.total_timesteps} steps")
                self.trainer.train(total_timesteps=self.trainer_config.total_timesteps)
            elif self.algo_name == 'bc':
                print(f"Training a policy using {self.algo_name} for {self.trainer_config.epochs} epochs")
                self.trainer.train(n_epochs=self.trainer_config.epochs)

            print("Evaluating the trained policy.")
            mean_rew, std_rew, mean_score, std_score = evaluate_policy(
                self.trainer.policy,  # type: ignore[arg-type]
                self.eval_env,
                n_eval_episodes= self.num_eval_episodes,
                render=False,
                policy_viz_params = self.policy_viz_params  # comment out to speed up
            )

            print(f"Mean and stddev reward after training: {mean_rew}, {std_rew}")
            print(f"Mean and stddev score after training: {mean_score}, {std_score}")

            print(f"Saving the policy to {policy_path}")
            torch.save(self.trainer.policy.state_dict(), policy_path)  

        return self.trainer.policy



