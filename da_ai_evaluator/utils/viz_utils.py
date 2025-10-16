import gymnasium as gym
import sys 
import torch

class VizPolicy():
    def __init__(self, policy, env_config, num_traj_viz = 10):
        self.policy = policy
        self.viz_env = gym.make(env_config.gym_id, **env_config.env_params)
        self.num_traj_viz = num_traj_viz

    def visualize(self):
        for _ in range(self.num_traj_viz):
            done = False
            obs, _ = self.viz_env.reset()
            print("Device of policy is", self.policy.device)
            #device = self.policy.device
            # send obs to device of policy, after converting it to a torch tensor
            #obs = torch.tensor(obs, device = device)
            while not done:
                self.viz_env.render()
                a, _ = self.policy.predict(obs)
                next_obs, rew, truncated, done, info = self.viz_env.step(a)
                obs = next_obs #torch.tensor(next_obs, device = device)

