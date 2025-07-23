from torch.utils.data import Dataset
import pickle
import os
import sys
from packages.imitation.src.imitation.data.types import TrajectoryWithRew

import numpy as np

from flappy_bird_gym.envs.game_logic import PIPE_WIDTH, PIPE_HEIGHT
from flappy_bird_gym.envs.game_logic import PLAYER_WIDTH, PLAYER_HEIGHT

def get_obs_flappy_bird_simple(game_state_list, screen_size, normalize_obs = True):
    flapp_bird_simple_obs_list = []
    for game_state in game_state_list:
        upper_pipes = game_state["upper_pipes"]
        lower_pipes = game_state["lower_pipes"]
        player_x = game_state["player_x"]
        player_y = game_state["player_y"]
        up_pipe = low_pipe = None
        h_dist = 0
        for up_pipe, low_pipe in zip(upper_pipes, lower_pipes):
            h_dist = (low_pipe["x"] + PIPE_WIDTH / 2 - (player_x - PLAYER_WIDTH / 2))
            h_dist += 3  # extra distance to compensate for the buggy hit-box
            if h_dist >= 0:
                break

        upper_pipe_y = up_pipe["y"] + PIPE_HEIGHT
        lower_pipe_y = low_pipe["y"]
        player_y = player_y

        v_dist = (upper_pipe_y + lower_pipe_y) / 2 - (player_y + PLAYER_HEIGHT/2)

        if normalize_obs:
            h_dist /= screen_size[0]
            v_dist /= screen_size[1]

        flapp_bird_simple_obs_list.append(np.array([
            h_dist,
            v_dist,
        ]))
    return np.array(flapp_bird_simple_obs_list, dtype=np.float32)


def convert_data_format(data_path, env_id, screen_size=(288, 512), normalize_obs=True):
    all_trajectories = []
    for traj_file in sorted(os.listdir(data_path)):
        print(f"Processing file: {traj_file}")
        if traj_file.endswith('.pkl'):
            with open(os.path.join(data_path, traj_file), 'rb') as f:
                trajectory = pickle.load(f)
                states, actions, obs, rewards = trajectory['game_states'], np.array(trajectory['actions']), np.array(trajectory['image_obs']), np.array(trajectory['rewards'], dtype=np.float32)
                
                get_obs_flappy_bird_simple
                simple_obs = get_obs_flappy_bird_simple(states, screen_size=screen_size, normalize_obs=normalize_obs)
                infos = [{"state": state, "simple_obs": simple_obs} for state, simple_obs in zip(states, simple_obs)][:-1]

                # making the saved trajectory compatible with the imitaiton learning package that we are using (I have assumed that I trust the IL package which I am using: https://imitation.readthedocs.io/en/latest/algorithms/bc.html, from berkeley, so might be reliable)
                print(f"Shape of obs: {obs.shape}, simple obs: {simple_obs.shape} actions: {actions.shape}, infos: {len(infos)}, rewards: {rewards.shape}") # printing the shapes of individual items in the trajectory class
                if env_id == "FlappyBird-v0":
                    trajectory = TrajectoryWithRew(obs = simple_obs, acts = actions, infos = infos, rews = rewards, terminal = True)
                elif env_id == "FlappyBirdRGB-v0":
                    trajectory = TrajectoryWithRew(obs = obs, acts = actions, infos = infos, rews = rewards, terminal = True)
                else:
                    raise NotImplementedError(f"Environment {env_id} is not supported for data conversion in this version!")
                all_trajectories.append(trajectory)  
            
    return all_trajectories
    





        

         