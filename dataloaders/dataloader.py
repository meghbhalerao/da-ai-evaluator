from torch.utils.data import Dataset
import pickle
import os
import sys
from packages.imitation.src.imitation.data.types import TrajectoryWithRew

def convert_data_format(data_path):
    all_trajectories = []
    for traj_file in os.listdir(data_path):
        print(f"Processing file: {traj_file}")
        if traj_file.endswith('.pkl'):
            with open(os.path.join(data_path, traj_file), 'rb') as f:
                trajectory = pickle.load(f)
                states, actions, obs, rewards = trajectory['game_states'], trajectory['actions'], trajectory['image_obs'], trajectory['rewards']
                infos = [{"state": state} for state in states]
                # making the saved trajectory compatible with the imitaiton learning package that we are using (I have assumed that I trust the IL package which I am using: https://imitation.readthedocs.io/en/latest/algorithms/bc.html, from berkeley, so might be reliable)
                trajectory = TrajectoryWithRew(obs = obs, acts = actions, infos = infos, rews = rewards,
                 terminal = True)
                print(trajectory)
                sys.exit()
                all_trajectories.append(trajectory)  
    return all_trajectories
    





        

         