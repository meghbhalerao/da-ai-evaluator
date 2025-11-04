from torch.utils.data import Dataset
import os
import numpy as np
import json

class HumanML3DDataset(Dataset):
    def __init__(self, train, data_root_folder, window, load_ds: bool = True, keep_same_len_window: bool = False):
        self.train = train
        self.data_root_folder = data_root_folder
        self.window = window
        self.load_ds = load_ds
        self.keep_same_len_window = keep_same_len_window

        self.parents = self.get_smpl_parents(use_joints24=True)

        if self.load_ds:
            train_json_path = os.path.join(self.data_root_folder, "../", "HumanML3D/humanml3d_train_seq_names.json")

            test_json_path = os.path.join(self.data_root_folder, "../", "HumanML3D/humanml3d_test_seq_names.json")

            if self.train:
                seq_names = json.load(open(train_json_path, "r"))["k_idx"]
            else:
                seq_names = json.load(open(test_json_path, "r"))["k_idx"]
            

        if keep_same_len_window:
            if self.train:
                pass

            




    # returns a list of integers
    def get_smpl_parents(self, use_joints24=True):
        assert self.data_root_folder is not None
        # Define paths for saved parent data
        data_dir = os.path.join(self.data_root_folder, "..", "smpl_all_models/")

        parents_22_path = os.path.join(data_dir, "smpl_parents_22.npy")
        parents_24_path = os.path.join(data_dir, "smpl_parents_24.npy")

        # Load the appropriate parents based on use_joints24 parameter
        if use_joints24:
            parents = np.load(parents_24_path)
        else:
            parents = np.load(parents_22_path)

        return parents


    