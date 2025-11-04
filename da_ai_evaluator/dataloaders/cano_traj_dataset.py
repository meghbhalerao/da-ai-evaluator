from torch.utils.data import Dataset
import numpy as np
import torch 
from bps_torch.tools import sample_sphere_uniform
from bps_torch.bps import bps_torch
import os
import joblib

class CanoObjectTrajDataset(Dataset):
    def __init__(self, train, data_root_folder, window, keep_same_len_window: bool = False, use_object_splits: bool = False, input_language_condition: bool = False, use_first_frame_bps: bool = False, use_random_frame_bps: bool = False, use_object_keypoints: bool = False, load_ds: bool = True):
        super().__init__()
        self.train = train

        # all folder paths that are related to the dataset
        self.data_root_folder = data_root_folder
        self.obj_geo_root_folder = os.path.join(
            self.data_root_folder, "captured_objects")
        
        self.rest_object_geo_folder = os.path.join(
            self.data_root_folder, "rest_object_geo")
        
        if not os.path.exists(self.rest_object_geo_folder):
            os.makedirs(self.rest_object_geo_folder)

        self.bps_path = "./bps.pt"

        self.language_anno_folder = os.path.join(
            self.data_root_folder, "omomo_text_anno_json_data")
    
        # self.contact_npy_folder = os.path.join(self.data_root_folder, "contact_labels_npy_files")
        self.contact_npy_folder = os.path.join(
            self.data_root_folder, "contact_labels_w_semantics_npy_files")
        
        # other dataset parameters
        self.window = window
        self.keep_same_len_window = keep_same_len_window
        self.use_object_splits = use_object_splits
        self.input_language_condition = input_language_condition
        self.use_first_frame_bps = use_first_frame_bps
        self.use_random_frame_bps = use_random_frame_bps
        self.use_object_keypoints = use_object_keypoints
        self.load_ds = load_ds

        self.train_objects = [
            "largetable",
            "woodchair",
            "plasticbox",
            "largebox",
            "smallbox",
            "trashcan",
            "monitor",
            "floorlamp",
            "clothesstand",
        ]  # 10 objects

        self.test_objects = ["smalltable", "whitechair", "suitcase", "tripod"]

        self.parents = self.get_smpl_parents(use_joints24=True)

        self.num_subjects = 17

        train_subjects, test_subjects = [], []
        for i in range(1, self.num_subjects + 1):
            subject_name = f"sub{i}"
            if i <= 16:
                train_subjects.append(subject_name)
            else:
                test_subjects.append(subject_name)
        if self.train:
            mode = 'train'
        else:
            mode = 'test'

        seq_data_path = os.path.join(data_root_folder, f"{mode}_diffusion_manip_seq_joints24.p")
        processed_data_path = os.path.join(data_root_folder, f"cano_{mode}_diffusion_manip_window_" + str(self.window) + "_joints24_same_len_window.p",)
        min_max_mean_std_data_path = os.path.join(data_root_folder, "cano_min_max_mean_std_data_window_" + str(self.window) + "_joints24_same_len_window.p",)

        # some extra paths are defined if the window length is not the same
        if not self.keep_same_len_window:
            standing_flag_path = os.path.join(
                    self.data_root_folder, f"{mode}_standing_flag_joints24.p"
                )
            wrist_relative_path = os.path.join(
                self.data_root_folder, f"{mode}_wrist_relative_joints24.p"
            )
            object_static_flag_path = os.path.join(
                self.data_root_folder, f"{mode}_object_static_flag_joints24.p"
            )
            root_traj_xy_ori_path = os.path.join(
                self.data_root_folder,
                f"{mode}_interaction_root_traj_xy_ori_joints24.p",
            )
            min_max_mean_std_data_path = os.path.join(
                data_root_folder,
                "cano_min_max_mean_std_data_window_" + str(self.window) + "_joints24.p",)
        
        if self.load_ds:
            if os.path.exists(processed_data_path):
                self.window_data_dict = joblib.load(processed_data_path)
            else:
                self.data_dict = joblib.load(seq_data_path)
                self.extract_rest_pose_object_geometry_and_rotation()
                self.cal_normalize_data_input()
                joblib.dump(self.window_data_dict, processed_data_path)


    def extract_rest_pose_object_geometry_and_rotation(self):
        self.rest_pose_object_dict = {}
        for seq_idx in self.data_dict:
            seq_name = self.data_dict[seq_idx]['seq_name']
            object_name = seq_name.split('_')[1]
            if object_name in ["vacuum", "mop"]:
                continue
            
            if object_name not in self.rest_pose_object_dict:
                obj_trans = self.data_dict[seq_idx]["obj_trans"][:, :, 0]  # T X 3
                obj_rot = self.data_dict[seq_idx]["obj_rot"]  # T X 3 X 3
                obj_scale = self.data_dict[seq_idx]["obj_scale"]  # T
                
                (rest_verts,
                    obj_mesh_faces,
                    rest_pose_ori_rot,
                    rest_pose_ori_com_pos,
                    obj_trans_to_com_pos,
                ) = self.convert_rest_pose_obj_geometry(object_name, obj_scale, obj_trans, obj_rot)

                self.rest_pose_object_dict[object_name] = {}
                self.rest_pose_object_dict[object_name]["ori_rotation"] = (rest_pose_ori_rot)  # 3 X 3
                                                                           
                self.rest_pose_object_dict[object_name]["ori_trans"] = (rest_pose_ori_com_pos) # 1 X 3

                self.rest_pose_object_dict[object_name]["obj_trans_to_com_pos"] = (obj_trans_to_com_pos) # 1 X 3

                





    # for a reference to what is basis point set, please refer to https://arxiv.org/pdf/1908.09186, essentially an efficient representation of point clouds
    def prep_bps_data(self):
        n_obj = 1024
        r_obj = 1.0
        if not os.path.exists(self.bps_path):
            bps_obj = sample_sphere_uniform(n_points=n_obj, radius=r_obj).reshape(
                1, -1, 3
            )
            bps = {
                "obj": bps_obj.cpu(),
                # 'sbj': bps_sbj.cpu(),
            }
            torch.save(bps, self.bps_path)
        self.bps = torch.load(self.bps_path)
        self.bps_torch = bps_torch()
        self.obj_bps = self.bps["obj"]

    def get_smpl_parents(self, use_joints24=True):
        assert self.data_root_folder is not None, "Data root folder is None! Please enter a filepath!"
        parents_22_path = os.path.join(os.path.join(self.data_root_folder, "../", "smpl_all_models" ), "smpl_parents_22.npy")
        parents_24_path = os.path.join(os.path.join(self.data_root_folder, "../", "smpl_all_models" ), "smpl_parents_24.npy")
        if use_joints24:
            parents = np.load(parents_24_path)
        else:
            parents = np.load(parents_22_path)
        return parents
    
