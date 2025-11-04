from torch.utils.data import Dataset
import os
from typing import List, Literal
import pickle
from utils.utils import find_all_indices

class TaskTrajectoryDataset(Dataset):
    def __init__(self, root_data_folder: str = None, phase: Literal["navigation", "interaction"] = "navigation", data_in_mem: bool = True):
        super().__init__()
        assert root_data_folder is not None, "Root data folder can't be none!"

        self.root_data_folder = root_data_folder
        self.num_data_phase = 0
        # get the pickle filepath list
        if data_in_mem: # load data in memory flag
            data_dict_list = []
            for path in os.listdir(root_data_folder):
                data_file = pickle.load(open(os.path.join(root_data_folder, path, "nav_interact_res.pkl")), "rb")

                idxs_phase = find_all_indices(data_file['raw_results_list'], phase)

                for idx in idxs_phase:
                    data_dict_list.append(data_file['raw_results_list'][idx])   
                    self.num_data_phase+=1
        else:
            raise NotImplementedError("Data loading from disk not implemented yet!")


    def __len__(self):
        return self.num_data_phase
    
    def __getitem__(self, index):
        return 
    
        

            

        