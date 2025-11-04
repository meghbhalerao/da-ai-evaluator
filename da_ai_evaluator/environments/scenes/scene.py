from typing import List, Dict, Tuple, Set, Optional, Union, Any, Callable, Literal
import numpy as np
import trimesh
import os
import json
import pyrender
import sys
# Defining an object class, which is a class that is used to instantiate objects, which will be used as a part of the scene.
class Object():
    def __init__(self, 
                name: str = None, 
                obj_type: Literal["static", "moveable"] = "static",
                quat_mat: Union[List, np.array] = [0,0,0]
                ):
        self._name = name
        self.obj_type = obj_type
        self.quat_mat = quat_mat

    def set_quat(self, quat_mat):
        self.quat_mat = quat_mat
    
    def set_obj_type(self, obj_type):
        self.obj_type = obj_type
    
    @property
    def name(self) -> str:
        return self._name

class Scene():
    def __init__(self, objs: List[Object], asset_root_dir: str = None):
        assert asset_root_dir is not None, "Root asset dir must not be none!"
        self.objs = objs
        self.obj_names = [obj.name for obj in objs]
        self.asset_root_dir = asset_root_dir
        
        self.mesh_list = [trimesh.load(os.path.join(self.asset_root_dir, "rest_object_geo" , self.obj_names[i] + ".ply")) for i in range(len(self.obj_names))]

        self.start_quat_list = [self.json_to_quat(json.load(open(os.path.join(self.asset_root_dir, "rest_object_geo" , self.obj_names[i] + ".json"), 'r'))) for i in range(len(self.obj_names))]

        self.scene = pyrender.Scene()
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
        s = np.sqrt(2)/2

        camera_pose = np.array([[0.0, -s,   s,   0.3], [1.0,  0.0, 0.0, 0.0], [0.0,  s,   s,   0.35], [0.0,  0.0, 0.0, 1.0],])

        self.scene.add(camera, pose = camera_pose)

        for mesh, pose in zip(self.mesh_list, self.start_quat_list):
            mesh = pyrender.Mesh.from_trimesh(mesh)
            self.scene.add(mesh, pose = pose)
    
    def json_to_quat(self, quat_dict):
        quat_mat = np.zeros((4,4), dtype=np.float32)
        rot_mat = np.array(quat_dict['rest_pose_ori_obj_rot'])
        quat_mat[3,3] = 1
        quat_mat[0:3, 0:3] = rot_mat
        trans_vec = np.array(quat_dict['obj_trans_to_com_pos'])
        quat_mat[0:3, 3] = np.squeeze(trans_vec)
        #quat_mat[3,0:3] = np.array(quat_dict['rest_pose_ori_com_pos'])
        return quat_mat


    def render(self):
        pyrender.Viewer(self.scene, use_raymond_lighting=True)
