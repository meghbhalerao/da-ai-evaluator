from environments.scenes.scene import Scene, Object
from typing import List, Literal

def merge_dicts_strict(d1: dict, d2: dict) -> dict:
    overlap = d1.keys() & d2.keys()
    if overlap:
        raise ValueError(f"Duplicate keys found: {overlap}")
    return {**d1, **d2}


def make_scene(env_cfg):
    obj_list = env_cfg.obj_list
    obj_type = env_cfg.obj_type
    obj_list = [Object(o,t) for o, t in zip(obj_list, obj_type)]
    scene = Scene(obj_list, env_cfg.asset_root_dir)
    return scene


def find_all_indices(input_list, element_to_find):
    """Returns a list of all indices using a list comprehension."""
    return [
        index
        for index, element in enumerate(input_list)
        if element == element_to_find
    ]