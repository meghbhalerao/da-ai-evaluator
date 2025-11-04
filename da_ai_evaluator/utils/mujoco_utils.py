import mujoco
import sys
import numpy as np

def convert_mujoco_humanoid_to_smplx(scene, data):
    """ mujoco humanoid to smplx conversion

    Args:
        scene (_type_): _description_
        data (_type_): _description_
    """


    # --- 1. Get the Humanoid's POSE vector (qpos) ---
    pelvis_joint_id = mujoco.mj_name2id(scene, mujoco.mjtObj.mjOBJ_JOINT, "Pelvis")
    humanoid_qpos_start_index = scene.jnt_qposadr[pelvis_joint_id]
    humanoid_qpos_vector = data.qpos[humanoid_qpos_start_index:]



    # --- 2. Get the Humanoid's VELOCITY vector (qvel) ---
    # We use 'jnt_veladr' to find its starting index in the 'qvel' array
    humanoid_qvel_start_index = scene.jnt_veladr[pelvis_joint_id]
    humanoid_qvel_vector = data.qvel[humanoid_qvel_start_index:]
    
    # --- 3. (Optional) Combine them into one large state vector ---
    humanoid_full_state_vector = np.concatenate([humanoid_qpos_vector, humanoid_qvel_vector])

    print(f"\n--- Humanoid Full State Vector ---")
    print(f"Humanoid qpos dimension: {len(humanoid_qpos_vector)}")
    print(f"Humanoid qvel dimension: {len(humanoid_qvel_vector)} (Total qvel size: {scene.nv})")
    print(f"Full combined state vector dimension: {len(humanoid_full_state_vector)}")
    








