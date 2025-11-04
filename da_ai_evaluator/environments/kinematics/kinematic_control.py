import mujoco
import mujoco.viewer
import numpy as np
import time
import sys

class MuJoCoKinematicViewer:
    """
    A class to load a MuJoCo model, perform kinematic manipulations,
    and view the simulation.
    """
    def __init__(self, xml_path: str, free_joint_name: str = 'Pelvis'):
        """
        Initializes the viewer by loading the model and data.

        Args:
            xml_path (str): Path to the MuJoCo XML file.
            free_joint_name (str): Name of the free joint to control (default: 'Pelvis').
        """
        self.model = None
        self.data = None
        self.pos_indices = None
        self.quat_indices = None
        self.free_joint_name = free_joint_name

        try:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
            self.data = mujoco.MjData(self.model)
        except Exception as e:
            print(f"Error loading XML '{xml_path}': {e}")
            raise # Re-raise the exception to indicate failure
        
        # Find the free joint's qpos indices
        try:
            joint_id = self.model.joint(self.free_joint_name).id
            qpos_adr = self.model.jnt_qposadr[joint_id]
            # First 3 are position (x, y, z)
            self.pos_indices = slice(qpos_adr, qpos_adr + 3)
            # Next 4 are orientation quaternion (w, x, y, z)
            self.quat_indices = slice(qpos_adr + 3, qpos_adr + 7)
            print(f"Successfully loaded model and found free joint '{self.free_joint_name}'.")
        except KeyError:
            print(f"Error: Free joint named '{self.free_joint_name}' not found in the model.")
            raise # Re-raise the exception

    def set_humanoid_pose(self, position: np.ndarray = None, orientation_quat: np.ndarray = None):
        """
        Sets the position and/or orientation of the humanoid's root kinematically.

        Args:
            position (np.ndarray, optional): Target [x, y, z] position. Defaults to None (no change).
            orientation_quat (np.ndarray, optional): Target [w, x, y, z] quaternion. Defaults to None (no change). Ensure it's normalized if provided.
        """
        if self.model is None or self.data is None:
            print("Error: Model not loaded properly.")
            return

        qpos_changed = False
        if position is not None:
            if len(position) == 3:
                self.data.qpos[self.pos_indices] = position
                qpos_changed = True
                print(f"Set '{self.free_joint_name}' position to: {position}")
            else:
                print("Warning: Position array must have 3 elements (x, y, z).")

        if orientation_quat is not None:
            if len(orientation_quat) == 4:
                # Optional: Normalize quaternion just in case
                # orientation_quat /= np.linalg.norm(orientation_quat)
                self.data.qpos[self.quat_indices] = orientation_quat
                qpos_changed = True
                print(f"Set '{self.free_joint_name}' orientation to: {orientation_quat}")
            else:
                print("Warning: Orientation quaternion array must have 4 elements (w, x, y, z).")

        # Update simulation state if qpos was modified
        if qpos_changed:
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)

    def view_simulation(self, duration: float = 10.0):
        """
        Launches the MuJoCo viewer and runs the simulation for a specified duration.

        Args:
            duration (float): How long to run the simulation viewer in seconds.
        """
        if self.model is None or self.data is None:
            print("Error: Model not loaded properly. Cannot launch viewer.")
            return

        print(f"Launching viewer for {duration} seconds...")
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            start_time = time.time()
            while viewer.is_running() and time.time() - start_time < duration:
                step_start = time.time()

                # --- Control logic could go here (e.g., setting data.ctrl) ---
                # For this kinematic example, we typically set the pose *before* calling view_simulation

                # Step the simulation
                mujoco.mj_step(self.model, self.data)

                # Sync the viewer
                viewer.sync()

                # Rudimentary real-time synchronization
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
        print("Simulation view finished.")

# --- Example Usage ---
if __name__ == "__main__":
    xml_file = '/home/mb230/projects/da-ai-evaluator/da_ai_evaluator/environments/mujoco_xmls/custom_scenes/scene_humanoid_smpl.xml' # IMPORTANT: Change this path!

    try:
        # 1. Instantiate the viewer object
        viewer_app = MuJoCoKinematicViewer(xml_path=xml_file, free_joint_name='Pelvis')

        # 2. Set the desired initial kinematic pose
        target_position = np.array([0, 0, 1]) # Example: Move forward and up
        # target_orientation = np.array([1.0, 0.0, 0.0, 0.0]) # Example: Default orientation (w,x,y,z)
        viewer_app.set_humanoid_pose(position=target_position, orientation_quat=[0,0,1,1]) #, orientation_quat=target_orientation)

        # 3. Run the viewer
        viewer_app.view_simulation(duration=10000) # View for 15 seconds

    except Exception as e:
        print(f"An error occurred during execution: {e}")