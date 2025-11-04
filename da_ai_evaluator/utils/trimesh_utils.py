import trimesh
import numpy as np

class TableScene:
    """
    Creates and manages a trimesh scene with a table and a box on top.

    This class provides direct access to the individual geometric components
    (tabletop, legs, box) and composite objects (the full table) for
    easier downstream analysis.
    
    Attributes:
        scene (trimesh.Scene): The main scene object containing all geometries.
        tabletop (trimesh.Trimesh): The tabletop mesh.
        legs (list[trimesh.Trimesh]): A list containing the four leg meshes.
        box (trimesh.Trimesh): The box mesh on top of the table.
        ground (trimesh.Trimesh): The ground plane mesh.
        table (trimesh.Trimesh): A single, composite mesh of the tabletop
                                 and all four legs combined.
        
        # --- Transforms used at creation ---
        tabletop_transform (np.ndarray): 4x4 pose of the tabletop.
        leg_transforms (list): List of 4x4 poses for the four legs.
        box_transform (np.ndarray): 4x4 pose of the box.
    """
    def __init__(self, 
                 table_height=1.0, 
                 tabletop_size=np.array([1.5, 0.8, 0.05]), 
                 leg_thickness=0.05, 
                 box_extents=np.array([0.3, 0.4, 0.2])):
        
        # 1. Create the main scene object
        self.scene = trimesh.Scene()
        
        # --- Store key dimensions ---
        self.table_height = table_height
        self.tabletop_size = np.array(tabletop_size)
        self.leg_thickness = leg_thickness
        self.box_extents = np.array(box_extents)

        # --- Internal calculations ---
        leg_height = self.table_height - self.tabletop_size[2]
        self.leg_extents = np.array([leg_thickness, leg_thickness, leg_height])
        
        # --- Placeholders for transforms ---
        self.tabletop_transform = None
        self.leg_transforms = []
        self.box_transform = None

        # 2. Build and add components
        self._create_tabletop()
        self._create_legs()
        self._create_box()
        self._create_ground()
        
        # 3. Create the composite table mesh for analysis
        table_parts = [self.tabletop] + self.legs
        self.table = trimesh.util.concatenate(table_parts)

    def _create_tabletop(self):
        """Creates the tabletop mesh and adds it to the scene."""
        tabletop_center_z = self.table_height - (self.tabletop_size[2] / 2.0)
        
        # Store the transform
        self.tabletop_transform = trimesh.transformations.translation_matrix(
            [0, 0, tabletop_center_z]
        )
        
        self.tabletop = trimesh.primitives.Box(
            extents=self.tabletop_size,
            transform=self.tabletop_transform
        )
        self.tabletop.visual.vertex_colors = [139, 69, 19]  # Brown
        
        self.scene.add_geometry(self.tabletop, geom_name="tabletop")

    def _create_legs(self):
        """Creates the four leg meshes and adds them to the scene."""
        self.legs = []
        self.leg_transforms = [] # Clear list
        
        leg_offset_x = (self.tabletop_size[0] / 2.0) - (self.leg_thickness / 2.0)
        leg_offset_y = (self.tabletop_size[1] / 2.0) - (self.leg_thickness / 2.0)
        leg_center_z = self.leg_extents[2] / 2.0
        
        leg_positions = [
            [leg_offset_x, leg_offset_y, leg_center_z],    # Front-Right
            [-leg_offset_x, leg_offset_y, leg_center_z],   # Front-Left
            [-leg_offset_x, -leg_offset_y, leg_center_z],  # Back-Left
            [leg_offset_x, -leg_offset_y, leg_center_z]    # Back-Right
        ]
        
        for i, pos in enumerate(leg_positions):
            # Store the transform
            transform = trimesh.transformations.translation_matrix(pos)
            self.leg_transforms.append(transform)
            
            leg = trimesh.primitives.Box(
                extents=self.leg_extents,
                transform=transform
            )
            leg.visual.vertex_colors = [139, 69, 19]  # Brown
            
            self.legs.append(leg)
            self.scene.add_geometry(leg, geom_name=f"leg_{i+1}")

    def _create_box(self):
        """Creates the box mesh on top of the table."""
        box_center_z = self.table_height + (self.box_extents[2] / 2.0)
        box_center_xy = [0.2, -0.1]
        
        # Store the transform
        self.box_transform = trimesh.transformations.translation_matrix(
            [box_center_xy[0], box_center_xy[1], box_center_z]
        )
        
        self.box = trimesh.primitives.Box(
            extents=self.box_extents,
            transform=self.box_transform
        )
        self.box.visual.vertex_colors = [222, 184, 135]  # Cardboard
        
        self.scene.add_geometry(self.box, geom_name="box_on_top")

    def _create_ground(self):
        """Creates an optional ground plane for context."""
        ground_transform = trimesh.transformations.translation_matrix([0, 0, -0.01])
        self.ground = trimesh.primitives.Box(
            extents=[5, 5, 0.02],
            transform=ground_transform
        )
        self.ground.visual.vertex_colors = [200, 200, 200]  # Light gray
        
        self.scene.add_geometry(self.ground, geom_name="ground")
    
    def show(self, **kwargs):
        """Displays the scene in a new window."""
        self.scene.show(**kwargs)

    # ---------------------------------------------------------------
    # Location & Pose Properties
    # ---------------------------------------------------------------
    
    # --- Box Properties ---
    @property
    def box_location(self) -> np.ndarray:
        """(PRESERVED) Returns the dynamic center of mass of the box."""
        return self.box.center_mass

    @property
    def box_bounds(self) -> np.ndarray:
        """(PRESERVED) Returns the dynamic AABB bounds [min, max] of the box."""
        return self.box.bounds
    
    @property
    def box_rotation(self) -> np.ndarray:
        """(NEW) Returns the 3x3 rotation matrix from the box's original transform."""
        # 
        return self.box_transform[:3, :3]
        
    @property
    def box_translation(self) -> np.ndarray:
        """(NEW) Returns the (x, y, z) translation from the box's original transform."""
        # 
        return self.box_transform[:3, 3]

    # --- Tabletop Properties ---
    @property
    def tabletop_location(self) -> np.ndarray:
        """(PRESERVED) Returns the dynamic center of mass of the tabletop."""
        return self.tabletop.center_mass
        
    @property
    def tabletop_rotation(self) -> np.ndarray:
        """(NEW) Returns the 3x3 rotation matrix from the tabletop's original transform."""
        return self.tabletop_transform[:3, :3]
        
    @property
    def tabletop_translation(self) -> np.ndarray:
        """(NEW) Returns the (x, y, z) translation from the tabletop's original transform."""
        return self.tabletop_transform[:3, 3]

    # --- Full Table (Composite) Properties ---
    @property
    def table_location(self) -> np.ndarray:
        """
        (PRESERVED) Returns the dynamic center of mass of the entire table 
        (tabletop + legs combined).
        """
        return self.table.center_mass

    # --- Leg Properties ---
    @property
    def leg_locations(self) -> list:
        """(PRESERVED) Returns a list of the dynamic center of mass for each leg."""
        return [leg.center_mass for leg in self.legs]

    @property
    def leg_rotations(self) -> list:
        """(NEW) Returns a list of 3x3 rotation matrices for each leg."""
        return [tf[:3, :3] for tf in self.leg_transforms]

    @property
    def leg_translations(self) -> list:
        """(NEW) Returns a list of (x, y, z) translations for each leg."""
        return [tf[:3, 3] for tf in self.leg_transforms]


# ---------------------------------------------------------------
# Updated Example Usage
# ---------------------------------------------------------------

if __name__ == '__main__':
    # 1. Create an instance of the class
    my_scene = TableScene()
    
    # 2. Access all the properties
    print("--- Accessing Location & Pose Properties ---")
    
    # Get the box's location (center of mass)
    print(f"Box Location (Center): {my_scene.box_location}")
    
    # Get the box's translation (from the 4x4 transform)
    print(f"Box Translation (from pose): {my_scene.box_translation}")
    
    # Get the box's rotation matrix
    print(f"Box Rotation Matrix:\n{my_scene.box_rotation}")

    # Get the location of the first leg
    print(f"\nFirst Leg Location (Center): {my_scene.leg_locations[0]}")
    
    # Get the rotation of the first leg
    print(f"First Leg Rotation:\n{my_scene.leg_rotations[0]}")

    # 3. Show the scene
    print("\nShowing scene...")
    my_scene.show()