# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import sys
import os 
from pathlib import Path
sys.path.insert(0, '.')

from isaaclab.sim.spawners.from_files.from_files import spawn_ground_plane
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, DeformableObjectCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.actuators import ImplicitActuatorCfg

_ASSETS_DIR = os.path.join(Path(__file__).resolve().parents[6], "resources")

sys.path.append(_ASSETS_DIR)
from config.piper_torque import PIPER_TORQUE_CFG as PIPER_STD_CFG

PLACEHOLDER = None  # Placeholder value to be replaced in __post_init__

@configclass
class PiperObjGraspingEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    frequency = 240
    episode_length_s = 6.0 

    # Choose the object to pick up: "coin", "cheerio", "syringe", "hex_nut", "pipette", "needle", "surgical_curved_tip", "surgical_scissors",
    # "surgical_long_tip", "surgical_knife_2", "surgical_knife", "small_surgical_tool"
    target_object: str = "surgical_knife" 
    
    # spaces definition
    action_space = 8 # torque for all 8 joints
    observation_space = 22  
    state_space = 0
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / frequency, render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=PIPER_STD_CFG.spawn,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                "joint1": 0.0,
                "joint2": 0.5,
                "joint3": -0.5,
                "joint4": 0.0,
                "joint5": 0.0,
                "joint6": 0.0,
                "joint7": 0.035,  # gripper open (max limit)
                "joint8": -0.035,  # gripper open (min limit for inverted joint)
            },
            joint_vel={".*": 0.0},
        ),
        actuators=PIPER_STD_CFG.actuators,
    )
    disable_robot_gravity = robot_cfg.spawn.rigid_props.disable_gravity

    robot_base_cfg: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/scene/robot_base",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[-0.103, 0.0, -0.9], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(usd_path=os.path.join(_ASSETS_DIR, f"scenes/tidybot_arx5_base_piper_pretty_fixed_lift.usd")),
    )
    
    # object to be picked up
    obj_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/TargetObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=PLACEHOLDER,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            scale=PLACEHOLDER,
            # mass_props=sim_utils.MassPropertiesCfg(mass=0.01), # NOTE: the mass in the USD should be correct!
                collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=PLACEHOLDER,
                rest_offset=0.0 
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=32, solver_velocity_iteration_count=0
            )
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=PLACEHOLDER,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True)
    background: str = "none" # one of: "none", "hospital", "warehouse", "automatic" (chosen based on the target obj)

    dof_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "joint8"]  # Main arm joints + gripper

    # action control
    joint_delta_scale = 1.0  # Scale for joint delta actions
    max_joint_delta = 0.1  # Maximum delta per step in radians
    action_type: str = "delta"  # "delta" or "absolute"

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

        # 2. Handle Object Selection Logic
        # Defaults
        scale = (1.0, 1.0, 1.0) 
        init_pos = (0.5, 0.0, 0.05)

        # for contact_offsetm, 0.02 (2cm) is good for large-ish objects, 0.002 (2mm) for smaller ones
        usd_path = os.path.join(_ASSETS_DIR, f"objects/{self.target_object}/{self.target_object}.usd")

        if self.target_object == "coin":
            contact_offset = 0.02    
        elif self.target_object in ["cheerio", "syringe", "pipette", "needle", "surgical_curved_tip", "surgical_scissors",
                                    "surgical_long_tip", "small_surgical_tool"]:
            contact_offset = 0.002  # smaller contact offset for cheerio otherwsie the cheerio is pushed away 
        elif self.target_object in ["hex_nut", "surgical_knife", "surgical_knife_2"]:
            contact_offset = 0.0002
            if self.target_object == "hex_nut":
                init_pos = (0.4, 0.0, 0.08) # higher initial pos for hex_nut as there is case beneath it
        else:
            raise ValueError(f"Unknown target_object: {self.target_object}")

        # Apply the object changes
        self.obj_cfg.spawn.usd_path = usd_path
        self.obj_cfg.spawn.scale = scale
        self.obj_cfg.spawn.collision_props.contact_offset = contact_offset
        self.obj_cfg.init_state.pos = init_pos
        
        print(f">> Loaded Target Object: {self.target_object} at {init_pos}")

        if self.background == "automatic":
            if self.target_object in ["coin", "cheerio", "hex_nut"]:
                self.background = "warehouse"
            else:
                self.background = "hospital"
            print(f">> Automatic Background Selection: {self.background}")

        if self.background == "none":
            from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
            spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        elif self.background == "hospital":
            self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="usd",
            usd_path=f"{_ASSETS_DIR}/scenes/hospital.usd",
        )
        elif self.background == "warehouse":
            self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="usd",
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
        )
        else:
            raise ValueError(f"Unknown background type: {self.background}")
