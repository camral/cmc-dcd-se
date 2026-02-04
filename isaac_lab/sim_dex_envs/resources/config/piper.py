# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# Ref: https://github.com/agilexrobotics/robot_lab/blob/master/source/robot_lab/robot_lab/assets/agilex.py

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
import os

# Get the path to the Piper USD file
_ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "resources")
_PIPER_USD_PATH = os.path.join(_ASSETS_DIR, "robots", "piper", "piper.usd")

PIPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_PIPER_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, fix_root_link=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.3, 0.3, 0.3),
            metallic=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": 1.0, 
            "joint3": -1.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
            "joint7": 0.0,  # gripper
            "joint8": 0.0,  # gripper
        },
        joint_vel={".*": 0.0},
    ),
    actuators={ 
        # Main arm joints (revolute)
        "arm_joints": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-6]"],
            stiffness=800.0,
            damping=20.0,
            effort_limit_sim=10.0,
        ),
        "hand": ImplicitActuatorCfg(
            joint_names_expr=["joint8", "joint7"],
            effort_limit=20.0,
            velocity_limit=0.01,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
