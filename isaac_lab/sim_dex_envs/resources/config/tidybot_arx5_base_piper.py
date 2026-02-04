# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for TidyBot ARX5 base with Piper arm robot."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
import os

# Get the path to the TidyBot ARX5 Base Piper USD file
_ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "resources")
_TIDYBOT_ARX5_BASE_PIPER_USD_PATH = os.path.join(
    _ASSETS_DIR, "robots", "tidybot_arx5_base_piper", "tidybot_arx5_base_piper.usd"
)

TIDYBOT_ARX5_BASE_PIPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_TIDYBOT_ARX5_BASE_PIPER_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            retain_accelerations=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            fix_root_link=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.4, 0.4, 0.4),
            metallic=0.2,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # Base joints
            "joint_x": 0.0,
            "joint_y": 0.0,
            "joint_th": 0.0,
            # Piper arm joints (initial pose for reaching)
            "joint1": 0.0,
            "joint2": 1.0,
            "joint3": -1.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
            # Gripper (open position)
            "joint7": 0.0,
            "joint8": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Prismatic joints for x, y movement (velocity control with zero stiffness)
        "base_xy_joints": ImplicitActuatorCfg(
            joint_names_expr=["joint_x", "joint_y"],
            stiffness=0.0,
            damping=1e5,
            effort_limit_sim=1000.0,
        ),
        # Revolute joint for theta rotation (velocity control with zero stiffness)
        "base_theta_joint": ImplicitActuatorCfg(
            joint_names_expr=["joint_th"],
            stiffness=0.0,
            damping=1e5,
            effort_limit_sim=1000.0,
        ),
        # Piper arm joints (position control)
        "arm_joints": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-6]"],
            stiffness=800.0,
            damping=20.0,
            effort_limit_sim=10.0,
        ),
        # Piper gripper (position control)
        "hand": ImplicitActuatorCfg(
            joint_names_expr=["joint7", "joint8"],
            effort_limit=20.0,
            velocity_limit=0.01,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
