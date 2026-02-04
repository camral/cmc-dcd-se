# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from collections import defaultdict

import torch
import time
from collections.abc import Sequence

import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prims_utils
from isaaclab.assets import Articulation, RigidObject, DeformableObject
# from isaaclab.sensors import TiledCamera0
import omni.kit.viewport.utility as vp_utils
from scipy.spatial.transform import Rotation as R

from .piper_obj_grasping_rl_env_cfg import PiperObjGraspingEnvCfg
from isaaclab.envs import DirectRLEnv

from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from omni.timeline import get_timeline_interface
from pxr import Usd, UsdGeom, Gf
import omni.usd
from omni.physx import get_physx_simulation_interface

from isaacsim.core.api.sensors import RigidContactView
from scipy.optimize import linear_sum_assignment

import sys
import os
import numpy as np


class PiperObjGraspingEnv(DirectRLEnv):
    cfg: PiperObjGraspingEnvCfg

    def __init__(self, cfg: PiperObjGraspingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Get joint indices for the arm and gripper
        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)

        # End-effector body index (link6 - wrist/end-effector)
        self.ee_body_idx = self.robot.find_bodies("link6")[0][0]

        # Get joint limits for clamping
        self.joint_lower_limits = self.robot.data.soft_joint_pos_limits[:, self.dof_idx, 0]
        self.joint_upper_limits = self.robot.data.soft_joint_pos_limits[:, self.dof_idx, 1]

        # Initialize joint targets and current positions
        self.robot_dof_targets = torch.zeros((self.num_envs, len(self.dof_idx)), device=self.device)
        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # CSV execution metrics
        self.step_count = 0
        self.current_reward = torch.zeros(self.num_envs, device=self.device)

        self.default_root_state = None # to be set on reset -- initial base position of the robot for each env

        self.contact_views = {}  # Initialized when get_contacts is called
        self.contact_buffer_size = 50  # Max number of contacts to track per link and object pair

        self.matching_matrix_mm = None
        self.mm_row_indx = None
        self.mm_col_indx = None


    def get_object_dimensions(self, prim_path):
        # TODO: technically this could be in some util file or in the RLEnv class
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)

        if not prim.IsValid():
            raise ValueError(f"Invalid prim path: {prim_path}")
        
        # Create an 'Imageable' wrapper for the prim to compute bounds
        imageable = UsdGeom.Imageable(prim)
        
        # Compute the world-aligned bounding box
        # TimeCode.Default() gets the static size; use a specific time if animated
        bbox = imageable.ComputeWorldBound(Usd.TimeCode.Default(), UsdGeom.Tokens.default_)
        range_val = bbox.GetRange()
        
        min_pt = range_val.GetMin()
        max_pt = range_val.GetMax()
        
        # Dimensions (Width=X, Depth=Y, Height=Z)
        width = max_pt[0] - min_pt[0]
        depth = max_pt[1] - min_pt[1]
        height = max_pt[2] - min_pt[2]
        
        return width, depth, height

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.obj = RigidObject(self.cfg.obj_cfg)
        # the prim path has *.* which needs to be replaced with env index 0 to get the actual prim path
        _obj_prim_path = self.cfg.obj_cfg.prim_path.replace(".*", "0")
        self.obj_width, self.obj_depth, self.obj_height = self.get_object_dimensions(_obj_prim_path)

        if self.obj_height > min(self.obj_width, self.obj_depth, self.obj_height):
            raise ValueError("The height of the object may be wrong... is it possible the height is not the smallest value?")
        
        # self.cube_3 = RigidObject(self.cfg.cube_3_cfg)
        # self.deformable_base = DeformableObject(self.cfg.deformable_base_cfg)
        # self.rigid_base = RigidObject(self.cfg.rigid_base)

        # Clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        # Add objects to scene
        self.scene.articulations["robot"] = self.robot

        robot_base_cfg = self.cfg.robot_base_cfg
        robot_base_cfg.spawn.func(
            robot_base_cfg.prim_path, robot_base_cfg.spawn, robot_base_cfg.init_state.pos, orientation=robot_base_cfg.init_state.rot
        )

        self.scene.rigid_objects["obj"] = self.obj
    
        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.torques = actions.clone()
    
        
    def _apply_action(self) -> None:
        self.robot.set_joint_effort_target(self.torques)

    def step(self, actions: torch.Tensor):

        observations, rewards, terminated, truncated, extras = super().step(actions)

        return observations, rewards, terminated, truncated, extras

    def _get_observations(self) -> dict:
        # Get current joint positions and velocities
        joint_pos = self.robot.data.joint_pos[:, self.dof_idx]
        joint_vel = self.robot.data.joint_vel[:, self.dof_idx]

        ee_pose_w = self.robot.data.body_pose_w[:, self.ee_body_idx]
        obj_pose_w = self.obj.data.root_pose_w

        # Scale joint positions to [-1, 1] range
        joint_pos_scaled = (
                2.0 * (joint_pos - self.joint_lower_limits) / (self.joint_upper_limits - self.joint_lower_limits) - 1.0
        )

        # TODO: change obs space? add nominal position of the springs? add velocities? add timestep
        # Concatenate observations: joint_pos (8) + ee_pos (3) + cube_pos (9) + joint_vel (8) + last_action (8) = 36
        observations = torch.cat([
            joint_pos_scaled, # 8
            ee_pose_w,  # 7 (position + orientation quaternion)
            obj_pose_w, # 7 (position + orientation quaternion)
        ], dim=-1)

        assert observations.shape[1] == self.cfg.observation_space, f"Observation space mismatch: expected {self.cfg.observation_space}, got {observations.shape[1]}"
        return {"policy": observations}

    def _get_rewards(self) -> torch.Tensor:
        reward = torch.zeros(self.num_envs, device=self.device)
        obj_pos = self.obj.data.root_pos_w
        
        # reward is the height of the object from the ground minus the half height of the object since the position is based on the center of the obj 
        # reward += torch.clamp(obj_pos[:, 2] - self.obj_height/2, min=0.0)  # Scale factor for visibility
        reward += torch.clamp(obj_pos[:, 2], min=0.0) # NOTE: until we have a reliable obj dimension estimator, we should not include height here
        self.current_reward = reward
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # No early termination conditions for now
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Timeout condition
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = list(range(self.cfg.scene.num_envs))
        super()._reset_idx(env_ids)

        # Reset to initial joint configuration
        default_joint_pos = self.robot.data.default_joint_pos[env_ids][:, self.dof_idx]

        # Reset robot_dof_targets to the default positions
        self.robot_dof_targets[env_ids] = default_joint_pos

        self.robot.write_joint_state_to_sim(
            default_joint_pos,
            torch.zeros_like(default_joint_pos),
            env_ids=env_ids,
            joint_ids=self.dof_idx
        )

        # Set the robot base position
        self.default_root_state = self.robot.data.default_root_state[env_ids]
        self.default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_state_to_sim(self.default_root_state, env_ids)

        # Reset obj
        obj_state = self.obj.data.default_root_state[env_ids].clone()
        # replace the height of the state from ground with the height of the obj to avoid it falling too much 
        if self.cfg.background != "warehouse":
            obj_state[0][2] = self.obj_height
        else:
            print("Warehouse background: using default obj height -- assuming there is still the bin box to collect from")
        obj_state[:, :3] += self.scene.env_origins[env_ids]
        self.obj.write_root_state_to_sim(obj_state, env_ids)

        # Reset step count
        self.step_count = 0

    def get_cube_positions(self) -> dict[str, torch.Tensor]:
        """Get current cube positions for logging."""
        return {
            "obj": self.obj.data.root_pos_w,
        }