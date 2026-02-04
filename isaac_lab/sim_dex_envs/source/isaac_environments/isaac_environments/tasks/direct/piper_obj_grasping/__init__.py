# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents
from .piper_obj_grasping_rl_env import PiperObjGraspingEnv
##
# Register Gym environments.
##

gym.register(
    id="Isaac-Piper-CSV-Execute-Obj-Grasping-Direct-RL-v0",
    entry_point=f"{__name__}.piper_obj_grasping_rl_env:PiperObjGraspingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.piper_obj_grasping_rl_env_cfg:PiperObjGraspingEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)