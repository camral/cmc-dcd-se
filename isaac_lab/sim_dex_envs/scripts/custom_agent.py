# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import isaac_environments.tasks  # noqa: F401


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    counter = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute zero actions
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

            if counter % 3 == 0:
                # then add # 0, -0.055, -0.015 to all the envs and the secodn spring
                # add 0, 0.055, -0.015 to all the envs and the third spring
                actions[:, 0] += 0.00001
                actions[:, 1] += 0.00001
                actions[:, 2] += 0.00001
                actions[:, 3] += 0.00001
                actions[:, 4] += 0.00001
                
                actions[:, 5] += -0.055
                actions[:, 6] += -0.015
                actions[:, 9] += 0.055
                actions[:, 10] += -0.015
                actions[:, 11] += 0.00001
            elif counter % 3 == 1:
                # 0.01, +-0.012500000000000011, -0.015
                actions[:, 0] += 0.00001
                actions[:, 1] += 0.00001
                actions[:, 2] += 0.00001
                actions[:, 3] += 0.00001
                actions[:, 4] += 0.00001

                actions[:, 4] += 0.01
                actions[:, 5] += 0.012500000000000011
                actions[:, 6] += -0.015
                actions[:, 8] += 0.01
                actions[:, 9] += -0.012500000000000011
                actions[:, 10] += -0.015

            elif counter % 3 == 2:
                # 0.01, +-0.02
                actions[:, 0] += 0.00001
                actions[:, 1] += 0.00001
                actions[:, 2] += 0.00001
                actions[:, 3] += 0.00001
                actions[:, 4] += 0.00001
                
                actions[:, 5] += 0.01
                actions[:, 6] += 0.02
                actions[:, 9] += 0.01
                actions[:, 10] += -0.02
                 
            
            # apply actions
            # TODO uncomment for deformation actions = (actions, simulation_app)
            env.step(actions)
            counter += 1

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
