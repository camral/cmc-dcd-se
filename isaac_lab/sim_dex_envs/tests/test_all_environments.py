# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Test script to load and verify all registered environments.

This script tests each environment in a separate subprocess to avoid issues with
multiple environments in a single Isaac Lab app session.

Usage:
    # Test all environments (orchestrator mode)
    python tests/test_all_environments.py

    # Test single environment (subprocess mode, used internally)
    python tests/test_all_environments.py --single-env Isaac-Piper-Reach-Direct-IK-Delta-v0

    # Custom timeout and steps
    python tests/test_all_environments.py --timeout 600 --num_steps 20

Requirements:
    - Isaac Sim must be installed
    - GPU must be available
"""

import argparse
import subprocess
import sys
import os


def get_all_environment_ids_internal():
    """Get all environment IDs - requires Isaac Sim to be running.

    This is called in subprocess mode after Isaac Sim is launched.

    Returns:
        List of environment IDs registered in this project.
    """
    import gymnasium as gym
    import isaac_environments.tasks  # noqa: F401

    env_ids = []
    for task_spec in gym.registry.values():
        if not task_spec.id.startswith("Isaac-"):
            continue

        entry_point = str(task_spec.entry_point)

        # Direct RL environments: entry_point starts with our package
        if entry_point.startswith("isaac_environments.tasks"):
            env_ids.append(task_spec.id)

    return sorted(env_ids)


def list_environments():
    """List all environments - launches Isaac Sim in subprocess.

    Returns:
        List of environment IDs, or empty list on failure.
    """
    from isaaclab.app import AppLauncher
    import sys

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app

    try:
        env_ids = get_all_environment_ids_internal()
        # Print as JSON with unique markers for parent process to parse
        import json
        print("===ENV_LIST_START===", flush=True)
        print(json.dumps(env_ids), flush=True)
        print("===ENV_LIST_END===", flush=True)
        sys.stdout.flush()
        return env_ids
    except Exception as e:
        import traceback
        print(f"===ERROR===: {e}", flush=True)
        traceback.print_exc()
        return []
    finally:
        simulation_app.close()


def get_all_environment_ids(timeout: int = 120):
    """Get all environment IDs by running list command in subprocess.

    Args:
        timeout: Timeout in seconds for listing environments.

    Returns:
        List of environment IDs registered in this project.
    """
    import json

    script_path = os.path.abspath(__file__)

    try:
        result = subprocess.run(
            [sys.executable, script_path, "--list-envs"],
            timeout=timeout,
            capture_output=True,
            text=True,
        )
        # Look for JSON between markers in stdout
        # Isaac Sim outputs many messages, so we use markers to find our data
        stdout = result.stdout
        start_marker = "===ENV_LIST_START==="
        end_marker = "===ENV_LIST_END==="

        if start_marker in stdout and end_marker in stdout:
            start_idx = stdout.index(start_marker) + len(start_marker)
            end_idx = stdout.index(end_marker)
            json_str = stdout[start_idx:end_idx].strip()
            try:
                env_ids = json.loads(json_str)
                if isinstance(env_ids, list):
                    return env_ids
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse JSON: {e}")

        # Fallback: try parsing each line as JSON
        for line in stdout.strip().split("\n"):
            try:
                env_ids = json.loads(line)
                if isinstance(env_ids, list):
                    return env_ids
            except json.JSONDecodeError:
                continue

        # Only print error if no envs found
        print("Warning: Could not parse environment list from subprocess output")
        if result.stderr:
            # Truncate long stderr output
            print(f"Stderr (truncated): {result.stderr[:500]}")
        return []
    except subprocess.TimeoutExpired:
        print(f"Timeout listing environments after {timeout}s")
        return []


def run_single_env_test(env_id: str, timeout: int = 300, num_steps: int = 10, device: str = "cuda:0"):
    """Run test for a single environment in a subprocess with timeout.

    Args:
        env_id: The gymnasium environment ID to test.
        timeout: Maximum time in seconds to wait for the test.
        num_steps: Number of simulation steps to run.
        device: CUDA device to use.

    Returns:
        Tuple of (success: bool, stdout: str, stderr: str)
    """
    script_path = os.path.abspath(__file__)

    try:
        result = subprocess.run(
            [
                sys.executable,
                script_path,
                "--single-env", env_id,
                "--num_steps", str(num_steps),
                "--device", device,
            ],
            timeout=timeout,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", f"TIMEOUT: Test exceeded {timeout} seconds"


def test_single_environment(env_id: str, num_steps: int = 10, device: str = "cuda:0"):
    """Test a single environment - launches its own Isaac Sim instance.

    This function is called in subprocess mode.

    Args:
        env_id: The gymnasium environment ID to test.
        num_steps: Number of simulation steps to run.
        device: CUDA device to use.

    Returns:
        True if test passed, False otherwise.
    """
    # Launch Isaac Sim first (must be done before other imports)
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app

    try:
        # Import after Isaac Sim is launched
        import gymnasium as gym
        import torch
        import isaac_environments.tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg

        print(f"Testing environment: {env_id}")

        # Parse environment configuration
        env_cfg = parse_env_cfg(env_id, device=device, num_envs=1, use_fabric=False)

        # Create environment
        env = gym.make(env_id, cfg=env_cfg)

        try:
            # Print environment info
            print(f"  Observation space: {env.observation_space}")
            print(f"  Action space: {env.action_space}")

            # Reset environment
            env.reset()

            # Run simulation steps with random actions
            for _ in range(num_steps):
                with torch.inference_mode():
                    # Sample random actions in [-1, 1]
                    actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
                    # Step the environment
                    env.step(actions)

            print(f"  Successfully completed {num_steps} steps")
            return True

        finally:
            # Always close the environment
            env.close()

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Always close the simulation app
        simulation_app.close()


def main(timeout: int = 300, num_steps: int = 10, device: str = "cuda:0"):
    """Main function to orchestrate testing of all environments.

    Args:
        timeout: Maximum time in seconds per environment test.
        num_steps: Number of simulation steps per environment.
        device: CUDA device to use.
    """
    from prettytable import PrettyTable

    print("=" * 80)
    print("Testing All Isaac Lab Environments (Subprocess Mode)")
    print("=" * 80)

    # Get all environment IDs (no Isaac Sim needed for this)
    env_ids = get_all_environment_ids()
    print(f"\nFound {len(env_ids)} environments to test")
    print(f"Timeout per environment: {timeout}s")
    print(f"Steps per environment: {num_steps}\n")

    # Track results
    results = {"passed": [], "failed": [], "timeout": []}

    # Test each environment in its own subprocess
    for i, env_id in enumerate(env_ids):
        print(f"\n[{i + 1}/{len(env_ids)}] Testing: {env_id}")
        print("-" * 60)

        success, stdout, stderr = run_single_env_test(
            env_id,
            timeout=timeout,
            num_steps=num_steps,
            device=device,
        )

        # Print subprocess output
        if stdout:
            for line in stdout.strip().split("\n"):
                print(f"  {line}")

        if success:
            results["passed"].append(env_id)
            print("  PASSED")
        elif "TIMEOUT" in stderr:
            results["timeout"].append(env_id)
            print(f"  TIMEOUT (>{timeout}s)")
        else:
            # Extract last line of stderr for brief error
            error_lines = stderr.strip().split("\n") if stderr else ["Unknown error"]
            brief_error = error_lines[-1] if error_lines else "Unknown error"
            results["failed"].append((env_id, brief_error))
            print(f"  FAILED: {brief_error}")

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    # Create results table
    table = PrettyTable(["Status", "Environment", "Details"])
    table.align["Environment"] = "l"
    table.align["Details"] = "l"

    for env_id in results["passed"]:
        table.add_row(["PASS", env_id, ""])

    for env_id in results["timeout"]:
        table.add_row(["TIMEOUT", env_id, f">{timeout}s"])

    for env_id, error in results["failed"]:
        # Truncate error message for display
        short_error = error[:50] + "..." if len(error) > 50 else error
        table.add_row(["FAIL", env_id, short_error])

    print(table)

    # Print final statistics
    total = len(env_ids)
    passed = len(results["passed"])
    timeout_count = len(results["timeout"])
    failed = len(results["failed"])

    print(f"\nResults: {passed}/{total} passed, {timeout_count}/{total} timeout, {failed}/{total} failed")

    if timeout_count > 0:
        print("\nTimeout environments:")
        for env_id in results["timeout"]:
            print(f"  - {env_id}")

    if failed > 0:
        print("\nFailed environments:")
        for env_id, error in results["failed"]:
            print(f"  - {env_id}")
            print(f"    Error: {error}")

    return passed == total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test all Isaac Lab environments.")
    parser.add_argument(
        "--single-env",
        type=str,
        default=None,
        help="Test a single environment (subprocess mode). Used internally.",
    )
    parser.add_argument(
        "--list-envs",
        action="store_true",
        help="List all environments as JSON (subprocess mode). Used internally.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds for each environment test (default: 300).",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=10,
        help="Number of simulation steps per environment (default: 10).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="CUDA device to use (default: cuda:0).",
    )

    args = parser.parse_args()

    if args.list_envs:
        # Subprocess mode: list environments
        list_environments()
        sys.exit(0)
    elif args.single_env:
        # Subprocess mode: test single environment
        success = test_single_environment(args.single_env, args.num_steps, args.device)
        sys.exit(0 if success else 1)
    else:
        # Main mode: orchestrate all tests
        success = main(args.timeout, args.num_steps, args.device)
        sys.exit(0 if success else 1)
