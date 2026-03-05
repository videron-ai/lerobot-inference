#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Synchronous inference script for real robots.

Unlike the RTC-based eval_with_real_robot.py, this runs fully synchronously:
the main loop blocks on each call to select_action(). Policies that maintain an
internal action chunk queue re-run inference automatically when the queue empties,
so the caller only ever requests one action per step.

Usage:
    python eval_sync.py \\
        --policy.path=<hf-repo-or-local-dir> \\
        --policy.device=cuda \\
        --robot.type=so100_follower \\
        --robot.port=/dev/ttyUSB0 \\
        --robot.id=so100_follower \\
        --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \\
        --task="Pick up the red cube" \\
        --duration=60
"""

import logging
import select
import sys
import time
from dataclasses import dataclass, field

import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.processor.factory import (
    make_default_robot_action_processor,
    make_default_robot_observation_processor,
)
from lerobot.rl.process import ProcessSignalHandler
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    koch_follower,
    so_follower,
)
from lerobot.robots.utils import make_robot_from_config
from lerobot.utils.hub import HubMixin
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- TASK CONFIGURATION ---
TASK_MAP = {
    "1": "Put the red lego in the blue bowl",
    "2": "Put all the legos on the table in the blue bowl",
    "3": "Move the robot to the home position",
}
current_task = TASK_MAP["1"]


def check_for_input(current_val: str) -> str:
    """Non-blocking stdin check for task switching."""
    if select.select([sys.stdin], [], [], 0)[0]:
        line = sys.stdin.readline().strip()
        if line in TASK_MAP:
            new_task = TASK_MAP[line]
            print(f">>> SWITCHING TO: {new_task}")
            return new_task
        else:
            print(f"Unknown command '{line}'. Available: {list(TASK_MAP.keys())}")
    return current_val


@dataclass
class SyncConfig(HubMixin):
    """Configuration for synchronous inference with real robots."""

    policy: PreTrainedConfig | None = None
    robot: RobotConfig | None = None
    duration: float = 30.0
    fps: float = 30.0
    task: str = field(default="", metadata={"help": "Default task to execute"})

    # Rerun visualization
    visualize: bool = False
    rerun_session_name: str = "eval_sync"
    rerun_ip: str | None = None
    rerun_port: int | None = None
    compress_images: bool = False

    def __post_init__(self):
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        else:
            raise ValueError("Policy path is required (--policy.path=...)")

        if self.robot is None:
            raise ValueError("Robot configuration must be provided")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


@parser.wrap()
def main(cfg: SyncConfig):
    init_logging()
    logger.info("Starting synchronous inference")

    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event

    # --- Load policy ---
    logger.info(f"Loading policy from {cfg.policy.pretrained_path}")
    policy_class = get_policy_class(cfg.policy.type)
    config = PreTrainedConfig.from_pretrained(cfg.policy.pretrained_path)

    if config.use_peft:
        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(cfg.policy.pretrained_path)
        policy = policy_class.from_pretrained(
            pretrained_name_or_path=peft_config.base_model_name_or_path, config=config
        )
        policy = PeftModel.from_pretrained(policy, cfg.policy.pretrained_path, config=peft_config)
    else:
        policy = policy_class.from_pretrained(cfg.policy.pretrained_path, config=config)

    policy = policy.to(cfg.policy.device)
    policy.eval()
    policy.reset()
    logger.info(f"Policy loaded on {cfg.policy.device}")

    # --- Load pre/post processors ---
    logger.info(f"Loading processors from {cfg.policy.pretrained_path}")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=None,
        preprocessor_overrides={
            "device_processor": {"device": cfg.policy.device},
        },
    )
    logger.info("Processors loaded")

    # --- Connect robot ---
    logger.info(f"Connecting robot: {cfg.robot.type}")
    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    robot_observation_processor = make_default_robot_observation_processor()
    robot_action_processor = make_default_robot_action_processor()
    dataset_features = hw_to_dataset_features(robot.observation_features, "observation")

    if cfg.visualize:
        init_rerun(
            session_name=cfg.rerun_session_name,
            ip=cfg.rerun_ip,
            port=cfg.rerun_port,
        )
        logger.info("Rerun visualization initialized")

    global current_task
    if cfg.task:
        current_task = cfg.task

    action_interval = 1.0 / cfg.fps
    start_time = time.time()
    step = 0

    logger.info(f"Running for {cfg.duration}s at {cfg.fps} Hz — task: '{current_task}'")
    logger.info(f"Type a task key {list(TASK_MAP.keys())} + Enter to switch tasks")

    try:
        while not shutdown_event.is_set() and (time.time() - start_time) < cfg.duration:
            t0 = time.perf_counter()

            # Non-blocking task switch from keyboard
            current_task = check_for_input(current_task)

            # --- Observation ---
            obs = robot.get_observation()

            if cfg.visualize:
                import rerun as rr
                rr.set_time_sequence("step", step)
                log_rerun_data(observation=obs, compress_images=cfg.compress_images)

            obs_processed = robot_observation_processor(obs)
            obs_dict = build_dataset_frame(dataset_features, obs_processed, prefix="observation")

            for name in obs_dict:
                obs_dict[name] = torch.from_numpy(obs_dict[name])
                if "image" in name:
                    obs_dict[name] = obs_dict[name].type(torch.float32) / 255
                    obs_dict[name] = obs_dict[name].permute(2, 0, 1).contiguous()
                obs_dict[name] = obs_dict[name].unsqueeze(0).to(cfg.policy.device)

            obs_dict["task"] = [current_task]
            obs_dict["robot_type"] = robot.name if hasattr(robot, "name") else ""

            # --- Preprocess ---
            batch = preprocessor(obs_dict)

            # --- Synchronous inference ---
            # Policy refills its internal chunk queue when empty; returns one action per call.
            action = policy.select_action(batch)  # (1, action_dim)

            # --- Postprocess & send ---
            action_postprocessed = postprocessor(action).squeeze(0).cpu()  # (action_dim,)
            action_dict = {
                key: action_postprocessed[i].item()
                for i, key in enumerate(robot.action_features)
            }
            action_processed = robot_action_processor((action_dict, None))
            robot.send_action(action_processed)

            step += 1
            if step % 50 == 0:
                elapsed = time.time() - start_time
                logger.info(f"[MAIN] step={step}  elapsed={elapsed:.1f}s  task='{current_task}'")

            # Pace the loop to target fps
            dt = time.perf_counter() - t0
            time.sleep(max(0.0, action_interval - dt - 0.001))

    finally:
        logger.info("Shutting down")
        shutdown_event.set()
        robot.disconnect()
        logger.info(f"Robot disconnected — total steps: {step}")


if __name__ == "__main__":
    main()
