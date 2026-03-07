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

Example:
    python eval_sync.py \\
        --policy.path=staudi25/pi05_lego_maximus_plus_aug \\
        --policy.device=cuda \\
        --robot.type=so101_follower \\
        --robot.port=/dev/follower \\
        --robot.id=videron_follower \\
        --robot.cameras="{desk_cam: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30, fourcc: 'MJPG'}, gripper: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: 'MJPG'}}" \\
        --task="Put all the legos on the table in the blue bowl" \\
        --robot.calibration_dir=/Experiments \\
        --duration=1000 \\
        --visualize=true \\
        --compress_images=true
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

# Subtracted from the target sleep duration to compensate for time.sleep() wake-up overhead.
SLEEP_OVERHEAD_S = 0.001

# Default task map used for runtime task switching when none is provided via config.
DEFAULT_TASK_MAP = {
    "1": "Put the red lego in the blue bowl",
    "2": "Put all the legos on the table in the blue bowl",
    "3": "Move the robot to the home position",
}


def check_for_input(current_val: str, task_map: dict[str, str]) -> str:
    """Non-blocking stdin check for runtime task switching. Unix only (uses select.select).

    Uses ``select.select`` with a zero timeout to poll stdin without blocking the
    control loop. If a line is available, it is read and matched against ``task_map``.
    A matching key causes the returned task string to change; an unrecognised key
    emits a warning and leaves the current task unchanged.

    .. note::
        ``select.select`` on file descriptors is Unix-specific. This function will
        not work on Windows where stdin is not a socket-like object.

    Args:
        current_val (str): The task description that is currently active. Returned
            unchanged when no input is available or the input is not a valid key.
        task_map (dict[str, str]): Mapping from short keyboard keys (e.g. ``"1"``,
            ``"2"``) to full task description strings (e.g. ``"Pick up the red cube"``).

    Returns:
        str: The newly selected task description if a valid key was typed, or
        ``current_val`` if no input was available or the key was unrecognised.
    """
    if select.select([sys.stdin], [], [], 0)[0]:
        line = sys.stdin.readline().strip()
        if line in task_map:
            new_task = task_map[line]
            logger.info(f"SWITCHING TO: {new_task}")
            return new_task
        else:
            logger.warning(f"Unknown command '{line}'. Available: {list(task_map.keys())}")
    return current_val


@dataclass
class SyncConfig(HubMixin):
    """Configuration dataclass for synchronous real-robot inference.

    Parsed from CLI arguments by the LeRobot ``@parser.wrap()`` decorator and
    optionally loadable from / saveable to the HuggingFace Hub via the
    ``HubMixin`` base class.

    Attributes:
        policy (PreTrainedConfig | None): Policy configuration, populated
            automatically from ``--policy.path`` during ``__post_init__``.
            Contains all hyperparameters needed to reconstruct the model as
            well as runtime settings such as ``device`` and ``pretrained_path``.
            Must not be ``None`` after initialisation (enforced by validation).
        robot (RobotConfig | None): Robot hardware configuration, e.g.
            ``so101_follower``. Passed directly as a CLI argument group.
            Must not be ``None`` after initialisation (enforced by validation).
        duration (float): Total wall-clock time in seconds for which the
            control loop runs. Default: ``30.0``.
        fps (float): Target control frequency in Hz. The loop paces itself to
            this rate using ``time.sleep``. Default: ``30.0``.
        task (str): Natural-language description of the task the policy should
            perform (e.g. ``"Pick up the red cube"``). Injected into every
            observation batch as ``obs_dict["task"]``. If empty, the first
            value in ``task_map`` is used as the default. Default: ``""``.
        task_map (dict[str, str]): Optional mapping from single-character keys
            to task description strings, used for runtime task switching via
            stdin. Example: ``{"1": "Pick up cube", "2": "Place in bowl"}``.
            If empty, ``DEFAULT_TASK_MAP`` is used. Default: ``{}``.
        visualize (bool): Whether to stream observations to a running Rerun
            viewer. Requires ``rerun-sdk`` to be installed. Default: ``False``.
        rerun_session_name (str): Rerun recording/session identifier shown in
            the viewer. Default: ``"eval_sync"``.
        rerun_ip (str | None): IP address of the remote Rerun viewer. ``None``
            connects to a local viewer. Default: ``None``.
        rerun_port (int | None): Port of the remote Rerun viewer. ``None`` uses
            the Rerun default port. Default: ``None``.
        compress_images (bool): When ``True``, JPEG-compress camera images
            before sending to Rerun to reduce bandwidth. Default: ``False``.
    """

    policy: PreTrainedConfig | None = None
    robot: RobotConfig | None = None
    duration: float = 30.0
    fps: float = 30.0
    task: str = field(default="", metadata={"help": "Default task to execute"})
    task_map: dict[str, str] = field(
        default_factory=dict,
        metadata={"help": "Key-to-task-description map for runtime task switching via stdin"},
    )

    # Rerun visualization
    visualize: bool = False
    rerun_session_name: str = "eval_sync"
    rerun_ip: str | None = None
    rerun_port: int | None = None
    compress_images: bool = False

    def __post_init__(self) -> None:
        """Resolve and validate fields that require post-construction logic.

        Reads ``--policy.path`` from the CLI argument namespace via
        ``parser.get_path_arg``, loads the corresponding ``PreTrainedConfig``
        (applying any ``--policy.*`` CLI overrides), and stores it on
        ``self.policy``. Also validates that a robot config was provided.

        Raises:
            ValueError: If ``--policy.path`` was not supplied on the command
                line, or if ``self.robot`` is ``None`` after dataclass
                initialisation.
        """
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
        """Return the names of fields whose values are HuggingFace Hub paths.

        Used by the ``HubMixin`` / LeRobot parser infrastructure to identify
        which fields should be resolved as pretrained-model paths (Hub repo IDs
        or local directories) rather than plain strings.

        Returns:
            list[str]: A list containing the single element ``"policy"``,
            indicating that ``SyncConfig.policy`` is a path field.
        """
        return ["policy"]


@parser.wrap()
def main(cfg: SyncConfig) -> None:
    """Run the synchronous inference loop against a live robot.

    This is the script's sole entry point. It performs the following stages in
    order, then enters the control loop:

    1. **Logging initialisation** — calls ``init_logging()`` and logs the
       resolved configuration.
    2. **Signal handling** — registers SIGINT/SIGTERM handlers via
       ``ProcessSignalHandler`` so the loop exits cleanly on Ctrl-C.
    3. **Policy loading** — instantiates the policy from ``cfg.policy``.
       If ``cfg.policy.use_peft`` is ``True``, the base weights are loaded
       first and the PEFT/LoRA adapter is applied on top.
    4. **Processor loading** — creates a ``preprocessor`` that normalises
       inputs and moves them to the target device, and a ``postprocessor``
       that de-normalises the raw action tensor produced by the policy.
    5. **Robot connection** — builds a ``Robot`` instance from ``cfg.robot``
       and calls ``robot.connect()``.
    6. **Rerun initialisation** (optional) — if ``cfg.visualize`` is ``True``,
       starts a Rerun recording session.
    7. **Control loop** — runs until ``cfg.duration`` seconds have elapsed or a
       shutdown signal is received. Each iteration:

       a. Checks stdin for a task-switch key (non-blocking).
       b. Calls ``robot.get_observation()`` → applies the robot observation
          processor → converts to a ``dict[str, torch.Tensor]`` batch.
       c. Runs the preprocessor batch through ``policy.select_action()``.
       d. Postprocesses the action tensor and sends it to the robot via
          ``robot.send_action()``.
       e. Sleeps to maintain the target FPS.

    8. **Shutdown** — disconnects the robot in a ``finally`` block regardless
       of how the loop exits.

    Args:
        cfg (SyncConfig): Fully resolved configuration dataclass. All fields
            are validated during ``SyncConfig.__post_init__``; by the time
            ``main`` is called, ``cfg.policy`` and ``cfg.robot`` are guaranteed
            to be non-``None``.

    Returns:
        None

    Raises:
        Exception: Any unhandled exception from policy inference propagates
            upward. Robot I/O errors (``get_observation``, ``send_action``)
            are caught, logged, and cause the loop to break gracefully.

    Data flow inside the loop (per step)::

        robot.get_observation()
            -> dict[str, np.ndarray]          # raw sensor readings
        robot_observation_processor(obs)
            -> dict[str, np.ndarray]          # standardised numpy arrays
        build_dataset_frame(dataset_features, obs_processed)
            -> dict[str, np.ndarray]          # named observation arrays
        {torch.from_numpy, /255, permute, unsqueeze, .to(device)}
            -> dict[str, torch.Tensor]        # shape (1, ...) on policy device
        preprocessor(obs_dict)
            -> dict[str, torch.Tensor]        # normalised batch
        policy.select_action(batch)
            -> torch.Tensor  shape (1, action_dim)
        postprocessor(action).squeeze(0).cpu()
            -> torch.Tensor  shape (action_dim,)  dtype float32
        {key: tensor[i].item() for i, key in enumerate(robot.action_features)}
            -> dict[str, float]               # one float per action dimension
        robot_action_processor((action_dict, None))
            -> robot-specific action object
        robot.send_action(action_processed)
            -> None
    """
    init_logging()
    logger.info("Starting synchronous inference")
    logger.info(
        f"  policy={cfg.policy.pretrained_path}  device={cfg.policy.device}"
        f"  robot={cfg.robot.type}  duration={cfg.duration}s  fps={cfg.fps}  task='{cfg.task}'"
    )

    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event

    # --- Load policy ---
    logger.info(f"Loading policy from {cfg.policy.pretrained_path}")
    policy_class = get_policy_class(cfg.policy.type)

    if cfg.policy.use_peft:
        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(cfg.policy.pretrained_path)
        policy = policy_class.from_pretrained(
            pretrained_name_or_path=peft_config.base_model_name_or_path, config=cfg.policy
        )
        policy = PeftModel.from_pretrained(policy, cfg.policy.pretrained_path, config=peft_config)
    else:
        policy = policy_class.from_pretrained(cfg.policy.pretrained_path, config=cfg.policy)

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

    # --- Rerun visualization ---
    if cfg.visualize:
        import rerun as rr

        init_rerun(
            session_name=cfg.rerun_session_name,
            ip=cfg.rerun_ip,
            port=cfg.rerun_port,
        )
        logger.info("Rerun visualization initialized")

    # Resolve task map and initial task
    task_map = cfg.task_map if cfg.task_map else DEFAULT_TASK_MAP
    current_task = cfg.task if cfg.task else next(iter(task_map.values()), "")

    action_interval = 1.0 / cfg.fps
    start_time = time.perf_counter()
    step = 0

    logger.info(f"Running for {cfg.duration}s at {cfg.fps} Hz — task: '{current_task}'")
    logger.info(f"Type a task key {list(task_map.keys())} + Enter to switch tasks")

    try:
        while not shutdown_event.is_set() and (time.perf_counter() - start_time) < cfg.duration:
            t0 = time.perf_counter()

            # Non-blocking task switch from keyboard
            current_task = check_for_input(current_task, task_map)

            # --- Observation ---
            try:
                obs = robot.get_observation()
            except Exception as e:
                logger.error(f"Failed to get observation: {e}")
                break

            if cfg.visualize:
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

            try:
                action_processed = robot_action_processor((action_dict, None))
                robot.send_action(action_processed)
            except Exception as e:
                logger.error(f"Failed to send action: {e}")
                break

            step += 1
            if step % 50 == 0:
                elapsed = time.perf_counter() - start_time
                actual_fps = step / elapsed if elapsed > 0 else 0.0
                logger.info(
                    f"[MAIN] step={step}  elapsed={elapsed:.1f}s  fps={actual_fps:.1f}  task='{current_task}'"
                )

            # Pace the loop to target fps; subtract SLEEP_OVERHEAD_S to compensate for
            # the typical wake-up latency of time.sleep().
            dt = time.perf_counter() - t0
            time.sleep(max(0.0, action_interval - dt - SLEEP_OVERHEAD_S))

    finally:
        logger.info("Shutting down")
        shutdown_event.set()
        robot.disconnect()
        logger.info(f"Robot disconnected — total steps: {step}")


if __name__ == "__main__":
    main()
