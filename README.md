# lerobot-inference

Synchronous inference scripts for running LeRobot policies on real robots.

## Overview

`eval_sync.py` is a synchronous inference loop for real robots using [LeRobot](https://github.com/huggingface/lerobot). Unlike RTC-based evaluation scripts, this runs fully synchronously — the main loop blocks on each `select_action()` call. Policies that maintain an internal action chunk queue re-run inference automatically when the queue empties, so the caller only ever requests one action per step.

## Features

- Synchronous inference loop with configurable FPS
- Runtime task switching via keyboard input (no restart required)
- Optional [Rerun](https://rerun.io) visualization
- PEFT/LoRA policy support
- Compatible with SO-100, Koch, and other LeRobot-supported robots

## Requirements

- [LeRobot](https://github.com/huggingface/lerobot) installed
- A supported robot (e.g. SO-100 Follower) connected via USB
- A pretrained LeRobot policy (local path or HuggingFace repo)

## Usage

```bash
python eval_sync.py \
    --policy.path=<hf-repo-or-local-dir> \
    --policy.device=cuda \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyUSB0 \
    --robot.id=so100_follower \
    --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --task="Pick up the red cube" \
    --duration=60
```

**Videron example:**

```bash
python eval_sync.py \
    --policy.path=staudi25/pi05_lego_maximus_plus_aug \
    --policy.device=cuda \
    --robot.type=so101_follower \
    --robot.port=/dev/follower \
    --robot.id=videron_follower \
    --robot.cameras="{desk_cam: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30, fourcc: 'MJPG'}, gripper: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: 'MJPG'}}" \
    --task="Put all the legos on the table in the blue bowl" \
    --robot.calibration_dir=/Experiments \
    --duration=1000 \
    --visualize=true \
    --compress_images=true
```

## Task Switching

While the script is running, you can switch tasks by typing a task key and pressing Enter:

| Key | Task |
|-----|------|
| `1` | Put the red lego in the blue bowl |
| `2` | Put all the legos on the table in the blue bowl |
| `3` | Move the robot to the home position |

## Optional: Rerun Visualization

Enable live observation visualization with [Rerun](https://rerun.io):

```bash
python eval_sync.py \
    --policy.path=<path> \
    --robot.type=so100_follower \
    --visualize=true \
    --rerun_session_name=my_session
```

## License

Apache 2.0 — see [LICENSE](https://www.apache.org/licenses/LICENSE-2.0).
