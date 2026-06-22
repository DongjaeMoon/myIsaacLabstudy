# G1 Real Catch Preflight Checklist

## Goal split
- **Must-succeed goal (tomorrow):** run `g1_catch_real.py` on the robot **without camera** and verify:
  - DDS connection works
  - policy loads
  - robot blends safely to ready posture
  - command loop stays stable
- **Stretch goal:** enable camera + AprilTag.

## Files that should exist before going to the lab
```text
myIsaacLabstudy/
├── sim2real/
│   ├── g1_catch_real.py
│   ├── apriltag_object_state_estimator.py
│   └── calib/
│       ├── head_camera_intrinsics.yaml   # optional for day-1
│       ├── head_camera_extrinsics.yaml   # optional for day-1
│       └── box_tag_config.yaml           # optional for day-1
├── unitree_sdk2_python/
└── xr_teleoperate/
```

## Before leaving for the lab
### 1) Freeze the code
- Do **not** edit UROP training files.
- Only use:
  - `sim2real/g1_catch_real.py`
  - `sim2real/apriltag_object_state_estimator.py`
  - calibration yaml files

### 2) Verify imports on the target machine
In Ubuntu terminal:
```bash
cd ~/mdj/myIsaacLabstudy
python3 -c "import torch, yaml, numpy; print('basic python ok')"
python3 -c "import unitree_sdk2py; print('unitree sdk ok')"
```

Optional camera-side packages:
```bash
python3 -c "import cv2; print('opencv ok')"
python3 -c "import zmq; print('zmq ok')"
```

### 3) Dry-run CLI only (no robot required)
```bash
cd ~/mdj/myIsaacLabstudy
python3 sim2real/g1_catch_real.py --help
```

### 4) Confirm policy path
```bash
ls -l /home/idim5080-2/mdj/myIsaacLabstudy/logs/rsl_rl/UROP_v12/2026-03-07_03-20-52/exported/policy.pt
```

### 5) Identify the network interface name now
```bash
ip addr
```
Write down the actual interface name you will use, e.g. `enp3s0`, `wlp2s0`, etc.

### 6) Prepare launch commands in a text file
No-camera:
```bash
python3 sim2real/g1_catch_real.py \
  --policy /home/idim5080-2/mdj/myIsaacLabstudy/logs/rsl_rl/UROP_v12/2026-03-07_03-20-52/exported/policy.pt \
  --net-iface YOUR_INTERFACE
```

With camera later:
```bash
python3 sim2real/g1_catch_real.py \
  --policy /home/idim5080-2/mdj/myIsaacLabstudy/logs/rsl_rl/UROP_v12/2026-03-07_03-20-52/exported/policy.pt \
  --net-iface YOUR_INTERFACE \
  --use-camera \
  --server-address CAMERA_PC_IP \
  --port 5555 \
  --intrinsics-yaml sim2real/calib/head_camera_intrinsics.yaml \
  --extrinsics-yaml sim2real/calib/head_camera_extrinsics.yaml \
  --tag-yaml sim2real/calib/box_tag_config.yaml
```

### 7) If you have time tonight: prepare calibration placeholders
Create:
- `sim2real/calib/head_camera_intrinsics.yaml`
- `sim2real/calib/head_camera_extrinsics.yaml`
- `sim2real/calib/box_tag_config.yaml`

Even rough placeholders are useful so file paths are already fixed.

## At the lab: exact order
1. Make sure the area around the robot is clear.
2. Connect to the correct network / interface.
3. Run the **no-camera** command first.
4. Verify the robot does not jerk and moves smoothly to ready posture.
5. Only if stable, try camera stream.
6. Only if camera stream is stable, try AprilTag estimation.
7. Only after that, try an actual slow handover / gentle toss.

## Things NOT to do tomorrow in the lab
- Do not rewrite UROP training code.
- Do not change action ordering.
- Do not start with full toss experiments first.
- Do not debug calibration before DDS / policy loop is stable.

## Minimum success criteria for tomorrow
- `g1_catch_real.py` runs on the robot
- `policy.pt` loads
- DDS connection works
- robot reaches/holds ready posture safely

