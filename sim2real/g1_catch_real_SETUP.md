# g1_catch_real.py setup

Put the files like this:

```text
myIsaacLabstudy/
├── sim2real/
│   ├── g1_catch_real.py
│   ├── apriltag_object_state_estimator.py
│   └── calib/
│       ├── head_camera_intrinsics.yaml
│       ├── head_camera_extrinsics.yaml
│       └── box_tag_config.yaml
```

First run **without camera**:

```bash
python3 sim2real/g1_catch_real.py \
  --policy /home/idim5080-2/mdj/myIsaacLabstudy/logs/rsl_rl/UROP_v12/2026-03-07_03-20-52/exported/policy.pt \
  --net-iface YOUR_INTERFACE
```

Then later run **with camera**:

```bash
python3 sim2real/g1_catch_real.py \
  --policy /home/idim5080-2/mdj/myIsaacLabstudy/logs/rsl_rl/UROP_v12/2026-03-07_03-20-52/exported/policy.pt \
  --net-iface YOUR_INTERFACE \
  --use-camera \
  --server-address 192.168.123.164 \
  --port 5555 \
  --intrinsics-yaml sim2real/calib/head_camera_intrinsics.yaml \
  --extrinsics-yaml sim2real/calib/head_camera_extrinsics.yaml \
  --tag-yaml sim2real/calib/box_tag_config.yaml
```

Notes:
- `prev_actions` is stored in **policy action order**, not motor order.
- Joint command targets are converted from policy action order to Unitree motor order inside the script.
- First version assumes **standing catch**: base linear velocity is set to zero.
- If camera is disabled, object observation is zero and `toss_signal=0`.
