import time
import cv2
import zmq
import yaml
import numpy as np

from apriltag_object_state_estimator import (
    AprilTagObjectStateEstimator,
    RobotBaseState,
    make_T,
)

def rpy_to_rotmat_xyz(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ===== paths =====
intr_path = "calib/head_camera_intrinsics.example.yaml"
extr_path = "calib/head_camera_extrinsics.example.yaml"
tag_path  = "calib/box_tag_config.example.yaml"

intr = load_yaml(intr_path)
extr = load_yaml(extr_path)
tagc = load_yaml(tag_path)

K = np.array(intr["camera_matrix"], dtype=np.float64)
D = np.array(intr.get("dist_coeffs", [0, 0, 0, 0, 0]), dtype=np.float64)

t_bc = np.array(extr["translation_m"], dtype=np.float64)
rpy_deg = np.array(extr["rpy_deg"], dtype=np.float64)
rpy = np.deg2rad(rpy_deg)
R_bc = rpy_to_rotmat_xyz(rpy[0], rpy[1], rpy[2])
T_b_c = make_T(R_bc, t_bc)

tag_center_in_box = np.array(tagc["tag_center_in_box_m"], dtype=np.float64)
tag_rpy = np.deg2rad(np.array(tagc["tag_rpy_in_box_deg"], dtype=np.float64))
R_tag_box = rpy_to_rotmat_xyz(tag_rpy[0], tag_rpy[1], tag_rpy[2])
T_tag_to_object = make_T(R_tag_box, tag_center_in_box)

estimator = AprilTagObjectStateEstimator(
    camera_matrix=K,
    dist_coeffs=D,
    tag_size_m=float(tagc["tag_size_m"]),
    T_b_c=T_b_c,
    T_tag_to_object=T_tag_to_object,
    tag_family=tagc.get("tag_family", "36h11"),
    target_tag_id=tagc.get("target_tag_id", 0),
)

ctx = zmq.Context()
sock = ctx.socket(zmq.SUB)
sock.connect("tcp://192.168.123.164:5555")
sock.setsockopt_string(zmq.SUBSCRIBE, "")

print("Waiting for image stream... press q to quit")

last_print = 0.0

while True:
    msg = sock.recv()
    np_img = np.frombuffer(msg, dtype=np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if frame is None:
        continue

    robot = RobotBaseState(
        pos_w=np.zeros(3, dtype=np.float64),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        lin_vel_w=np.zeros(3, dtype=np.float64),
        ang_vel_w=np.zeros(3, dtype=np.float64),
    )

    est = estimator.update(frame_bgr=frame, robot=robot, timestamp_s=time.time())

    text = "NO TAG"
    if est.valid:
        text = f"pos_b = {np.round(est.rel_pos_b, 3)}"
        now = time.time()
        if now - last_print > 0.2:
            print(
                "valid =", est.valid,
                "rel_pos_b =", np.round(est.rel_pos_b, 3),
                "rel_lin_vel_b =", np.round(est.rel_lin_vel_b, 3),
            )
            last_print = now

    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.imshow("AprilTag ZMQ Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

sock.close()
ctx.term()
cv2.destroyAllWindows()
