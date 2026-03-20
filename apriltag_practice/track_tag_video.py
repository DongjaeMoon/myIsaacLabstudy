import csv
import math
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


VIDEO_PATH = "videos/tag_practice_1.mp4"

OUTPUT_MP4_PATH = "outputs/annotated_tag_practice_1.mp4"
OUTPUT_WEBM_PATH = "outputs/annotated_tag_practice_1.webm"
OUTPUT_CSV_PATH = "outputs/tag_practice_1_track.csv"
OUTPUT_TXT_PATH = "outputs/tag_practice_1_track.txt"
OUTPUT_PLOT_PATH = "outputs/tag_practice_1_plot.png"

TARGET_TAG_ID = 0
TAG_SIZE_M = 0.033   # 3.3 cm

USE_APPROX_CAMERA = True

# 회전 옵션:
# "none", "cw", "ccw", "180"
ROTATE_FRAME = "cw"

# calibration 안 했을 때 임시 intrinsic
K_USER = np.array([
    [1000.0,    0.0, 640.0],
    [   0.0, 1000.0, 360.0],
    [   0.0,    0.0,   1.0],
], dtype=np.float32)

D_USER = np.zeros((5, 1), dtype=np.float32)


def rotate_frame(frame, mode):
    if mode == "none":
        return frame
    if mode == "cw":
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if mode == "ccw":
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if mode == "180":
        return cv2.rotate(frame, cv2.ROTATE_180)
    raise ValueError(f"unknown ROTATE_FRAME mode: {mode}")


def build_rotated_camera_matrix(K, width, height, mode):
    if mode == "none":
        return K.copy(), width, height

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    if mode == "cw":
        # new width = old height, new height = old width
        new_w = height
        new_h = width
        K_new = np.array([
            [fy, 0.0, height - 1 - cy],
            [0.0, fx, cx],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        return K_new, new_w, new_h

    if mode == "ccw":
        new_w = height
        new_h = width
        K_new = np.array([
            [fy, 0.0, cy],
            [0.0, fx, width - 1 - cx],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        return K_new, new_w, new_h

    if mode == "180":
        new_w = width
        new_h = height
        K_new = np.array([
            [fx, 0.0, width - 1 - cx],
            [0.0, fy, height - 1 - cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        return K_new, new_w, new_h

    raise ValueError(f"unknown ROTATE_FRAME mode: {mode}")


def make_object_points(tag_size_m: float) -> np.ndarray:
    h = tag_size_m / 2.0
    return np.array(
        [
            [-h,  h, 0.0],   # top-left
            [ h,  h, 0.0],   # top-right
            [ h, -h, 0.0],   # bottom-right
            [-h, -h, 0.0],   # bottom-left
        ],
        dtype=np.float32,
    )


def rvec_to_euler_deg(rvec: np.ndarray):
    R, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0

    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


def fmt_or_nan(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "nan"
    return f"{x:.6f}"


def main():
    video_path = Path(VIDEO_PATH)
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    Path("outputs").mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-6:
        fps = 30.0
    dt = 1.0 / fps

    raw_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if USE_APPROX_CAMERA:
        f = 0.9 * max(raw_width, raw_height)
        K_base = np.array([
            [f,   0.0, raw_width / 2.0],
            [0.0, f,   raw_height / 2.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        D = np.zeros((5, 1), dtype=np.float32)
    else:
        K_base = K_USER.copy()
        D = D_USER.copy()

    K, width, height = build_rotated_camera_matrix(K_base, raw_width, raw_height, ROTATE_FRAME)

    mp4_fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer_mp4 = cv2.VideoWriter(OUTPUT_MP4_PATH, mp4_fourcc, fps, (width, height))

    # webm writer: 환경에 따라 실패할 수도 있음
    webm_fourcc = cv2.VideoWriter_fourcc(*"VP80")
    writer_webm = cv2.VideoWriter(OUTPUT_WEBM_PATH, webm_fourcc, fps, (width, height))
    webm_enabled = writer_webm.isOpened()

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

    obj_pts = make_object_points(TAG_SIZE_M)

    prev_tvec = None
    vel_ema = np.zeros(3, dtype=np.float32)
    alpha = 0.25

    detected_count = 0

    times = []
    detected_list = []
    tx_list, ty_list, tz_list = [], [], []
    vx_list, vy_list, vz_list = [], [], []
    roll_list, pitch_list, yaw_list = [], [], []

    txt_lines = []
    txt_lines.append("Coordinate convention (camera frame):")
    txt_lines.append("  +x = image right")
    txt_lines.append("  +y = image down")
    txt_lines.append("  +z = forward away from camera")
    txt_lines.append("")

    with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8") as f_csv:
        csv_writer = csv.writer(f_csv)
        csv_writer.writerow([
            "frame", "time_s", "detected",
            "tx_m", "ty_m", "tz_m",
            "vx_mps", "vy_mps", "vz_mps",
            "roll_deg", "pitch_deg", "yaw_deg"
        ])

        frame_idx = 0
        while True:
            ret, frame_raw = cap.read()
            if not ret:
                break

            frame = rotate_frame(frame_raw, ROTATE_FRAME)
            display = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            corners, ids, _ = detector.detectMarkers(gray)

            detected = False

            tvec = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
            vel = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
            roll = np.nan
            pitch = np.nan
            yaw = np.nan

            if ids is not None and len(ids) > 0:
                ids_flat = ids.flatten()
                matches = np.where(ids_flat == TARGET_TAG_ID)[0]

                if len(matches) > 0:
                    idx = int(matches[0])
                    img_pts = corners[idx].reshape(4, 2).astype(np.float32)

                    ok, rvec, tvec_raw = cv2.solvePnP(
                        obj_pts,
                        img_pts,
                        K,
                        D,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE,
                    )

                    if ok:
                        detected = True
                        detected_count += 1

                        tvec = tvec_raw.reshape(3).astype(np.float32)
                        roll, pitch, yaw = rvec_to_euler_deg(rvec)

                        if prev_tvec is not None:
                            vel_raw = (tvec - prev_tvec) / dt
                            vel_ema = alpha * vel_raw + (1.0 - alpha) * vel_ema
                            vel = vel_ema.copy()
                        else:
                            vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)

                        prev_tvec = tvec.copy()

                        cv2.aruco.drawDetectedMarkers(display, [corners[idx]], np.array([[TARGET_TAG_ID]]))
                        axis_len = TAG_SIZE_M * 0.5
                        cv2.drawFrameAxes(display, K, D, rvec, tvec_raw, axis_len, 2)

                        cx = int(np.mean(img_pts[:, 0]))
                        cy = int(np.mean(img_pts[:, 1]))
                        cv2.circle(display, (cx, cy), 4, (0, 0, 255), -1)
                else:
                    prev_tvec = None
            else:
                prev_tvec = None

            time_s = frame_idx * dt

            cv2.putText(display, f"frame={frame_idx}/{frame_count}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.putText(display, f"time={time_s:.3f}s  detected={detected}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if detected else (0, 0, 255), 2)

            cv2.putText(display, "camera frame: +x right, +y down, +z forward", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

            cv2.putText(display, f"pos_cam [m]=({fmt_or_nan(float(tvec[0]))}, {fmt_or_nan(float(tvec[1]))}, {fmt_or_nan(float(tvec[2]))})",
                        (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

            if np.any(np.isnan(vel)):
                vel_text = "vel_cam [m/s]=(nan, nan, nan)"
            else:
                vel_text = f"vel_cam [m/s]=({vel[0]:+.3f}, {vel[1]:+.3f}, {vel[2]:+.3f})"

            cv2.putText(display, vel_text, (20, 155),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

            rpy_text = f"rpy [deg]=({fmt_or_nan(float(roll))}, {fmt_or_nan(float(pitch))}, {fmt_or_nan(float(yaw))})"
            cv2.putText(display, rpy_text, (20, 185),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

            writer_mp4.write(display)
            if webm_enabled:
                writer_webm.write(display)

            csv_writer.writerow([
                frame_idx,
                f"{time_s:.6f}",
                int(detected),
                fmt_or_nan(float(tvec[0])),
                fmt_or_nan(float(tvec[1])),
                fmt_or_nan(float(tvec[2])),
                fmt_or_nan(float(vel[0])) if not np.isnan(vel[0]) else "nan",
                fmt_or_nan(float(vel[1])) if not np.isnan(vel[1]) else "nan",
                fmt_or_nan(float(vel[2])) if not np.isnan(vel[2]) else "nan",
                fmt_or_nan(float(roll)) if not np.isnan(roll) else "nan",
                fmt_or_nan(float(pitch)) if not np.isnan(pitch) else "nan",
                fmt_or_nan(float(yaw)) if not np.isnan(yaw) else "nan",
            ])

            txt_lines.append(
                f"frame={frame_idx:04d}  t={time_s:.3f}s  detected={int(detected)}  "
                f"pos=({fmt_or_nan(float(tvec[0]))}, {fmt_or_nan(float(tvec[1]))}, {fmt_or_nan(float(tvec[2]))})  "
                f"vel=({fmt_or_nan(float(vel[0])) if not np.isnan(vel[0]) else 'nan'}, "
                f"{fmt_or_nan(float(vel[1])) if not np.isnan(vel[1]) else 'nan'}, "
                f"{fmt_or_nan(float(vel[2])) if not np.isnan(vel[2]) else 'nan'})  "
                f"rpy=({fmt_or_nan(float(roll)) if not np.isnan(roll) else 'nan'}, "
                f"{fmt_or_nan(float(pitch)) if not np.isnan(pitch) else 'nan'}, "
                f"{fmt_or_nan(float(yaw)) if not np.isnan(yaw) else 'nan'})"
            )

            times.append(time_s)
            detected_list.append(int(detected))
            tx_list.append(float(tvec[0]))
            ty_list.append(float(tvec[1]))
            tz_list.append(float(tvec[2]))

            if np.any(np.isnan(vel)):
                vx_list.append(np.nan)
                vy_list.append(np.nan)
                vz_list.append(np.nan)
            else:
                vx_list.append(float(vel[0]))
                vy_list.append(float(vel[1]))
                vz_list.append(float(vel[2]))

            roll_list.append(float(roll))
            pitch_list.append(float(pitch))
            yaw_list.append(float(yaw))

            frame_idx += 1

    cap.release()
    writer_mp4.release()
    if webm_enabled:
        writer_webm.release()

    with open(OUTPUT_TXT_PATH, "w", encoding="utf-8") as f_txt:
        f_txt.write("\n".join(txt_lines))

    detect_ratio = detected_count / max(frame_idx, 1)

    print("=" * 70)
    print(f"processed frames : {frame_idx}")
    print(f"detected frames  : {detected_count}")
    print(f"detection ratio  : {detect_ratio:.3f}")
    print(f"saved mp4        : {OUTPUT_MP4_PATH}")
    if webm_enabled:
        print(f"saved webm       : {OUTPUT_WEBM_PATH}")
    else:
        print("webm writer      : not available in this OpenCV/FFmpeg build")
    print(f"saved csv        : {OUTPUT_CSV_PATH}")
    print(f"saved txt        : {OUTPUT_TXT_PATH}")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(times, tx_list, label="tx [m]")
    axes[0].plot(times, ty_list, label="ty [m]")
    axes[0].plot(times, tz_list, label="tz [m]")
    axes[0].set_ylabel("position [m]")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(times, vx_list, label="vx [m/s]")
    axes[1].plot(times, vy_list, label="vy [m/s]")
    axes[1].plot(times, vz_list, label="vz [m/s]")
    axes[1].set_ylabel("velocity [m/s]")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(times, detected_list, label="detected")
    axes[2].set_xlabel("time [s]")
    axes[2].set_ylabel("detected")
    axes[2].set_ylim([-0.1, 1.1])
    axes[2].grid(True)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH, dpi=150)

    print(f"saved plot       : {OUTPUT_PLOT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
