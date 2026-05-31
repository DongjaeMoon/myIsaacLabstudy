# AprilTag Real Preflight

Use this before enabling any real-robot policy test.

1. Print one large AprilTag first, preferably `tag36h11` id `0`.
2. Measure the actual printed black-square size in meters and update `tag_size_m`.
3. Mount the tag rigidly on the box front face first.
4. Start with a single tag, not multiple tags.
5. Only after the single-tag pipeline works should you consider multi-tag fusion.
6. Verify the camera frame first with `python test_apriltag_zmq.py --show ...`.
7. Verify tag detection before involving the robot policy.
8. Verify the sign of `rel_pos_b` by moving the box forward, back, left, right, up, and down.
9. Verify `rel_lin_vel_b` goes near zero when the box is stationary.
10. Verify the tag-hidden convention returns zeros with `tag_visible=0`.
11. Verify the observation pipeline with `--no-policy` first.
12. Only then test the gated policy config.
13. Only then test the no-gate policy config.
14. Only then try handover or a very weak toss.
15. Do not do a full toss until object observation is verified.

Notes:

- The current real AprilTag path uses IMU attitude plus a zero base position/velocity approximation.
- This is intended for the first standing-in-place real test, not for walking or aggressive torso motion.
- The template calibration YAMLs are placeholders and must be replaced with measured values for reliable catching.
