"""
camera_view.py — live viewer for Intel RealSense D405.
Press Q to quit, F to toggle fullscreen.
"""

import numpy as np
import cv2
import pyrealsense2 as rs

W, H = 1280, 720

pipeline = rs.pipeline()
config   = rs.config()
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, W, H, rs.format.z16,  30)

align    = rs.align(rs.stream.color)
colormap = rs.colorizer()
colormap.set_option(rs.option.color_scheme, 2)  # White-to-Black depth scheme

profile  = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale  = depth_sensor.get_depth_scale()

WIN = "D405 — Color | Depth  [F=fullscreen  Q=quit]"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, W * 2, H)

fullscreen = False
print("D405 streaming — F=fullscreen, Q=quit")

try:
    while True:
        frames      = pipeline.wait_for_frames()
        aligned     = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_img = np.asanyarray(color_frame.get_data())
        depth_col = np.asanyarray(colormap.colorize(depth_frame).get_data())

        # Depth distance overlay at centre point
        cx, cy = W // 2, H // 2
        dist_m = depth_frame.get_distance(cx, cy)
        label  = f"{dist_m*100:.1f} cm" if dist_m > 0 else "---"
        cv2.drawMarker(color_img, (cx, cy), (0, 255, 0),
                       cv2.MARKER_CROSS, 20, 2)
        cv2.putText(color_img, label, (cx + 14, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        combined = np.hstack([color_img, depth_col])
        cv2.imshow(WIN, combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            fullscreen = not fullscreen
            flag = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, flag)
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
