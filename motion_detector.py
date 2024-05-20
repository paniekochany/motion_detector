#!/usr/bin/env python
"""
Simple motion detection app.
"""

import cv2
from datetime import datetime
import pathlib
import time

PREVIEW = 0
MOTION_DETECTION = 1

mode = PREVIEW

def initialize_camera(source_index=0):
    source = cv2.VideoCapture(source_index)
    if not source.isOpened():
        raise Exception('Failed to open video device.')
    return source

def create_window(win_name):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

def initialize_snapshots_path():
    path = pathlib.Path.cwd() / 'snapshots'
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return  path

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21,21), 0)
    return blurred

def detect_motion(frame_initial_, frame_next_, contour_area_threshold=1000):
    diff = cv2.absdiff(frame_initial_, frame_next_)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) >= contour_area_threshold:
            return True
    return False

def log_movement(snapshots_path, active_frames):
    now = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    with open('logfile.txt', 'a') as f:
        f.write(f'{now}\n')
    ret_frame = active_frames[len(active_frames)//2]
    frame_path = snapshots_path / f'{now}.jpg'
    cv2.imwrite(frame_path.as_posix(), ret_frame)

def main():
    mode = PREVIEW
    source = initialize_camera()
    win_name = 'MOTION DETECTOR'
    create_window(win_name)
    snapshots_path = initialize_snapshots_path()

    ret, frame_initial = source.read()
    frame_initial_ = preprocess_frame(frame_initial)

    active_frames = []
    active=True

    while active:
        ret, frame_next = source.read()
        if not ret:
            break

        cv2.imshow(win_name, frame_next)

        if mode == MOTION_DETECTION:
            frame_next_ = preprocess_frame(frame_next)
            movement_detected = detect_motion(frame_initial_, frame_next_)

            if movement_detected:
                active_frames.append(frame_next)
            elif len(active_frames) > 30:
                log_movement(snapshots_path, active_frames)
                active_frames.clear()

            frame_initial_ = frame_next_

        key = cv2.waitKey(1)
        if key in [ord('Q'), ord('q'), 27]:
            active = False
        if key in [ord('D'),ord('d')]:
            mode = MOTION_DETECTION
            time.sleep(5)
        if key in [ord('P'), ord('p')]:
            mode = PREVIEW

    source.release()
    cv2.destroyWindow(win_name)

if __name__ == '__main__':
    main()