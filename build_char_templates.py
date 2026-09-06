import argparse
import os
import uuid
import subprocess

import cv2
import numpy as np

from char_ocr import TEMPLATE_DIR, STANDARD_SIZE, VALID_CHARS, char_dirname, segment_characters

CROP_BOX = (1035, 1080, 1245, 1390)  # ymin, ymax, xmin, xmax - same box as extract_timestamps.py


def grab_frame(video_path, timestamp):
    cmd = [
        'ffmpeg',
        '-ss', str(timestamp),
        '-i', video_path,
        '-vframes', '1',
        '-f', 'image2pipe',
        '-vcodec', 'mjpeg',
        'pipe:1'
    ]
    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if process.returncode != 0 or not process.stdout:
        return None
    image_array = np.frombuffer(process.stdout, dtype=np.uint8)
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)


def save_templates(video_path, timestamp, label):
    label = label.strip()
    img = grab_frame(video_path, timestamp)
    if img is None:
        print(f"Could not grab a frame from {video_path} @ {timestamp}s - check the path/timestamp.")
        return

    ymin, ymax, xmin, xmax = CROP_BOX
    crop = img[ymin:ymax, xmin:xmax]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    segments, binary = segment_characters(gray)

    if len(segments) != len(label):
        os.makedirs(TEMPLATE_DIR, exist_ok=True)
        debug_path = os.path.join(TEMPLATE_DIR, "_debug_last_segmentation.png")
        cv2.imwrite(debug_path, binary)
        print(f"Found {len(segments)} character segment(s) but label {label!r} has {len(label)} "
              f"character(s) - mismatch, nothing saved.")
        print(f"Saved the attempted segmentation to {debug_path} so you can see what went wrong.")
        return

    saved = 0
    for (x0, x1), ch in zip(segments, label):
        if ch not in VALID_CHARS:
            print(f"Skipping unrecognized label character: {ch!r} (only 0-9 and ':' are supported)")
            continue
        char_img = binary[:, x0:x1]
        resized = cv2.resize(char_img, STANDARD_SIZE, interpolation=cv2.INTER_AREA)
        out_dir = os.path.join(TEMPLATE_DIR, char_dirname(ch))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{uuid.uuid4().hex[:8]}.png")
        cv2.imwrite(out_path, resized)
        saved += 1

    print(f"Saved {saved} character template(s) from {video_path} @ {timestamp}s (label: {label!r})")
    report_progress()


def report_progress():
    order = "0123456789:"
    missing = []
    counts = {}
    for ch in order:
        d = os.path.join(TEMPLATE_DIR, char_dirname(ch))
        n = len(os.listdir(d)) if os.path.isdir(d) else 0
        counts[ch] = n
        if n == 0:
            missing.append(ch)

    summary = "  ".join(f"{c}={counts[c]}" for c in order)
    print(f"Template counts: {summary}")
    if missing:
        print(f"Still missing templates for: {', '.join(missing)}")
    else:
        print("All characters (0-9 and colon) have at least one template.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grab a frame from a video at a given timestamp, segment the on-screen "
                     "HH:MM text into per-character images, and save them as labeled templates."
    )
    parser.add_argument("video_path", nargs="?", help="Path to the video file")
    parser.add_argument("timestamp", nargs="?", type=float, help="Second to grab the frame at")
    parser.add_argument("label", nargs="?", help='Exact on-screen text at that timestamp, e.g. "12:55"')
    parser.add_argument("--status", action="store_true", help="Just print current template coverage and exit")
    args = parser.parse_args()

    if args.status or not args.video_path:
        report_progress()
    else:
        save_templates(args.video_path, args.timestamp, args.label)
