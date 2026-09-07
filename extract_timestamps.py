import re
import subprocess
import numpy as np
import cv2
import pytesseract
from PIL import Image

import char_ocr

# Matches a clean HH:MM (or H:MM) OCR read, e.g. "12:35"
TIMESTAMP_RE = re.compile(r'^\d{1,2}:\d{2}$')

def extract_timestamp_from_frame(image_path, crop_box):
    """
    Crops a specific region of interest (ROI) from an image and runs OCR 
    configured strictly for digital timestamps.
    crop_box format: (ymin, ymax, xmin, xmax)
    """
    img = cv2.imread(image_path)
    
    # 1. Crop using NumPy array slicing [ymin:ymax, xmin:xmax]
    ymin, ymax, xmin, xmax = crop_box
    cropped_roi = img[ymin:ymax, xmin:xmax]
    
    # Optional preprocessing for better OCR accuracy:
    # Convert to grayscale and scale up to help Tesseract read digital fonts cleanly
    gray = cv2.cvtColor(cropped_roi, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # 2. Run Tesseract OCR optimized for a single line of numbers/symbols
    # --psm 7 treats the image as a single text line
    # whitelist restricts characters to digits, colons, spaces, and hyphens
    custom_config = r'--psm 7 --user-patterns time.pattern -c tessedit_char_whitelist=0123456789:- '
    timestamp_text = pytesseract.image_to_string(scaled, config=custom_config)
    
    return timestamp_text.strip()

def extract_timestamp_at_second(video_path, timestamp, crop_box, keyframe_only=False):
    print(f"extract_timestamp_at_second: {timestamp} - {crop_box}" + (" [keyframe fallback]" if keyframe_only else ""))
    # Construct an ffmpeg command to seek to the timestamp,
    # grab 1 frame (-vframes 1), and output raw image bytes to stdout (-f image2pipe)
    cmd = ['ffmpeg']
    if keyframe_only:
        # Fallback path: skip decoding forward to the exact PTS and just grab the
        # nearest keyframe instead. Keyframes are fully self-contained I-frames, so
        # they can't suffer the block artifacts a partially-decoded frame can - at
        # the cost of the frame potentially being a couple seconds off target.
        cmd += ['-noaccurate_seek']
    cmd += ['-ss', str(timestamp)]
    if keyframe_only:
        cmd += ['-skip_frame', 'nokey']
    cmd += [
        '-i', video_path,
        '-vframes', '1',
        '-f', 'image2pipe',
        '-vcodec', 'mjpeg',
        'pipe:1'
    ]

    # Run ffmpeg, suppress stderr noise, and capture stdout bytes
    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    
    if process.returncode != 0 or not process.stdout:
        return "" # Handle read failures gracefully
    
    # Convert raw bytes to a numpy array, then decode into an OpenCV image
    image_array = np.frombuffer(process.stdout, dtype=np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    if img is None:
        return ""

    # Crop and recognize via template matching against char_templates/
    # (Tesseract proved unreliable on this camera's segmented-display font -
    # see char_ocr.py)
    ymin, ymax, xmin, xmax = crop_box
    cropped_roi = img[ymin:ymax, xmin:xmax]
    gray = cv2.cvtColor(cropped_roi, cv2.COLOR_BGR2GRAY)

    timestamp_text = char_ocr.recognize_timestamp(gray) or ""

    print(f"Return: {timestamp_text}")
    return timestamp_text

def get_timestamp(video_path, timestamp, crop_box, jitter_steps=(0.5, 1.0, 1.5, 2.0)):
    """
    Reads the on-screen timestamp at `timestamp` seconds. Falls back, in order, to:
    1. A keyframe-only grab (helps when the accurate-seek frame was decode-corrupted).
    2. Nearby timestamps, +/- increasing steps (helps when the source frame itself
       is bad, e.g. captured mid-digit-rollover on the camera's own overlay - a
       keyframe-only grab of the *same* frame won't fix that, only a different frame will).
    """
    text = extract_timestamp_at_second(video_path, timestamp, crop_box)
    if TIMESTAMP_RE.match(text):
        return text

    text = extract_timestamp_at_second(video_path, timestamp, crop_box, keyframe_only=True)
    if TIMESTAMP_RE.match(text):
        return text

    for offset in jitter_steps:
        for candidate in (timestamp + offset, timestamp - offset):
            if candidate < 0:
                continue
            text = extract_timestamp_at_second(video_path, candidate, crop_box)
            if TIMESTAMP_RE.match(text):
                return text

    return None  # couldn't get a clean read - unresolved, not "confirmed different"

def get_video_duration(video_path):
    """Queries the exact duration of the video in seconds using ffprobe."""
    cmd = [
        'ffprobe', 
        '-v', 'error', 
        '-show_entries', 'format=duration', 
        '-of', 'default=noprint_wrappers=1:nokey=1', 
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0

def find_times(video_path, crop_box):
    print(f"Next file: {video_path}")
    duration = get_video_duration(video_path)
    print(f"Duration: {duration}")
    if duration <= 0:
        return []

    low = 0.0
    high = duration - 0.5
    
    start_str = get_timestamp(video_path, low, crop_box)
    end_str = get_timestamp(video_path, high, crop_box)

    # An unresolved read (None) is "not proven different", not "confirmed different" -
    # given these clips are short and the on-screen clock is only minute-resolution,
    # the safe default is to assume no transition rather than invent a fake one.
    if start_str is None and end_str is None:
        return [(0.0, duration, None)]
    if start_str is None:
        return [(0.0, duration, end_str)]
    if end_str is None or start_str == end_str:
        return [(0.0, duration, start_str)]

    while high - low > 1.0:
        mid = (low + high) / 2.0
        mid_str = get_timestamp(video_path, mid, crop_box)

        if mid_str is None or mid_str == start_str:
            low = mid
        else:
            high = mid
            
    transition_sec = round(high)
    return [
        (0.0, transition_sec, start_str),
        (transition_sec, duration, end_str)
    ]

import os
import glob
import json

def process_video_directory(root_dir, crop_box):
    # os.walk bottom-up or standard traversal to find leaf directories
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Filter for MP4 files in the current folder (case-insensitive)
        mp4_files = [f for f in filenames if f.lower().endswith('.mp4')]
        
        # Skip folders that contain no MP4 files (not our leaf target)
        if not mp4_files:
            continue
            
        print(f"Processing folder: {dirpath} ({len(mp4_files)} clips)")
        folder_results = {}
        
        for filename in sorted(mp4_files):
            video_path = os.path.join(dirpath, filename)
            
            # Run your transition detection function
            segments = find_times(video_path, crop_box)
            folder_results[filename] = [
                {"startPos": start, "endPos": end, "timestamp": ts} 
                for start, end, ts in segments
            ]
            
        # Write results to a text file inside that specific leaf folder
        output_file_path = os.path.join(dirpath, 'clip_timestamps.json')
        with open(output_file_path, 'w', encoding='utf-8') as f:
            # Using JSON makes it exceptionally easy to read back later 
            # while remaining human-readable as a plain text file.
            json.dump(folder_results, f, indent=2)
            
        print(f"Saved results to {output_file_path}")

# Example invocation:
# crop_box = (ymin, ymax, xmin, xmax)
# process_video_directory("/path/to/elephants_root", crop_box)

# Example usage coordinates (adjust based on where your camera puts the timestamp)
# e.g., Top-left banner: y from 20 to 80, x from 30 to 300
if __name__ == "__main__":
    box = (1035, 1080, 1245, 1390)
    # detected_timeframes = find_times("./250216/Sequence1_00.20/Camera_B1/IMG_0066.MP4", box)
    detected_timeframes = process_video_directory("./250216", box)
    print(f"Detected Timestamp: {detected_timeframes}")
