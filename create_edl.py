import os
import subprocess
import sys

import blank_calculator


def get_video_fps(video_path):
    """Reads the source clip's actual native frame rate via ffprobe. This has to
    match both the fps used to write the EDL's timecodes below AND the framerate
    selected in DaVinci's import dialog - a mismatch anywhere in that chain scales
    every duration in the timeline, silently wrecking the blank-duration math."""
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        num, den = result.stdout.strip().split('/')
        return float(num) / float(den)
    except (ValueError, ZeroDivisionError):
        return None


def blank_template_filename(total_seconds):
    """Maps a chosen blank duration (in seconds) to its template clip's filename,
    e.g. 10 -> '0m10s_Template.MP4', 70 -> '1m10s_Template.MP4'."""
    minutes, seconds = divmod(int(total_seconds), 60)
    return f"{minutes}m{seconds:02d}s_Template.MP4"


def seconds_to_timecode(seconds, fps=25):
    """Converts a float number of seconds into CMX 3600 timecode (HH:MM:SS:FF).
    fps may be fractional (e.g. NTSC 29.97) - frames-per-second is rounded to an
    integer for the HH:MM:SS:FF wraparound, but the precise fps is still used to
    compute the frame count itself."""
    fps_whole = int(round(fps))
    total_frames = int(round(seconds * fps))
    ff = total_frames % fps_whole
    total_seconds = total_frames // fps_whole
    ss = total_seconds % 60
    total_minutes = total_seconds // 60
    mm = total_minutes % 60
    hh = total_minutes // 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


def _add_event(edl_lines, event_index, record_frames, duration_sec, clip_name, fps):
    """Appends one EDL cut event and returns the updated record_frames counter."""
    src_in = seconds_to_timecode(0.0, fps)
    src_out = seconds_to_timecode(duration_sec, fps)

    rec_in = seconds_to_timecode(record_frames / fps, fps)
    record_frames += int(round(duration_sec * fps))
    rec_out = seconds_to_timecode(record_frames / fps, fps)

    edl_lines.append(f"{event_index:03d}  AX       V     C        {src_in} {src_out} {rec_in} {rec_out}")
    edl_lines.append(f"* FROM CLIP NAME: {clip_name}")
    edl_lines.append("")

    return record_frames


def generate_edl_from_clip_analysis(json_path, output_edl_path, fps=None):
    """
    Builds one CMX 3600 EDL for a single camera folder: one cut per clip file
    (its full duration - a file with two timestamp segments from a mid-clip
    minute rollover is still one continuous recording, not two cuts) with the
    blank_calculator-chosen gap between each pair of clips, and no trailing
    gap after the last clip.

    fps=None (the default) auto-detects the real frame rate from the first
    clip in the folder rather than guessing - see get_video_fps().
    """
    clips, decisions = blank_calculator.process_timestamp_json(json_path)

    if fps is None:
        first_video = os.path.join(os.path.dirname(json_path), clips[0]["filename"])
        fps = get_video_fps(first_video)
        if fps is None:
            print(f"WARNING: could not detect fps for {first_video}, defaulting to 25")
            fps = 25

    edl_lines = [
        "TITLE: WILDLIFE_AUTO_TIMELINE",
        "FCM: NON-DROP FRAME",
        ""
    ]

    record_frames = 1 * 3600 * int(round(fps))  # timeline starts at 01:00:00:00
    event_index = 1

    for i, clip in enumerate(clips):
        record_frames = _add_event(
            edl_lines, event_index, record_frames, clip["duration"], clip["filename"], fps
        )
        event_index += 1

        if i < len(decisions):
            gap_sec = decisions[i]["chosen_blank"]
            if gap_sec > 0:
                record_frames = _add_event(
                    edl_lines, event_index, record_frames, gap_sec,
                    blank_template_filename(gap_sec), fps
                )
                event_index += 1

    with open(output_edl_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(edl_lines))

    print(f"Generated EDL: {output_edl_path} ({len(clips)} clips, {len(decisions)} gaps, {fps} fps - "
          f"select this exact frame rate when importing into DaVinci)")


def generate_edls_for_directory(root_dir, fps=None):
    """Walks root_dir and builds one EDL per folder containing a
    clip_timestamps.json (i.e. one camera's clip sequence -> one timeline)."""
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        if 'clip_timestamps.json' not in filenames:
            continue

        json_path = os.path.join(dirpath, 'clip_timestamps.json')
        output_edl_path = os.path.join(dirpath, 'timeline.edl')
        generate_edl_from_clip_analysis(json_path, output_edl_path, fps=fps)


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "./250216"
    generate_edls_for_directory(root)
