import json
import os

def seconds_to_timecode(seconds, fps=25):
    """Converts a float number of seconds into CMX 3600 timecode (HH:MM:SS:FF)."""
    total_frames = int(round(seconds * fps))
    ff = total_frames % fps
    total_seconds = total_frames // fps
    ss = total_seconds % 60
    total_minutes = total_seconds // 60
    mm = total_minutes % 60
    hh = total_minutes // 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"

def generate_edl_from_json(json_path, output_edl_path, fps=25, default_gap_sec=10.0):
    if not os.path.exists(json_path):
        print(f"❌ JSON file not found: {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Sort filenames alphabetically, matching your workflow conventions
    sorted_filenames = sorted(data.keys())

    edl_lines = [
        "TITLE: WILDLIFE_AUTO_TIMELINE",
        "FCM: NON-DROP FRAME",
        ""
    ]

    # Start timeline record timecode at 01:00:00:00
    record_frames = 1 * 3600 * fps 
    event_index = 1

    for filename in sorted_filenames:
        segments = data[filename]

        for seg in segments:
            start_sec = seg["startPos"]
            end_sec = seg["endPos"]
            duration_sec = end_sec - start_sec

            src_in = seconds_to_timecode(start_sec, fps)
            src_out = seconds_to_timecode(end_sec, fps)
            
            rec_in = seconds_to_timecode(record_frames / fps, fps)
            record_frames += int(round(duration_sec * fps))
            rec_out = seconds_to_timecode(record_frames / fps, fps)

            # Format the CMX 3600 EDL event line
            edl_line = f"{event_index:03d}  AX       V     C        {src_in} {src_out} {rec_in} {rec_out}"
            edl_lines.append(edl_line)
            edl_lines.append(f"* FROM CLIP NAME: {filename}")
            edl_lines.append("")
            event_index += 1

        # Simple placeholder logic: insert a default blank spacer after each file/segment
        if default_gap_sec > 0:
            placeholder_dur = default_gap_sec
            src_in = "00:00:00:00"
            src_out = seconds_to_timecode(placeholder_dur, fps)
            
            rec_in = seconds_to_timecode(record_frames / fps, fps)
            record_frames += int(round(placeholder_dur * fps))
            rec_out = seconds_to_timecode(record_frames / fps, fps)

            edl_line = f"{event_index:03d}  AX       V     C        {src_in} {src_out} {rec_in} {rec_out}"
            edl_lines.append(edl_line)
            edl_lines.append(f"* FROM CLIP NAME: placeholder_{int(default_gap_sec)}s.mp4")
            edl_lines.append("")
            event_index += 1

    # Write out the final EDL text file
    with open(output_edl_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(edl_lines))

    print(f"✅ Successfully generated EDL: {output_edl_path}")

# Example usage:
generate_edl_from_json("./clip_timestamps.json", "./timeline.edl")