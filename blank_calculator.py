import json
import sys
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

# Available blank templates:
#
# 0:10
# 0:40
# 1:10
# 1:40
# ...
# 9:10
# 9:40
#
ALLOWED_BLANKS = list(range(10, 581, 30))


# ============================================================
# TIME UTILITIES
# ============================================================

def timestamp_to_minutes(timestamp):
    """
    Convert a 12-hour HH:MM timestamp into minutes.

    IMPORTANT:
    The timestamps do NOT contain AM/PM.

    The sequence itself determines chronology.

    Example:

        12:56
        12:57
        01:00
        01:01

    means:

        12:56
        12:57
        13:00
        13:01

    NOT:

        12:56
        12:57
        next day 01:00
    """

    hours, minutes = map(int, timestamp.split(":"))

    if hours < 1 or hours > 12:
        raise ValueError(
            f"Invalid 12-hour timestamp: {timestamp}"
        )

    if minutes < 0 or minutes > 59:
        raise ValueError(
            f"Invalid timestamp: {timestamp}"
        )

    return hours * 60 + minutes


def unwrap_timestamps(timestamps):
    """
    Convert 12-hour timestamps into a continuous timeline.

    The clock has a 12-hour cycle.

    Example:

        12:57
        01:00

    becomes:

        777
        780

    Example:

        01:59
        02:00

    becomes:

        119
        120

    The function only advances the clock when necessary
    to keep the sequence chronological.
    """

    result = []

    previous = None
    offset = 0

    for timestamp in timestamps:

        current = timestamp_to_minutes(timestamp)

        adjusted = current + offset

        if previous is not None:

            while adjusted < previous:

                offset += 12 * 60

                adjusted = current + offset

        result.append(adjusted)

        previous = adjusted

    return result


def seconds_to_duration(seconds):
    """
    Convert seconds into M:SS.s format.
    """

    if seconds is None:
        return "None"

    sign = "-" if seconds < 0 else ""

    seconds = abs(seconds)

    minutes = int(seconds // 60)

    remaining = seconds - minutes * 60

    if abs(remaining - round(remaining)) < 0.001:

        secs = int(round(remaining))

        return f"{sign}{minutes}:{secs:02d}"

    return f"{sign}{minutes}:{remaining:04.1f}"


# ============================================================
# CLIP ANALYSIS
# ============================================================

def analyze_clip(segments, absolute_timestamps):
    """
    Determine the possible real-world timing of a clip.

    CASE A:
        Timestamp changes during the clip.

        The clip position can be calculated exactly.

    CASE B:
        Timestamp remains constant.

        The clip can occur anywhere within that minute.

    IMPORTANT:

    We do NOT use neighboring clips here.

    Neighboring clips are handled later when evaluating
    transitions.
    """

    duration = segments[-1]["endPos"]

    # --------------------------------------------------------
    # CASE A:
    # Timestamp changes during clip.
    # --------------------------------------------------------

    for i in range(1, len(segments)):

        if (
            absolute_timestamps[i]
            != absolute_timestamps[i - 1]
        ):

            transition_position = (
                segments[i]["startPos"]
            )

            transition_time = (
                absolute_timestamps[i] * 60
            )

            # Assume the timestamp changed exactly at
            # the minute boundary.

            estimated_start = (
                transition_time
                - transition_position
            )

            estimated_end = (
                estimated_start
                + duration
            )

            return {
                "start_min": estimated_start,
                "start_max": estimated_start,
                "estimated_start": estimated_start,

                "end_min": estimated_end,
                "end_max": estimated_end,
                "estimated_end": estimated_end,

                "confidence": "high"
            }

    # --------------------------------------------------------
    # CASE B:
    # Timestamp does not change.
    # --------------------------------------------------------

    minute_start = (
        absolute_timestamps[0] * 60
    )

    # The entire clip has to fit inside the minute.

    start_min = minute_start

    start_max = (
        minute_start
        + 60
        - duration
    )

    estimated_start = (
        start_min
        + start_max
    ) / 2

    end_min = (
        start_min
        + duration
    )

    end_max = (
        start_max
        + duration
    )

    estimated_end = (
        end_min
        + end_max
    ) / 2

    return {
        "start_min": start_min,
        "start_max": start_max,
        "estimated_start": estimated_start,

        "end_min": end_min,
        "end_max": end_max,
        "estimated_end": estimated_end,

        "confidence": "low"
    }


# ============================================================
# TRANSITION ANALYSIS
# ============================================================

def calculate_blank(clip_a, clip_b):
    """
    Calculate the uncertainty range for the blank between
    two clips.

    We calculate:

        earliest possible blank
        latest possible blank
        midpoint / most likely blank

    WITHOUT forcing the two clips to touch neighboring clips.
    """

    min_blank = max(
        0,
        clip_b["start_min"]
        - clip_a["end_max"]
    )

    max_blank = (
        clip_b["start_max"]
        - clip_a["end_min"]
    )

    estimated_blank = (
        clip_b["estimated_start"]
        - clip_a["estimated_end"]
    )

    return (
        min_blank,
        max_blank,
        estimated_blank
    )


# ============================================================
# CONSTRAINT PROPAGATION
# ============================================================

def propagate_constraints(clips):
    """
    Tighten every clip's uncertainty range using the rest of
    the sequence, via the one hard physical rule we know
    for sure:

        clips cannot overlap in real time.

        i.e. clip[i].start must be >= clip[i-1].end

    High-confidence (CASE A) clips already have an exact
    start/end, so they act as fixed anchors. Low-confidence
    (CASE B) clips only know "somewhere in this minute" --
    but that window can be narrowed by anchors elsewhere in
    the sequence, propagating through any run of consecutive
    CASE B clips in between.

    This is a standard two-pass interval tightening:

        FORWARD PASS (left -> right):
            raise start_min[i] using end_min[i-1]
            (clip i can't have started before the earliest
            possible end of clip i-1)

        BACKWARD PASS (right -> left):
            lower end_max[i] using start_max[i+1]
            (clip i can't have ended after the latest
            possible start of clip i+1)

    One forward sweep + one backward sweep fully propagates
    along this kind of linear chain -- no repeated passes
    needed.

    Mutates `clips` in place.
    """

    n = len(clips)

    # --------------------------------------------------------
    # FORWARD PASS: tighten start_min / end_min
    # --------------------------------------------------------

    for i in range(1, n):

        prev = clips[i - 1]
        cur = clips[i]

        new_start_min = max(
            cur["start_min"],
            prev["end_min"]
        )

        if new_start_min > cur["start_max"]:

            # The data is contradictory (e.g. a misread
            # timestamp). Flag it and clamp rather than
            # silently producing an inverted range.

            cur["contradiction"] = True
            new_start_min = cur["start_max"]

        cur["start_min"] = new_start_min
        cur["end_min"] = new_start_min + cur["duration"]

    # --------------------------------------------------------
    # BACKWARD PASS: tighten end_max / start_max
    # --------------------------------------------------------

    for i in range(n - 2, -1, -1):

        nxt = clips[i + 1]
        cur = clips[i]

        new_end_max = min(
            cur["end_max"],
            nxt["start_max"]
        )

        if new_end_max < cur["end_min"]:

            cur["contradiction"] = True
            new_end_max = cur["end_min"]

        cur["end_max"] = new_end_max
        cur["start_max"] = new_end_max - cur["duration"]

    # --------------------------------------------------------
    # Recompute estimates and confidence after tightening.
    # --------------------------------------------------------

    for cur in clips:

        cur["estimated_start"] = (
            cur["start_min"] + cur["start_max"]
        ) / 2

        cur["estimated_end"] = (
            cur["end_min"] + cur["end_max"]
        ) / 2

        if (
            cur["confidence"] == "low"
            and (cur["start_max"] - cur["start_min"]) < 1.0
        ):

            # Propagation pinned this clip down almost
            # exactly, even though its own timestamp never
            # changed.

            cur["confidence"] = "high (propagated)"


# ============================================================
# CHOOSE TEMPLATE
# ============================================================

def choose_blank(
    min_blank,
    max_blank,
    estimated_blank
):
    """
    Choose the most likely available blank.

    1. Find templates inside the possible range.

    2. If templates exist:
           choose the one closest to the midpoint.

    3. If no template exists:
           choose the nearest template to the midpoint.

    OUTSIDE_RANGE is retained as a warning.
    """

    possible = [
        blank
        for blank in ALLOWED_BLANKS
        if min_blank <= blank <= max_blank
    ]

    # --------------------------------------------------------
    # Compatible template
    # --------------------------------------------------------

    if possible:

        chosen = min(
            possible,
            key=lambda blank:
                abs(blank - estimated_blank)
        )

        distance = abs(
            chosen - estimated_blank
        )

        return (
            chosen,
            possible,
            "COMPATIBLE",
            distance
        )

    # --------------------------------------------------------
    # No compatible template
    # --------------------------------------------------------

    chosen = min(
        ALLOWED_BLANKS,
        key=lambda blank:
            abs(blank - estimated_blank)
    )

    distance = abs(
        chosen - estimated_blank
    )

    return (
        chosen,
        possible,
        "OUTSIDE_RANGE",
        distance
    )


# ============================================================
# PROCESS JSON
# ============================================================

def process_timestamp_json(json_path):

    with open(
        json_path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    # --------------------------------------------------------
    # Collect timestamps in clip order.
    # --------------------------------------------------------

    all_timestamps = []

    for segments in data.values():

        for segment in segments:

            all_timestamps.append(
                segment["timestamp"]
            )

    # --------------------------------------------------------
    # Convert to continuous 12-hour timeline.
    # --------------------------------------------------------

    unwrapped_timestamps = unwrap_timestamps(
        all_timestamps
    )

    # --------------------------------------------------------
    # Analyze each clip independently.
    # --------------------------------------------------------

    clips = []

    timestamp_index = 0

    for filename, segments in data.items():

        count = len(segments)

        clip_timestamps = (
            unwrapped_timestamps[
                timestamp_index:
                timestamp_index + count
            ]
        )

        timestamp_index += count

        analysis = analyze_clip(
            segments,
            clip_timestamps
        )

        clips.append({
            "filename": filename,
            "duration": segments[-1]["endPos"],
            **analysis
        })

    # --------------------------------------------------------
    # Tighten low-confidence ranges using the rest of the
    # sequence (see propagate_constraints docstring).
    # --------------------------------------------------------

    propagate_constraints(clips)

    # --------------------------------------------------------
    # Calculate every transition.
    # --------------------------------------------------------

    decisions = []

    for i in range(len(clips) - 1):

        clip_a = clips[i]
        clip_b = clips[i + 1]

        (
            min_blank,
            max_blank,
            estimated_blank
        ) = calculate_blank(
            clip_a,
            clip_b
        )

        (
            chosen,
            possible,
            status,
            distance
        ) = choose_blank(
            min_blank,
            max_blank,
            estimated_blank
        )

        # ----------------------------------------------------
        # Calculate how well the selected blank sits inside
        # the possible range.
        # ----------------------------------------------------

        if max_blank > min_blank:

            range_width = (
                max_blank - min_blank
            )

            midpoint_position = (
                estimated_blank - min_blank
            ) / range_width

        else:

            range_width = 0

            midpoint_position = 0.5

        decisions.append({

            "clip_a":
                clip_a["filename"],

            "clip_b":
                clip_b["filename"],

            "min_blank":
                min_blank,

            "max_blank":
                max_blank,

            "estimated_blank":
                estimated_blank,

            "possible_blanks":
                possible,

            "chosen_blank":
                chosen,

            "status":
                status,

            "distance_from_estimate":
                distance,

            "range_width":
                range_width
        })

    return clips, decisions


# ============================================================
# PRINT REPORT
# ============================================================

def print_report(clips, decisions):

    print()

    print("=" * 75)
    print("CLIP ANALYSIS")
    print("=" * 75)

    for clip in clips:

        print()

        print(clip["filename"])

        print(
            f"  Duration: "
            f"{seconds_to_duration(clip['duration'])}"
        )

        print(
            f"  Start range: "
            f"{seconds_to_duration(clip['start_min'])}"
            f" - "
            f"{seconds_to_duration(clip['start_max'])}"
        )

        print(
            f"  Estimated start: "
            f"{seconds_to_duration(clip['estimated_start'])}"
        )

        print(
            f"  End range: "
            f"{seconds_to_duration(clip['end_min'])}"
            f" - "
            f"{seconds_to_duration(clip['end_max'])}"
        )

        print(
            f"  Estimated end: "
            f"{seconds_to_duration(clip['estimated_end'])}"
        )

        print(
            f"  Confidence: "
            f"{clip['confidence']}"
        )

        if clip.get("contradiction"):

            print(
                "  WARNING: sequence is inconsistent here "
                "(check for a misread timestamp)"
            )

    print()

    print("=" * 75)
    print("BLANK DECISIONS")
    print("=" * 75)

    for decision in decisions:

        print()

        print(
            f"{decision['clip_a']}"
            f" → "
            f"{decision['clip_b']}"
        )

        print(
            f"  Possible gap: "
            f"{seconds_to_duration(decision['min_blank'])}"
            f" - "
            f"{seconds_to_duration(decision['max_blank'])}"
        )

        print(
            f"  Estimated gap: "
            f"{seconds_to_duration(decision['estimated_blank'])}"
        )

        possible = [
            seconds_to_duration(blank)
            for blank in decision["possible_blanks"]
        ]

        print(
            f"  Compatible templates: "
            f"{possible}"
        )

        print(
            f"  CHOSEN BLANK: "
            f"{seconds_to_duration(decision['chosen_blank'])}"
        )

        print(
            f"  Status: "
            f"{decision['status']}"
        )

        print(
            f"  Distance from estimate: "
            f"{seconds_to_duration(decision['distance_from_estimate'])}"
        )




# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Run:
    #
    #     python3 blank_calculator.py
    #
    # Uses:
    #
    #     test.json
    #
    # Or:
    #
    #     python3 blank_calculator.py my_sequence.json
    #

    if len(sys.argv) > 1:

        json_file = Path(
            sys.argv[1]
        )

    else:

        json_file = Path(
            "test.json"
        )

    if not json_file.exists():

        print(
            f"ERROR: Could not find {json_file}"
        )

        sys.exit(1)

    clips, decisions = (
        process_timestamp_json(
            json_file
        )
    )

    print_report(
        clips,
        decisions
    )
