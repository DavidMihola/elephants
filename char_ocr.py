import glob
import os

import cv2

TEMPLATE_DIR = "char_templates"
STANDARD_SIZE = (40, 60)  # (width, height) every saved/matched character is resized to
VALID_CHARS = "0123456789:"


def char_dirname(ch):
    return "colon" if ch == ":" else ch


def segment_characters(gray_crop):
    """Splits a cropped timestamp strip into per-character images by finding
    all-black gap columns between characters (colon's two dots share an x-range
    with each other, so they stay together as one segment)."""
    _, binary = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    col_has_ink = binary.sum(axis=0) > 0

    segments = []
    in_char = False
    start = 0
    for x, ink in enumerate(col_has_ink):
        if ink and not in_char:
            in_char, start = True, x
        elif not ink and in_char:
            in_char = False
            segments.append((start, x))
    if in_char:
        segments.append((start, len(col_has_ink)))

    return segments, binary


_templates_cache = None


def _load_templates():
    global _templates_cache
    if _templates_cache is None:
        templates = {}
        for ch in VALID_CHARS:
            paths = glob.glob(os.path.join(TEMPLATE_DIR, char_dirname(ch), "*.png"))
            templates[ch] = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in paths]
        _templates_cache = templates
    return _templates_cache


def match_character(char_img):
    """Returns (best_char, best_score) for a single segmented+resized character image."""
    best_char, best_score = None, -2.0
    for ch, examples in _load_templates().items():
        for template in examples:
            score = float(cv2.matchTemplate(char_img, template, cv2.TM_CCOEFF_NORMED)[0][0])
            if score > best_score:
                best_score, best_char = score, ch
    return best_char, best_score


def recognize_timestamp(gray_crop, min_score=0.5):
    """Segments a cropped HH:MM strip and recognizes each character via template
    matching. Returns the recognized string (e.g. '12:55') or None if segmentation
    didn't yield a plausible 5-character HH:MM shape or any character matched poorly."""
    segments, binary = segment_characters(gray_crop)
    if len(segments) != 5:
        return None

    chars = []
    for x0, x1 in segments:
        char_img = cv2.resize(binary[:, x0:x1], STANDARD_SIZE, interpolation=cv2.INTER_AREA)
        ch, score = match_character(char_img)
        if ch is None or score < min_score:
            return None
        chars.append(ch)

    if chars[2] != ":":
        return None
    return "".join(chars)
