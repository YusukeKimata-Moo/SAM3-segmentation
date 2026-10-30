"""Render current object names and build portable, ID-qualified output paths."""

from functools import lru_cache
from pathlib import Path
import re
import warnings

import numpy as np
from PIL import Image, ImageDraw, ImageFont


LABEL_FORMAT = "{object_id}: {name} (default Object {object_id} names display the ID only)"


def object_label(object_id, item):
    """Read the current name without changing the stable application ID."""
    name = item["name"]
    return str(object_id) if name == f"Object {object_id}" else f"{object_id}: {name}"


def mask_folder_name(object_id, item):
    """Return one portable path component, bounded in characters and UTF-8 bytes."""
    prefix = f"object_{object_id:06d}"
    if item["name"] == f"Object {object_id}":
        return prefix
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f\x7f-\x9f]', "_", item["name"])
    name = re.sub(r"\s+", "_", name.strip(" ."))[:80]
    name = name.encode("utf-8")[:160].decode("utf-8", errors="ignore").rstrip(" .")
    return f"{prefix}_{name or 'unnamed'}"


@lru_cache(maxsize=1)
def _font_paths():
    # Use an installed system font supporting the object names.
    from matplotlib import font_manager

    paths = set(font_manager.findSystemFonts())
    paths.add(font_manager.findfont("DejaVu Sans"))
    preferred = ("meiryo", "yugoth", "msgothic", "noto", "ipa", "arialuni", "dejavu")

    def priority(path):
        name = Path(path).name.lower()
        return next((index for index, token in enumerate(preferred) if token in name), len(preferred)), name

    return tuple(sorted(paths, key=priority))


@lru_cache(maxsize=128)
def _font_path(text):
    from matplotlib.ft2font import FT2Font

    required = {ord(character) for character in text if not character.isspace()}
    best_path, best_coverage = None, -1
    for path in _font_paths():
        try:
            characters = FT2Font(path).get_charmap()
        except (OSError, RuntimeError, ValueError):
            continue
        coverage = len(required.intersection(characters))
        if coverage == len(required):
            return path
        if coverage > best_coverage:
            best_path, best_coverage = path, coverage
    warnings.warn("Available fonts do not support every object-name character; "
                  "unsupported characters may appear as missing glyphs. Install a suitable system font.",
                  RuntimeWarning, stacklevel=2)
    return best_path


def draw_object_label(image, mask, object_id, item, *, selected=False):
    """Draw a white outlined name, scaling and placing the full label inside the canvas."""
    rows, columns = np.nonzero(mask)
    if not len(rows):
        return
    text = object_label(object_id, item)
    # Keep each name on one line, including names supplied through Apply points.
    text = " ".join(text.splitlines())
    font_size = max(12, min(24, round(min(image.size) / 32))) + (2 if selected else 0)
    path = _font_path(text)
    font = ImageFont.truetype(path, font_size) if path else ImageFont.load_default()
    drawing = ImageDraw.Draw(image)
    try:
        left, top, right, bottom = drawing.textbbox((0, 0), text, font=font, stroke_width=1)
    except UnicodeEncodeError:
        warnings.warn("The fallback font cannot render this object name; the image label uses an explicit placeholder.",
                      RuntimeWarning, stacklevel=2)
        text = f"{object_id}: [unsupported name]"
        left, top, right, bottom = drawing.textbbox((0, 0), text, font=font, stroke_width=1)
    label = Image.new("RGBA", (max(1, right - left), max(1, bottom - top)))
    ImageDraw.Draw(label).text((-left, -top), text, font=font, fill="white",
                               stroke_width=1, stroke_fill="black")
    width, height = image.size
    margin = 2 if min(width, height) > 4 else 0
    scale = min(1, (width - 2 * margin) / label.width, (height - 2 * margin) / label.height)
    if scale < 1:
        label = label.resize((max(1, int(label.width * scale)), max(1, int(label.height * scale))),
                             Image.Resampling.LANCZOS)
    x = min(max(margin, round(float(columns.mean()) - label.width / 2)), width - margin - label.width)
    y = min(max(margin, round(float(rows.mean()) - label.height / 2)), height - margin - label.height)
    image.paste(label, (x, y), label)
