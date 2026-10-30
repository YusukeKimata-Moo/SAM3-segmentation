"""Read TIFF and grayscale raster images and prepare model inputs and previews."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
import tifffile


@dataclass
class ImageSettings:
    input_path: str
    axes: str = ""
    series: int = 0
    start: int = 0
    stop: int | None = None
    step: int = 1
    intensity_low: float | None = None
    intensity_high: float | None = None


def _natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", path.name)]


class TiffSequence:
    """Read TIFF sequences or raster frames; shape is (time, height, width)."""

    _SUPPORTED_AXES = set("TCZYXS")
    _TIFF_EXTENSIONS = {".tif", ".tiff"}
    _RASTER_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    def __init__(self, settings: ImageSettings):
        self.settings = settings
        source = Path(settings.input_path).expanduser()
        if not source.exists():
            raise FileNotFoundError(f"Input path does not exist: {source}")
        if settings.step <= 0:
            raise ValueError("step must be greater than zero")
        if settings.start < 0 or (settings.stop is not None and settings.stop < 0):
            raise ValueError("start and stop must be non-negative")
        if source.is_dir():
            files = sorted(
                (path for path in source.iterdir()
                 if path.is_file() and path.suffix.casefold() in self._TIFF_EXTENSIONS | self._RASTER_EXTENSIONS),
                key=lambda path: (_natural_key(path), path.name),
            )
            if not files:
                raise ValueError("The input directory contains no supported TIFF, JPG/JPEG, or PNG files")
        elif source.is_file() and source.suffix.casefold() in self._TIFF_EXTENSIONS | self._RASTER_EXTENSIONS:
            files = [source]
        else:
            raise ValueError("Input must be a TIFF, JPG/JPEG, or PNG file or a directory of supported images")

        self._files = files
        self._file_info: list[dict[str, object]] = []
        self._arrays: dict[int, np.ndarray] = {}
        self._offsets: list[int] = []
        common_height: int | None = None
        common_width: int | None = None
        common_dtype: str | None = None
        common_color_mode: str | None = None
        frame_total = 0

        for path in files:
            info = self._inspect_file(path) if path.suffix.casefold() in self._TIFF_EXTENSIONS else self._inspect_raster_file(path)
            self._offsets.append(frame_total)
            frame_total += int(info["frame_count"])
            if common_height is None:
                common_height, common_width = int(info["height"]), int(info["width"])
                common_dtype = str(info["dtype"])
                common_color_mode = str(info["color_mode"])
            elif (int(info["height"]) != common_height or int(info["width"]) != common_width
                  or str(info["dtype"]) != common_dtype or str(info["color_mode"]) != common_color_mode):
                raise ValueError("All image files in a sequence must have matching XY dimensions, dtype, and color mode")
            self._file_info.append(info)

        selected_stop = frame_total if settings.stop is None else min(settings.stop, frame_total)
        if settings.start >= frame_total or selected_stop <= settings.start:
            raise ValueError("The requested frame range contains no frames")
        self.frame_indices = tuple(range(settings.start, selected_stop, settings.step))
        self.shape = (len(self.frame_indices), int(common_height), int(common_width))
        self._source_frame_count = frame_total
        self._dtype = str(common_dtype)
        self.is_rgb = common_color_mode == "RGB"
        axes_by_file = [str(info["axes"]) for info in self._file_info]
        self.metadata = {
            "axes": axes_by_file[0] if len(set(axes_by_file)) == 1 else axes_by_file,
            "source_names": [path.name for path in files],
            "dtype": self._dtype,
            "color_mode": common_color_mode,
            "source_frame_count": frame_total,
            "selected_frame_indices": list(self.frame_indices),
        }

    def _inspect_file(self, path: Path) -> dict[str, object]:
        with tifffile.TiffFile(path) as tif:
            if not 0 <= self.settings.series < len(tif.series):
                raise ValueError(f"series index {self.settings.series} is outside the available range for {path.name}")
            series = tif.series[self.settings.series]
            shape = tuple(int(v) for v in series.shape)
            stored_dimensions = dict(zip(str(series.axes).upper(), shape))
            axes = self.settings.axes.upper() if self.settings.axes else str(series.axes).upper()
            if len(axes) != len(shape):
                raise ValueError(f"Axes '{axes}' do not match the {len(shape)} dimensions in {path.name}")
            if len(set(axes)) != len(axes):
                raise ValueError(f"Axes '{axes}' contain duplicate labels")
            dimensions = dict(zip(axes, shape))
            if stored_dimensions.get("Z", 1) > 1 or dimensions.get("Z", 1) > 1:
                raise ValueError("Z stacks are not supported. Use a 2D image or a 2D time-lapse sequence; do not relabel depth as time.")
            if stored_dimensions.get("C", 1) > 1 or dimensions.get("C", 1) > 1:
                raise ValueError("Only single-channel grayscale or RGB TIFF input is supported. Non-RGB multichannel images are not supported; do not relabel channels as time.")
            first_page = series.pages[0]
            is_rgb = first_page.photometric == tifffile.PHOTOMETRIC.RGB
            if is_rgb:
                if first_page.samplesperpixel != 3 or first_page.extrasamples or stored_dimensions.get("S") != 3:
                    raise ValueError("RGB TIFF input must contain exactly three color samples and no alpha channel.")
                if "S" not in axes or axes.index("S") != str(series.axes).upper().index("S"):
                    raise ValueError("Keep the RGB sample axis S in its stored position; only override unknown time axes.")
            elif (first_page.samplesperpixel != 1 or dimensions.get("S", 1) > 1 or stored_dimensions.get("S", 1) > 1
                  or first_page.photometric not in (tifffile.PHOTOMETRIC.MINISBLACK, tifffile.PHOTOMETRIC.MINISWHITE)):
                raise ValueError("Use single-channel grayscale TIFFs or TIFFs tagged as RGB; color samples cannot be inferred from an axis override.")
            unsupported = set(axes) - self._SUPPORTED_AXES
            if unsupported:
                if not self.settings.axes:
                    raise ValueError(f"Ambiguous or unsupported TIFF axes {axes!r} in {path.name}; provide an explicit axes override")
                raise ValueError(f"Unsupported axis labels {sorted(unsupported)} in explicit axes override")
            if "Y" not in axes or "X" not in axes:
                raise ValueError(f"TIFF axes must include Y and X in {path.name}")
            dtype = np.dtype(series.dtype)
            if dtype not in (np.dtype("uint8"), np.dtype("uint16")):
                raise ValueError(f"Only uint8 and uint16 TIFF input is supported; {path.name} has {dtype}")
            return {
                "shape": shape,
                "axes": axes,
                "frame_count": dimensions.get("T", 1),
                "height": dimensions["Y"],
                "width": dimensions["X"],
                "dtype": dtype.name,
                "color_mode": "RGB" if is_rgb else "grayscale",
            }

    def _inspect_raster_file(self, path: Path) -> dict[str, object]:
        if self.settings.series != 0:
            raise ValueError(f"Raster image series must be 0 for {path.name}")
        axes = self.settings.axes.upper()
        if axes not in ("", "YX"):
            raise ValueError(f"Raster image axes must be empty or 'YX' for {path.name}")

        extension = path.suffix.casefold()
        try:
            with Image.open(path) as image:
                expected_format = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG"}[extension]
                if image.format != expected_format:
                    raise ValueError(
                        f"Image format {image.format!r} does not match the {extension} extension for {path.name}"
                    )
                if expected_format == "PNG" and (getattr(image, "is_animated", False) or getattr(image, "n_frames", 1) != 1):
                    raise ValueError(f"Animated PNG images are not supported: {path.name}")
                if expected_format == "PNG" and ("transparency" in image.info or "tRNS" in image.info):
                    raise ValueError(f"PNG transparency is not supported: {path.name}")
                grayscale_modes = {"1", "L", "I", "I;16", "I;16B", "I;16L"}
                if image.mode not in grayscale_modes | {"RGB"}:
                    raise ValueError(
                        f"Only grayscale or RGB JPG/JPEG and PNG images are supported; {path.name} has mode {image.mode!r}"
                    )
                if expected_format == "JPEG" and image.mode not in {"L", "RGB"}:
                    raise ValueError(
                        f"JPEG input must be 8-bit grayscale or RGB; {path.name} has mode {image.mode!r}"
                    )
                if expected_format == "PNG" and image.mode in {"I", "I;16", "I;16B", "I;16L"}:
                    pixels = np.asarray(image)
                    if pixels.size and (int(pixels.min()) < 0 or int(pixels.max()) > 65535):
                        raise ValueError(f"16-bit grayscale PNG values must be within 0..65535: {path.name}")
                    dtype = "uint16"
                else:
                    dtype = "uint8"
                height, width = image.height, image.width
                color_mode = "RGB" if image.mode == "RGB" else "grayscale"
        except (OSError, SyntaxError) as error:
            raise ValueError(f"Could not read raster image {path.name}: {error}") from error
        return {
            "shape": (height, width),
            "axes": "YXS" if color_mode == "RGB" else "YX",
            "frame_count": 1,
            "height": height,
            "width": width,
            "dtype": dtype,
            "color_mode": color_mode,
            "format": "raster",
            "mode": image.mode,
        }


    def _load_file(self, file_index: int) -> np.ndarray:
        if file_index not in self._arrays:
            self._arrays.clear()
            path = self._files[file_index]
            if self._file_info[file_index].get("format") == "raster":
                with Image.open(path) as image:
                    arr = np.asarray(image)
                if arr.dtype.kind == "i":
                    arr = arr.astype(np.uint16)
                if self._file_info[file_index].get("mode") == "1":
                    arr = arr.astype(np.uint8) * 255
                canonical = arr[None, None, None, :, :, :] if self._file_info[file_index]["color_mode"] == "RGB" else arr[None, None, None, :, :, None]
            else:
                with tifffile.TiffFile(path) as tif:
                    arr = tif.series[self.settings.series].asarray()
                axes = str(self._file_info[file_index]["axes"])
                present = list(axes)
                for axis in "TCZYXS":
                    if axis not in present:
                        arr = np.expand_dims(arr, axis=arr.ndim)
                        present.append(axis)
                permutation = [present.index(axis) for axis in "TCZYXS"]
                canonical = np.transpose(arr, permutation)
            self._arrays[file_index] = canonical
        return self._arrays[file_index]

    def _locate(self, original_index: int) -> tuple[int, int]:
        if original_index < 0 or original_index >= self._source_frame_count:
            raise IndexError(f"frame index {original_index} is outside the source range")
        file_index = len(self._offsets) - 1
        for candidate, offset in enumerate(self._offsets):
            end = offset + int(self._file_info[candidate]["frame_count"])
            if offset <= original_index < end:
                file_index = candidate
                break
        return file_index, original_index - self._offsets[file_index]

    def raw_frame(self, index: int) -> np.ndarray:
        """Return raw YX grayscale or YXS RGB values in the original integer dtype."""
        file_index, local_time = self._locate(int(index))
        info = self._file_info[file_index]
        if info.get("format") == "raster":
            frame = self._load_file(file_index)[0, 0, 0]
            return frame if self.is_rgb else frame[..., 0]
        axes = str(info["axes"])
        shape = tuple(info["shape"])
        pixel_axes = ("YXS" if axes.endswith("YXS") else "SYX") if self.is_rgb else "YX"
        if axes.endswith(pixel_axes):
            with tifffile.TiffFile(self._files[file_index]) as tif:
                pages = tif.series[self.settings.series].pages
                leading_shape = shape[:-len(pixel_axes)]
                page_count = math.prod(leading_shape)
                if len(pages) == page_count and tuple(pages[0].shape) == shape[-len(pixel_axes):]:
                    selected = {"T": local_time, "C": 0, "Z": 0, "S": 0}
                    indices = tuple(selected[axis] for axis in axes[:-len(pixel_axes)])
                    page_index = int(np.ravel_multi_index(indices, leading_shape)) if leading_shape else 0
                    frame = pages[page_index].asarray()
                    return np.moveaxis(frame, 0, -1) if pixel_axes == "SYX" else frame
        frame = np.asarray(self._load_file(file_index)[local_time, 0, 0])
        return frame if self.is_rgb else frame[..., 0]

    @staticmethod
    def _scale_to_rgb(frame: np.ndarray, low: float, high: float) -> np.ndarray:
        if high <= low:
            pixels = np.zeros(frame.shape, dtype=np.uint8)
        else:
            scaled = ((np.asarray(frame, dtype=np.float64) - low) / (high - low)) * 255.0
            pixels = np.clip(np.rint(scaled), 0, 255).astype(np.uint8)
        return pixels if frame.ndim == 3 else np.repeat(pixels[:, :, None], 3, axis=2)

    def preview(self, index: int, low: float | None = None, high: float | None = None) -> np.ndarray:
        if low is None:
            low = self.settings.intensity_low
        if high is None:
            high = self.settings.intensity_high
        if low is None or high is None:
            sample_low, sample_high = self._sample_limits()
            low = sample_low if low is None else low
            high = sample_high if high is None else high
        if not math.isfinite(low) or not math.isfinite(high):
            raise ValueError("Display intensity limits must be finite")
        if high < low:
            raise ValueError("high must be greater than or equal to low")
        return self._scale_to_rgb(self.raw_frame(index), float(low), float(high))

    def _sample_limits(self) -> tuple[float, float]:
        if self.settings.intensity_low is not None and self.settings.intensity_high is not None:
            return float(self.settings.intensity_low), float(self.settings.intensity_high)
        if self.is_rgb:
            return 0.0, float(np.iinfo(self._dtype).max)
        sampled: list[np.ndarray] = []
        count = min(32, len(self.frame_indices))
        positions = np.linspace(0, len(self.frame_indices) - 1, count, dtype=int)
        for position in np.unique(positions):
            frame = self.raw_frame(self.frame_indices[int(position)])
            flat = frame.ravel()
            stride = max(1, math.ceil(flat.size / 65536))
            sampled.append(flat[::stride].astype(np.float64, copy=False))
        if not sampled:
            return 0.0, 0.0
        values = np.concatenate(sampled)
        computed_low, computed_high = (float(v) for v in np.percentile(values, [1.0, 99.8]))
        return computed_low, computed_high

    def prepare(self, cache_dir: str | Path) -> dict[str, object]:
        destination = Path(cache_dir).expanduser()
        if destination.exists():
            raise FileExistsError(f"Cache path already exists: {destination}")
        low = self.settings.intensity_low
        high = self.settings.intensity_high
        if low is None or high is None:
            sampled_low, sampled_high = self._sample_limits()
            low = sampled_low if low is None else float(low)
            high = sampled_high if high is None else float(high)
        low, high = float(low), float(high)
        if high < low:
            raise ValueError("intensity_high must be greater than or equal to intensity_low")
        if not math.isfinite(low) or not math.isfinite(high):
            raise ValueError("Intensity limits must be finite")
        destination.mkdir(parents=True, exist_ok=False)
        for output_index, source_index in enumerate(self.frame_indices):
            rgb = self._scale_to_rgb(self.raw_frame(source_index), low, high)
            Image.fromarray(rgb, mode="RGB").save(destination / f"{output_index:06d}.png", format="PNG")
        return {
            "resource_path": str(destination.resolve()),
            "low": low,
            "high": high,
            "frame_count": len(self.frame_indices),
            "height": self.shape[1],
            "width": self.shape[2],
        }

    def fingerprint(self) -> dict[str, object]:
        sources = []
        for path in self._files:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            sources.append({"name": path.name, "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()})
        settings = asdict(self.settings)
        settings["input_path"] = Path(self.settings.input_path).name
        payload = {"settings": settings, "sources": sources}
        stable_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return {**payload, "sha256": hashlib.sha256(stable_json.encode("utf-8")).hexdigest()}


def labels_from_masks(object_ids: Sequence[int], masks: Sequence[np.ndarray]) -> np.ndarray:
    """Combine same-shaped boolean masks into an integer label image."""
    if len(object_ids) != len(masks):
        raise ValueError("object_ids and masks must have the same length")
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) for value in object_ids):
        raise ValueError("object IDs must be integers")
    ids = [int(value) for value in object_ids]
    if any(value <= 0 for value in ids):
        raise ValueError("object IDs must be positive integers")
    if len(set(ids)) != len(ids):
        raise ValueError("object IDs must be unique")
    if any(value > np.iinfo(np.uint32).max for value in ids):
        raise ValueError("object IDs must not exceed the uint32 maximum")
    shape: tuple[int, ...] | None = None
    normalized: list[tuple[int, np.ndarray]] = []
    for object_id, mask in zip(ids, masks):
        array = np.asarray(mask)
        if array.ndim != 2:
            raise ValueError("Each mask must be a two-dimensional YX array")
        if array.dtype != np.bool_:
            raise ValueError("Each mask must have boolean dtype")
        if shape is None:
            shape = array.shape
        elif array.shape != shape:
            raise ValueError("All masks must have matching shapes")
        normalized.append((object_id, array))
    if shape is None:
        raise ValueError("At least one mask is required to determine the output shape")
    maximum_id = max(ids)
    dtype = np.uint16 if maximum_id <= np.iinfo(np.uint16).max else np.uint32
    labels = np.zeros(shape, dtype=dtype)
    for object_id, mask in sorted(normalized, key=lambda pair: pair[0]):
        labels[mask & (labels == 0)] = object_id
    return labels
