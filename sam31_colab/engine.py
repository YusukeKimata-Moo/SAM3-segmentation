"""Track selected image frames and export masks with reproducible prompt history."""

from __future__ import annotations

import colorsys
import copy
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image
import tifffile

from .io import labels_from_masks
from .labels import LABEL_FORMAT, draw_object_label, mask_folder_name


PROJECT_VERSION = 1
NAMING_POLICY = "concept-prefix-v1"


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)




class TrackingProject:
    """Track and refine detections; use a new project to recover a failed session."""

    def __init__(self, sequence, predictor, prepared, environment=None, max_objects=16):
        self.sequence = sequence
        self.predictor = predictor
        self.prepared = dict(prepared)
        self.environment = dict(environment or {})
        self.max_objects = _integer(max_objects, "max_objects", 1)
        if self.max_objects > 256:
            raise ValueError("max_objects must be at most 256 and match the predictor configuration")
        configured_limit = self.environment.get("max_objects")
        if configured_limit is not None and configured_limit != self.max_objects:
            raise ValueError("max_objects does not match the loaded predictor")
        count, height, width = sequence.shape
        if tuple(prepared.get(key) for key in ("frame_count", "height", "width")) != (count, height, width):
            raise ValueError("Prepared frames do not match the selected image sequence")
        for key in ("low", "high"):
            if not math.isfinite(float(prepared[key])):
                raise ValueError("Prepared intensity limits must be finite")
        self.objects = {}
        self.results = {}
        self.current_frame_outputs = {}
        self.current_frame_index = None
        self.events = []
        self.dirty = True
        self.rebuild_required = False
        self._closed = False
        self._raw_to_app = {}
        self._next_app_id = 1
        self._points = {}
        self._has_prompts = False
        self._concept = False
        self._concept_name = ""
        self._concept_prompt = None
        self.name_prefix = ""
        self._naming_policy = NAMING_POLICY
        self._coverage = []
        response = predictor.handle_request({
            "type": "start_session", "resource_path": prepared["resource_path"],
            "offload_video_to_cpu": True, "offload_state_to_cpu": False,
        })
        self.session_id = response["session_id"]

    def _ready(self):
        if self._closed:
            raise RuntimeError("This project is closed")
        if self.rebuild_required:
            raise RuntimeError("The predictor session failed; save successful events and replay them in a new project")

    def _frame(self, frame):
        frame = _integer(frame, "frame")
        if frame >= self.sequence.shape[0]:
            raise ValueError("Frame is outside the selected sequence")
        return frame

    def _object(self, object_id):
        object_id = _integer(object_id, "object_id", 1)
        if object_id not in self.objects:
            raise ValueError("Unknown object ID")
        return self.objects[object_id]

    def _invalidate(self):
        self.dirty = True
        self.results = {}
        self.current_frame_outputs = {}
        self.current_frame_index = None
        self._coverage = []

    def _register(self, raw_id):
        raw_id = _integer(raw_id, "predictor object ID")
        if raw_id in self._raw_to_app:
            return self._raw_to_app[raw_id]
        if sum(not item["excluded"] for item in self.objects.values()) >= self.max_objects:
            raise RuntimeError("Predictor object count exceeds the configured limit")
        app_id = self._next_app_id
        self._next_app_id += 1
        color = [round(channel * 255) for channel in colorsys.hsv_to_rgb((app_id * 0.61803398875) % 1, 0.75, 1)]
        self.objects[app_id] = {"name": f"{self.effective_name_prefix} {app_id}", "color": color,
                                "raw_id": raw_id, "excluded": False}
        self._raw_to_app[raw_id] = app_id
        return app_id

    def _unused_raw_id(self):
        maximum = max(self._raw_to_app, default=-1)
        sessions = getattr(self.predictor, "_all_inference_states", {})
        state = sessions.get(self.session_id, {}).get("state", {})
        metadata = state.get("tracker_metadata", {})
        if metadata:
            maximum = max(maximum, int(metadata.get("max_obj_id", -1)))
            known_ids = _array(metadata.get("obj_ids_all_gpu", [])).ravel()
            if known_ids.size:
                maximum = max(maximum, int(known_ids.max()))
            if known_ids.size >= self.max_objects:
                raise ValueError("Configured object limit reached, including currently hidden tracker objects")
        return maximum + 1

    def _decode(self, response, expected_frame=None):
        frame = self._frame(response["frame_index"])
        if expected_frame is not None and frame != expected_frame:
            raise ValueError("Predictor returned a different prompt frame")
        outputs = response["outputs"]
        raw_ids = _array(outputs["out_obj_ids"])
        masks = _array(outputs["out_binary_masks"])
        if raw_ids.ndim != 1 or not np.issubdtype(raw_ids.dtype, np.integer):
            raise ValueError("Predictor IDs must be a one-dimensional integer array")
        if len(set(raw_ids.tolist())) != len(raw_ids) or np.any(raw_ids < 0):
            raise ValueError("Predictor IDs must be unique nonnegative integers")
        if masks.shape != (len(raw_ids), *self.sequence.shape[1:]) or masks.dtype != np.bool_:
            raise ValueError("Predictor masks must be boolean arrays of shape object,Y,X")
        for raw_id in sorted(raw_ids.tolist()):
            self._register(raw_id)
        result = {}
        signatures = {}
        for raw_id, mask in zip(raw_ids.tolist(), masks):
            app_id = self._raw_to_app[raw_id]
            signatures[str(raw_id)] = hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()
            if not self.objects[app_id]["excluded"]:
                result[app_id] = np.array(mask, dtype=bool, copy=True)
        return frame, result, signatures

    def _request(self, request, frame, reserve_raw=None):
        saved_objects = copy.deepcopy(self.objects)
        saved_mapping = dict(self._raw_to_app)
        saved_next = self._next_app_id
        self._invalidate()
        try:
            if reserve_raw is not None:
                self._register(reserve_raw)
            response = self.predictor.handle_request({**request, "session_id": self.session_id})
            _, masks, signatures = self._decode(response, frame)
        except BaseException:
            self.objects, self._raw_to_app, self._next_app_id = saved_objects, saved_mapping, saved_next
            self.rebuild_required = True
            raise
        self.current_frame_outputs = masks
        self.current_frame_index = frame
        return masks, signatures

    def add_concept(self, frame, text, boxes=None, box_labels=None):
        """Apply the initial text and/or normalized XYWH boxes before point prompts."""
        self._ready()
        frame = self._frame(frame)
        if self._has_prompts:
            raise ValueError("Use update_concept to redetect after concept or point prompts")
        text, boxes, box_labels = self._validate_concept(text, boxes, box_labels)
        return self._apply_concept(frame, text, boxes, box_labels, redetect=False)

    @property
    def concept_prompt(self):
        """Return the saved reference frame, text, and normalized example boxes."""
        return copy.deepcopy(self._concept_prompt)

    def update_concept(self, frame, text, boxes=None, box_labels=None, *, mode="add", name_prefix=None):
        """Add or replace saved examples, then redetect with fresh object identities."""
        self._ready()
        frame = self._frame(frame)
        if mode not in ("add", "replace"):
            raise ValueError("Box update must be add or replace")
        if name_prefix is not None and not isinstance(name_prefix, str):
            raise ValueError("Name prefix must be a string")
        boxes = [] if boxes is None else list(boxes)
        box_labels = [] if box_labels is None else list(box_labels)
        saved = self._concept_prompt
        if mode == "add" and saved is not None and saved["boxes"]:
            if frame != saved["frame"]:
                raise ValueError(f"Saved examples use reference frame {saved['frame']}; use the reference frame or Replace saved")
            boxes = saved["boxes"] + boxes
            box_labels = saved["box_labels"] + box_labels
        text, boxes, box_labels = self._validate_concept(text, boxes, box_labels)
        if name_prefix is not None and name_prefix.strip() != self.name_prefix:
            self.set_name_prefix(name_prefix)
        return self._apply_concept(frame, text, boxes, box_labels, redetect=self._has_prompts)

    def _validate_concept(self, text, boxes, box_labels):
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        text = text.strip()
        boxes = [] if boxes is None else boxes
        box_labels = [] if box_labels is None else box_labels
        if len(boxes) != len(box_labels):
            raise ValueError("Each box requires a 0 or 1 label")
        clean_boxes = []
        for box in boxes:
            if len(box) != 4:
                raise ValueError("Boxes must contain normalized x,y,width,height")
            x, y, width, height = [float(value) for value in box]
            if not all(math.isfinite(value) for value in (x, y, width, height)) or not (
                0 <= x < 1 and 0 <= y < 1 and 0 < width <= 1 - x and 0 < height <= 1 - y
            ):
                raise ValueError("Boxes must be finite normalized XYWH coordinates inside the image")
            clean_boxes.append([x, y, width, height])
        clean_labels = self._labels(box_labels)
        if not text and not clean_boxes:
            raise ValueError("Enter a concept or at least one box")
        return text, clean_boxes, clean_labels

    def _apply_concept(self, frame, text, boxes, box_labels, *, redetect):
        request = {"type": "add_prompt", "frame_index": frame}
        if text:
            request["text"] = text
        if boxes:
            request.update(bounding_boxes=boxes, bounding_box_labels=box_labels)
        saved_objects = copy.deepcopy(self.objects)
        saved_mapping, saved_next, saved_points = dict(self._raw_to_app), self._next_app_id, self._points
        previous_concept_name = self._concept_name
        if redetect:
            for item in self.objects.values():
                item["excluded"] = True
            self._raw_to_app = {}
            self._points = {}
        self._concept_name = text
        try:
            masks, signatures = self._request(request, frame)
        except BaseException:
            self.objects, self._raw_to_app, self._next_app_id = saved_objects, saved_mapping, saved_next
            self._points = saved_points
            self._concept_name = previous_concept_name
            self._invalidate()
            self.rebuild_required = True
            raise
        self._has_prompts = self._concept = True
        self._concept_prompt = {"frame": frame, "text": text, "boxes": copy.deepcopy(boxes),
                                "box_labels": list(box_labels)}
        self.events.append({"type": "redetect" if redetect else "concept", "frame": frame, "text": text,
                            "boxes": copy.deepcopy(boxes), "box_labels": list(box_labels), "signatures": signatures})
        return masks

    @staticmethod
    def _labels(labels):
        result = [_integer(value, "prompt label") for value in labels]
        if any(value not in (0, 1) for value in result):
            raise ValueError("Prompt labels must be 0 or 1")
        return result

    def add_points(self, frame, object_id, points, labels, name=""):
        """Append pixel XY points to this object's points on this frame."""
        self._ready()
        frame = self._frame(frame)
        if not isinstance(name, str):
            raise ValueError("name must be a string")
        if not len(points) or len(points) != len(labels):
            raise ValueError("Provide points and one 0 or 1 label per point")
        clean_labels = self._labels(labels)
        clean_points = []
        height, width = self.sequence.shape[1:]
        for point in points:
            if len(point) != 2:
                raise ValueError("Points must contain pixel x,y coordinates")
            x, y = [float(value) for value in point]
            if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < width and 0 <= y < height):
                raise ValueError("Point coordinates must be finite and inside the image")
            clean_points.append([x, y])
        is_new = object_id is None
        if is_new:
            if sum(not item["excluded"] for item in self.objects.values()) >= self.max_objects:
                raise ValueError("Configured object limit reached")
            if 1 not in clean_labels:
                raise ValueError("A new object requires at least one positive point")
            raw_id = self._unused_raw_id()
            object_id = self._next_app_id
        else:
            object_id = _integer(object_id, "object_id", 1)
            item = self._object(object_id)
            if item["excluded"]:
                raise ValueError("Excluded objects cannot receive points")
            raw_id = item["raw_id"]
        previous_points, previous_labels = self._points.get((frame, object_id), ([], []))
        combined_points = previous_points + clean_points
        combined_labels = previous_labels + clean_labels
        normalized_points = [[x / width, y / height] for x, y in combined_points]
        request = {"type": "add_prompt", "frame_index": frame, "obj_id": raw_id,
                   "points": normalized_points, "point_labels": combined_labels,
                   "clear_old_points": True, "rel_coordinates": True}
        masks, signatures = self._request(request, frame, raw_id if is_new else None)
        self._points[frame, object_id] = (combined_points, combined_labels)
        if name:
            self.objects[object_id]["name"] = name
        self._has_prompts = True
        self.events.append({"type": "points", "frame": frame, "object_id": object_id,
                            "new_object": is_new, "raw_id": raw_id, "points": clean_points,
                            "labels": clean_labels, "name": name, "signatures": signatures})
        return masks

    def remove_object(self, object_id, frame=0):
        self._ready()
        frame = self._frame(frame)
        object_id = _integer(object_id, "object_id", 1)
        item = self._object(object_id)
        if item["excluded"]:
            raise ValueError("Object is already excluded")
        masks, signatures = self._request({"type": "remove_object", "frame_index": frame,
                                           "obj_id": item["raw_id"], "is_user_action": False}, frame)
        self.objects[object_id]["excluded"] = True
        masks.pop(object_id, None)
        self.events.append({"type": "remove", "object_id": object_id, "frame": frame,
                            "signatures": signatures})
        return masks

    def rename_object(self, object_id, name):
        """Record a human-assigned object name."""
        self._ready()
        object_id = _integer(object_id, "object_id", 1)
        item = self._object(object_id)
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Object name must be nonempty")
        item.update(name=name.strip())
        self.events.append({"type": "rename", "object_id": object_id,
                            "name": name.strip()})

    @property
    def effective_name_prefix(self):
        """Use the entered prefix, the concept text, or the default Object label."""
        concept = self._concept_name if self._naming_policy == NAMING_POLICY else ""
        return self.name_prefix or concept or "Object"

    def set_name_prefix(self, prefix, *, rename_existing=False):
        """Name future objects; optionally replace every active object's current name."""
        self._ready()
        if not isinstance(prefix, str):
            raise ValueError("Name prefix must be a string")
        if not isinstance(rename_existing, bool):
            raise ValueError("rename_existing must be a boolean")
        prefix = prefix.strip()
        self.name_prefix = prefix
        if rename_existing:
            for object_id, item in self.objects.items():
                if not item["excluded"]:
                    item["name"] = f"{self.effective_name_prefix} {object_id}"
        self.events.append({"type": "set_name_prefix", "prefix": prefix,
                            "rename_existing": rename_existing})

    def propagate(self, start_frame=0, direction="both", progress=None):
        """Replace the result snapshot; export requires coverage of every selected frame."""
        self._ready()
        start_frame = self._frame(start_frame)
        if direction not in {"both", "forward", "backward"}:
            raise ValueError("direction must be both, forward, or backward")
        if not self._has_prompts:
            raise ValueError("Add a concept or points before propagation")
        frame_count = self.sequence.shape[0]
        expected = set(range(frame_count)) if direction == "both" else (
            set(range(start_frame, frame_count)) if direction == "forward" else set(range(start_frame + 1)))
        saved_objects = copy.deepcopy(self.objects)
        saved_mapping = dict(self._raw_to_app)
        saved_next = self._next_app_id
        self._invalidate()
        pending = {}
        signatures = {}
        duplicate_anchor_seen = False
        try:
            responses = self.predictor.handle_stream_request({
                "type": "propagate_in_video", "session_id": self.session_id,
                "propagation_direction": direction, "start_frame_index": start_frame,
                "max_frame_num_to_track": frame_count,
            })
            for response in responses:
                frame, masks, signature = self._decode(response)
                if frame not in expected:
                    raise ValueError("Predictor returned a frame outside the requested propagation range")
                if frame in pending:
                    if direction != "both" or frame != start_frame or duplicate_anchor_seen:
                        raise ValueError("Predictor returned an unexpected duplicate frame")
                    duplicate_anchor_seen = True
                    continue
                pending[frame] = masks
                signatures[str(frame)] = signature
                self.current_frame_outputs = masks
                self.current_frame_index = frame
                if progress:
                    progress(len(pending), len(expected))
            if set(pending) != expected:
                raise RuntimeError("Propagation ended before all requested frames were returned")
        except BaseException:
            self.objects, self._raw_to_app, self._next_app_id = saved_objects, saved_mapping, saved_next
            self._invalidate()
            self.rebuild_required = True
            raise
        self.results = dict(sorted(pending.items()))
        self._coverage = sorted(expected)
        self.dirty = expected != set(range(frame_count))
        self.events.append({"type": "propagate", "start_frame": start_frame,
                            "direction": direction, "signatures": signatures})
        return self.results

    def _project_data(self):
        allowed = {"python", "torch", "cuda", "gpu", "free_vram_gib", "total_vram_gib",
                   "sam_revision", "checkpoint_revision", "checkpoint_file", "max_objects",
                   "flash_attention_3", "compile", "packages"}
        environment = {key: value for key, value in self.environment.items() if key in allowed}
        if "packages" in environment:
            packages = environment["packages"]
            environment["packages"] = {
                key: value for key, value in packages.items()
                if isinstance(key, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", key)
                and isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9.+!_-]+", value)
            } if isinstance(packages, dict) else {}
        for key, value in list(environment.items()):
            if isinstance(value, str) and ("/" in value or "\\" in value):
                environment.pop(key)
        return {"schema": "sam31-colab-project", "version": PROJECT_VERSION,
                "naming_policy": self._naming_policy,
                "fingerprint": self.sequence.fingerprint(),
                "prepared": {key: self.prepared[key] for key in ("low", "high", "frame_count", "height", "width")},
                "max_objects": self.max_objects, "environment": environment,
                "events": copy.deepcopy(self.events),
                "objects": {str(key): copy.deepcopy(value) for key, value in self.objects.items()},
                "coverage": self._coverage, "complete": not self.dirty,
                "overlap_policy": "lowest positive object ID wins in overlays; individual masks are preserved",
                "duplicate_anchor_policy": "first (forward) output wins",
                "point_policy": "append per object and frame; resend accumulated points with clear_old_points=true"}

    def save_project(self, path, *, export_info=None):
        """Save JSON without input/cache absolute paths or authentication values."""
        destination = Path(path).expanduser()
        self._safe_output(destination)
        data = self._project_data()
        if export_info is not None:
            data["export"] = copy.deepcopy(export_info)
        serialized = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
        return destination

    def replay_project(self, path):
        """Replay on a fresh project; exact mask identity checks can reject nondeterministic inference."""
        self._ready()
        if self.events or self.objects:
            raise ValueError("Replay requires a new empty project")
        with Path(path).open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        if data.get("schema") != "sam31-colab-project" or data.get("version") != PROJECT_VERSION:
            raise ValueError("Unsupported project schema or version")
        naming_policy = data.get("naming_policy", "legacy")
        if not isinstance(naming_policy, str) or naming_policy not in (NAMING_POLICY, "legacy"):
            raise ValueError("Unsupported project naming policy")
        saved_environment = data.get("environment", {})
        for key in ("sam_revision", "checkpoint_revision"):
            if key in saved_environment and key in self.environment and saved_environment[key] != self.environment[key]:
                raise ValueError(f"Model environment differs from the saved project: {key}")
        if data.get("fingerprint") != self.sequence.fingerprint():
            raise ValueError("Input fingerprint or image settings differ from the saved project")
        expected_prepared = {key: self.prepared[key] for key in ("low", "high", "frame_count", "height", "width")}
        if data.get("prepared") != expected_prepared or data.get("max_objects") != self.max_objects:
            raise ValueError("Prepared intensity settings or object limit differ from the saved project")
        self._naming_policy = naming_policy
        try:
            for event in data["events"]:
                kind = event["type"]
                if kind == "concept":
                    self.add_concept(event["frame"], event["text"], event["boxes"], event["box_labels"])
                elif kind == "redetect":
                    self.update_concept(event["frame"], event["text"], event["boxes"], event["box_labels"], mode="replace")
                elif kind == "points":
                    if event["new_object"]:
                        if event["object_id"] != self._next_app_id or event["raw_id"] != self._unused_raw_id():
                            raise ValueError("New-object identity differs from the saved project")
                    elif self._object(event["object_id"])["raw_id"] != event["raw_id"]:
                        raise ValueError("Point target identity differs from the saved project")
                    self.add_points(event["frame"], None if event["new_object"] else event["object_id"],
                                    event["points"], event["labels"], event["name"])
                elif kind == "remove":
                    self.remove_object(event["object_id"], event["frame"])
                elif kind == "rename":
                    self.rename_object(event["object_id"], event["name"])
                elif kind == "set_name_prefix":
                    self.set_name_prefix(event["prefix"], rename_existing=event["rename_existing"])
                elif kind == "propagate":
                    self.propagate(event["start_frame"], event["direction"])
                else:
                    raise ValueError("Unknown project event")
                if self.events[-1] != event:
                    raise ValueError("Replay object IDs or masks differ; refusing to apply subsequent prompts to uncertain identities")
            if {str(key): value for key, value in self.objects.items()} != data["objects"]:
                raise ValueError("Replayed object metadata differs from the saved project")
        except BaseException:
            self._invalidate()
            self.rebuild_required = True
            raise
        return self

    load_project = replay_project

    def _safe_output(self, destination):
        target = destination.resolve()
        source = Path(self.sequence.settings.input_path).expanduser().resolve()
        cache = Path(self.prepared["resource_path"]).expanduser().resolve()
        protected = [(source, source.is_dir()), (cache, True)]
        protected.extend((Path(path).resolve(), False) for path in getattr(self.sequence, "_files", []))
        for item, directory in protected:
            if target == item or target in item.parents or (directory and item in target.parents):
                raise ValueError("Output must not overlap the original input or prepared-frame cache")

    def export(self, output_dir, output_mode="mask"):
        """Export segmentation images and one JSON containing prompts and reproducibility settings."""
        self._ready()
        if self.dirty or set(self.results) != set(range(self.sequence.shape[0])):
            raise RuntimeError("Run complete propagation after all corrections before exporting")
        if output_mode not in ("mask", "overlay", "both"):
            raise ValueError('output_mode must be "mask", "overlay", or "both"')
        save_masks = output_mode in ("mask", "both")
        save_overlays = output_mode in ("overlay", "both")
        destination = Path(output_dir).expanduser()
        self._safe_output(destination)
        if destination.exists():
            if not destination.is_dir() or any(destination.iterdir()):
                raise FileExistsError("Choose an empty or new export directory; existing files will not be overwritten")
        else:
            destination.mkdir(parents=True, exist_ok=False)
        if save_masks:
            (destination / "masks").mkdir()
        if save_overlays:
            (destination / "overlays").mkdir()
        active_ids = sorted(key for key, item in self.objects.items() if not item["excluded"])
        mask_folders = {object_id: Path("masks") / mask_folder_name(object_id, self.objects[object_id])
                        for object_id in active_ids} if save_masks else {}
        if save_masks:
            for object_id in active_ids:
                (destination / mask_folders[object_id]).mkdir()
        zero_mask = np.zeros(self.sequence.shape[1:], dtype=bool)
        for local_frame, masks in sorted(self.results.items()):
            source_frame = self.sequence.frame_indices[local_frame]
            full_masks = {object_id: masks.get(object_id, zero_mask) for object_id in active_ids}
            if save_overlays:
                frame_labels = (labels_from_masks(active_ids, list(full_masks.values())) if active_ids
                                else np.zeros(self.sequence.shape[1:], dtype=np.uint16))
                overlay = self.sequence.preview(source_frame, low=self.prepared["low"], high=self.prepared["high"])
            for object_id, mask in full_masks.items():
                if save_masks:
                    tifffile.imwrite(destination / mask_folders[object_id] / f"frame_{local_frame:06d}.tif",
                                     mask.astype(np.uint8) * 255, metadata={"axes": "YX"})
                if save_overlays:
                    visible = frame_labels == object_id
                    overlay[visible] = np.rint(0.55 * overlay[visible] + 0.45 * np.array(self.objects[object_id]["color"])).astype(np.uint8)
            if save_overlays:
                overlay_image = Image.fromarray(overlay)
                for object_id, mask in full_masks.items():
                    draw_object_label(overlay_image, mask, object_id, self.objects[object_id])
                overlay_image.save(destination / "overlays" / f"frame_{local_frame:06d}.png")
        self.save_project(destination / "project.json", export_info={
            "complete": True, "output_mode": output_mode,
            "source_frame_indices": list(self.sequence.frame_indices),
            "frame_filename": "frame_{local_frame_index:06d}",
            "mask_values": {"background": 0, "foreground": 255},
            "overlay_opacity": 0.45, "overlay_object_ids": True,
            "mask_folders": {str(key): value.as_posix() for key, value in mask_folders.items()},
            "overlay_label_format": LABEL_FORMAT,
        })
        return destination

    def close(self):
        if not self._closed:
            self.predictor.handle_request({"type": "close_session", "session_id": self.session_id})
            self._closed = True
