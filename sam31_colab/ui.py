"""Interactive annotation controls for a selected image tracking project."""

from __future__ import annotations

from html import escape
import math

import ipywidgets as widgets
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
import numpy as np
from PIL import Image

from .labels import draw_object_label


class TrackingUI:
    """Display ``widget`` with an interactive Matplotlib backend in a notebook.

    Coordinates use original image pixels. Contrast affects display only. Close
    this UI before replacing its project; close the project separately.
    """

    def __init__(self, project):
        self.project = project
        self._pending = []
        self._busy = False
        self._closed = False
        self._updating = False
        self._displayed_frame = None
        self._observers = []
        self._buttons = []
        count, self._height, self._width = project.sequence.shape
        dtype = np.dtype(project.sequence.metadata["dtype"])
        maximum = float(np.iinfo(dtype).max)
        initial_low = float(project.prepared.get("low", 0))
        initial_high = float(project.prepared.get("high", maximum))
        self.frame = widgets.IntSlider(min=0, max=count - 1, value=0,
                                       description="Frame", continuous_update=False)
        self.frame_label = widgets.Label()
        self.object = widgets.Dropdown(description="Object", options=[("New object", None)])
        self.tool = widgets.ToggleButtons(
            options=[("+ Point", "point_positive"), ("− Point", "point_negative"),
                     ("+ Box", "box_positive"), ("− Box", "box_negative")],
            value="point_positive", description="Tool")
        saved = project.concept_prompt
        self.concept = widgets.Text(value=saved["text"] if saved is not None else "",
                                    description="Concept", placeholder="e.g. dog")
        self.box_update = widgets.Dropdown(description="Box update",
                                           options=[("Add to saved", "add"), ("Replace saved", "replace")])
        self.saved_label = widgets.Label()
        self.apply_concept = widgets.Button(description="Apply concept / boxes", button_style="primary")
        self.apply_points = widgets.Button(description="Apply points", button_style="primary")
        self.undo = widgets.Button(description="Undo pending")
        self.clear = widgets.Button(description="Clear pending")
        self.object_name = widgets.Text(description="Name", placeholder="Object name")
        self.name_prefix = widgets.Text(value=project.name_prefix, description="Name prefix",
                                        placeholder="Blank: concept or Object",
                                        style={"description_width": "initial"})
        self.apply_prefix = widgets.Button(description="Apply prefix to all")
        self._displayed_object_name = ""
        self.save_metadata = widgets.Button(description="Save name")
        self.exclude = widgets.Button(description="Exclude object", button_style="warning")
        self.direction = widgets.Dropdown(description="Direction", options=["both", "forward", "backward"])
        self.track = widgets.Button(description="Track", button_style="success", icon="play")
        self.progress = widgets.IntProgress(min=0, max=count, value=0, description="Tracked")
        self.contrast = widgets.FloatRangeSlider(
            min=0, max=maximum, value=(max(0, min(maximum, initial_low)),
                                      max(0, min(maximum, initial_high))),
            step=1, description="Display range", continuous_update=False,
            readout_format=".0f", style={"description_width": "initial"})
        self.alpha = widgets.FloatSlider(min=0, max=1, step=0.05, value=0.4,
                                          description="Mask opacity", continuous_update=False,
                                          style={"description_width": "initial"})
        self.status = widgets.Label(value="Choose a tool and annotate the image.")
        self.state_label = widgets.Label()
        self.pending_label = widgets.Label()
        self.legend = widgets.HTML()
        with plt.ioff():
            self.figure, self.axes = plt.subplots(figsize=(9, 6), constrained_layout=True)
        canvas = self.figure.canvas
        self._plot_is_widget = isinstance(canvas, widgets.Widget)
        if self._plot_is_widget:
            self._plot_widget = canvas
            if hasattr(canvas, "toolbar_visible"):
                canvas.toolbar_visible = True
            if hasattr(canvas, "header_visible"):
                canvas.header_visible = False
            if hasattr(canvas, "resizable"):
                canvas.resizable = True
        else:
            self._plot_widget = widgets.Output()
        self._selector = None
        self._click_connection = canvas.mpl_connect("button_press_event", self._on_click)
        self._controls = [self.frame, self.object, self.tool, self.concept, self.box_update, self.apply_concept,
                          self.apply_points, self.undo, self.clear, self.object_name,
                          self.save_metadata, self.name_prefix, self.apply_prefix, self.exclude, self.direction,
                          self.track, self.contrast, self.alpha]
        self._mutations = [self.tool, self.concept, self.box_update, self.apply_concept, self.apply_points,
                           self.undo, self.clear, self.object_name,
                           self.save_metadata, self.name_prefix, self.apply_prefix, self.exclude, self.track, self.direction]
        self.widget = widgets.VBox([
            widgets.HTML("<b>SAM 3.1 · Interactive tracking</b>"),
            widgets.Label("Frames and source indices are zero-based. Use the toolbar to zoom or pan."),
            self._row(self.frame, self.frame_label),
            self._row(self.contrast, self.alpha), self._plot_widget,
            self._row(self.object, self.tool), self.pending_label,
            self._row(self.concept, self.box_update, self.apply_concept, self.apply_points),
            self.saved_label,
            widgets.Label("Saved boxes are dashed. Reapplying detection replaces objects/IDs, discards point "
                          "corrections and previous tracking; redo Track before export."),
            self._row(self.undo, self.clear),
            self._row(self.name_prefix, self.apply_prefix),
            widgets.Label("Names use prefix + object ID. Applying to all replaces individual names."),
            self._row(self.object_name, self.save_metadata, self.exclude),
            self.legend, self._row(self.direction, self.track, self.progress),
            self.state_label, self.status,
            widgets.Label("Boxes provide detection examples; points target the selected object. "
                          "Pending input clears on frame, object, or point/box tool changes."),
        ], layout=widgets.Layout(width="100%", max_width="1000px"))
        self._observe(self.frame, self._on_frame)
        self._observe(self.object, self._on_object)
        self._observe(self.tool, self._on_tool)
        self._observe(self.contrast, self._on_display)
        self._observe(self.alpha, self._on_display)
        for button, callback in [(self.apply_concept, self._apply_concept),
                                 (self.apply_points, self._apply_points),
                                 (self.undo, self._undo), (self.clear, self._clear),
                                 (self.save_metadata, self._save_metadata),
                                 (self.apply_prefix, self._apply_prefix),
                                 (self.exclude, self._exclude), (self.track, self._track)]:
            button.on_click(callback)
            self._buttons.append((button, callback))
        self.refresh()

    @staticmethod
    def _row(*children):
        return widgets.Box(children, layout=widgets.Layout(
            display="flex", flex_flow="row wrap", align_items="center", width="100%"))

    def _observe(self, widget, callback):
        widget.observe(callback, names="value")
        self._observers.append((widget, callback))

    def _message(self, text):
        self.status.value = str(text).replace("\n", " ")[:400]

    def _can_annotate(self):
        return not (self._closed or self._busy or self.project.rebuild_required)

    def _navigation_active(self):
        toolbar = getattr(self.figure.canvas, "toolbar", None)
        return bool(getattr(toolbar, "mode", "") or self.axes.get_navigate_mode())

    def _on_click(self, event):
        if not self._can_annotate() or self._navigation_active():
            return
        if not self.tool.value.startswith("point") or event.inaxes is not self.axes or event.button != 1:
            return
        if not self._valid_point(event.xdata, event.ydata):
            return
        self._pending.append({"kind": "point", "coordinates": [float(event.xdata), float(event.ydata)],
                              "label": int(self.tool.value.endswith("positive"))})
        self._message("Point added. Apply points to submit the pending annotations.")
        self._redraw_safely()

    def _valid_point(self, x, y):
        return x is not None and y is not None and math.isfinite(x) and math.isfinite(y) and (
            0 <= x < self._width and 0 <= y < self._height)

    def _on_rectangle(self, press, release):
        if not self._can_annotate() or self._navigation_active() or not self.tool.value.startswith("box"):
            return
        values = (press.xdata, press.ydata, release.xdata, release.ydata)
        if any(value is None or not math.isfinite(value) for value in values):
            return
        if any(not 0 <= value <= bound for value, bound in zip(
                values, (self._width, self._height, self._width, self._height))):
            return
        left, right = sorted((float(press.xdata), float(release.xdata)))
        top, bottom = sorted((float(press.ydata), float(release.ydata)))
        if right <= left or bottom <= top:
            return
        self._pending.append({"kind": "box", "coordinates": [left / self._width, top / self._height,
                              (right - left) / self._width, (bottom - top) / self._height],
                              "label": int(self.tool.value.endswith("positive"))})
        self._message("Box added. Apply concept / boxes to update the saved examples and redetect.")
        self._redraw_safely()

    def _clear_pending(self, reason=None):
        had_pending = bool(self._pending)
        self._pending.clear()
        if had_pending and reason:
            self._message(f"Pending annotations cleared: {reason}.")

    def _on_frame(self, change):
        if self._updating or self._closed or self._busy:
            return
        self._clear_pending("frame changed")
        self.refresh(sync_prefix=False)

    def _on_object(self, change):
        if self._updating or self._closed or self._busy:
            return
        self._clear_pending("object changed")
        self.refresh(sync_prefix=False)

    def _on_tool(self, change):
        if self._updating or self._closed or self._busy:
            return
        if change["old"].split("_")[0] != change["new"].split("_")[0]:
            self._clear_pending("point/box tool changed")
        self.refresh(sync_prefix=False)

    def _on_display(self, change):
        if not (self._updating or self._closed or self._busy):
            self._redraw_safely()

    def _run(self, action, success):
        if not self._can_annotate():
            return
        try:
            action()
        except Exception as error:
            self._message(f"Error: {type(error).__name__}: {error}")
        else:
            self._message(success)
        finally:
            self.refresh(sync_prefix=False)

    def _apply_concept(self, button=None):
        def apply():
            if any(entry["kind"] != "box" for entry in self._pending):
                raise ValueError("Apply or clear pending points before applying a concept")
            boxes = [entry for entry in self._pending if entry["kind"] == "box"]
            self.project.update_concept(self.frame.value, self.concept.value,
                                        boxes=[entry["coordinates"] for entry in boxes],
                                        box_labels=[entry["label"] for entry in boxes],
                                        mode=self.box_update.value, name_prefix=self.name_prefix.value)
            self._clear_pending()
            self.object.value = None
            self.object_name.value = ""
        self._run(apply, "Detection applied. Review the new objects and Track before exporting.")

    def _apply_points(self, button=None):
        def apply():
            if any(entry["kind"] != "point" for entry in self._pending):
                raise ValueError("Apply or clear pending boxes before applying points")
            points = [entry for entry in self._pending if entry["kind"] == "point"]
            previous_ids = set(self.project.objects)
            selected = self.object.value
            self._save_prefix()
            name = self.object_name.value if selected is None or self.object_name.value != self._displayed_object_name else ""
            self.project.add_points(self.frame.value, selected,
                                    [entry["coordinates"] for entry in points],
                                    [entry["label"] for entry in points], name=name)
            self._clear_pending()
            if selected is None:
                added = sorted(set(self.project.objects) - previous_ids)
                if added:
                    self._sync_objects(added[0])
        self._run(apply, "Points applied. Review masks and track when ready.")

    def _undo(self, button=None):
        if self._can_annotate() and self._pending:
            self._pending.pop()
            self._message("Last pending annotation removed.")
            self._redraw_safely()

    def _clear(self, button=None):
        if self._can_annotate():
            self._clear_pending()
            self._message("Pending annotations cleared.")
            self._redraw_safely()

    def _save_metadata(self, button=None):
        def save():
            if self.object.value is None:
                raise ValueError("Select an existing object first")
            self.project.rename_object(self.object.value, self.object_name.value)
        self._run(save, "Object name saved.")

    def _save_prefix(self):
        prefix = self.name_prefix.value.strip()
        if prefix != self.project.name_prefix:
            self.project.set_name_prefix(prefix)

    def _apply_prefix(self, button=None):
        self._run(lambda: self.project.set_name_prefix(self.name_prefix.value, rename_existing=True),
                  "Prefix applied to all active objects and future objects.")

    def _exclude(self, button=None):
        def exclude():
            if self.object.value is None:
                raise ValueError("Select an existing object first")
            self.project.remove_object(self.object.value, frame=self.frame.value)
            self._clear_pending()
        self._run(exclude, "Object excluded. Track again to update the result snapshot.")

    def _track(self, button=None):
        if not self._can_annotate():
            return
        if self._pending:
            self._message("Apply or clear pending annotations before tracking.")
            return
        self._busy = True
        self.progress.value = 0
        self.progress.bar_style = "info"
        self._set_controls()
        self._message("Tracking…")
        try:
            self._save_prefix()
            self.project.propagate(start_frame=self.frame.value, direction=self.direction.value,
                                   progress=self._update_progress)
        except Exception as error:
            self.progress.bar_style = "danger"
            self._message(f"Error: {type(error).__name__}: {error}")
        else:
            self.progress.bar_style = "warning" if self.project.dirty else "success"
            self._message("Tracking finished. " + (
                "Partial coverage; run both directions for complete export." if self.project.dirty
                else "All selected frames are ready for export."))
        finally:
            self._busy = False
            self.refresh(sync_prefix=False)

    def _update_progress(self, completed, total):
        self.progress.max = max(1, int(total))
        self.progress.value = min(int(completed), self.progress.max)
        self.progress.send_state()
        self.figure.canvas.flush_events()

    def _sync_objects(self, selected=None):
        previous = self.object.value if selected is None else selected
        options = [("New object", None)] + [
            (f"{object_id}: {item['name']}", object_id)
            for object_id, item in sorted(self.project.objects.items()) if not item["excluded"]]
        self._updating = True
        try:
            self.object.options = options
            self.object.value = previous if previous in [value for _, value in options] else None
        finally:
            self._updating = False
        if self.object.value != previous:
            self._clear_pending("selected object is unavailable")

    def _set_controls(self):
        for control in self._controls:
            control.disabled = self._closed or self._busy
        if self.project.rebuild_required:
            for control in self._mutations:
                control.disabled = True
        if self.object.value is None:
            self.save_metadata.disabled = True
            self.exclude.disabled = True
        if self._selector is not None:
            self._selector.set_active(self._can_annotate() and self.tool.value.startswith("box"))

    def refresh(self, *, sync_prefix=True):
        """Read project state and redraw without changing inference intensity limits."""
        if self._closed:
            return
        if sync_prefix:
            self.name_prefix.value = self.project.name_prefix
            saved = self.project.concept_prompt
            if saved is not None:
                self.concept.value = saved["text"]
        self._sync_objects()
        item = self.project.objects.get(self.object.value)
        if item is not None:
            self.object_name.value = item["name"]
        else:
            self.object_name.value = ""
        self._displayed_object_name = self.object_name.value
        source_index = self.project.sequence.frame_indices[self.frame.value]
        self.frame_label.value = f"Local {self.frame.value} / {self.frame.max} · source {source_index}"
        saved = self.project.concept_prompt
        if saved is None:
            self.saved_label.value = "Saved examples: 0 · no reference frame"
        else:
            positive = sum(saved["box_labels"])
            self.saved_label.value = (f"Saved examples: {len(saved['boxes'])} ({positive} target / "
                                      f"{len(saved['boxes']) - positive} background) · reference local frame {saved['frame']}")
        if self.project.rebuild_required:
            self.state_label.value = "Session failed: save successful events and replay in a new project."
        elif self.project.dirty:
            self.state_label.value = "Results need full tracking. Current-frame masks are prompt / partial previews."
        else:
            self.state_label.value = "Complete result snapshot: all selected frames tracked."
        entries = []
        for object_id, object_data in sorted(self.project.objects.items()):
            color = self._color(object_data)
            rgb = ",".join(str(round(channel * 255)) for channel in color)
            selected = " <b>(selected)</b>" if self.object.value == object_id else ""
            excluded = " (excluded)" if object_data["excluded"] else ""
            entries.append(f'<span style="color:rgb({rgb})">●</span> '
                           f'{object_id}: {escape(str(object_data["name"]))}{selected}{excluded}')
        self.legend.value = "<br>".join(entries) or "No objects yet."
        self._set_controls()
        self._redraw_safely()

    @staticmethod
    def _color(item):
        return np.clip(np.asarray(item["color"], dtype=float) / 255, 0, 1)

    def _visible_masks(self):
        if self.project.rebuild_required:
            return {}
        if self.project.current_frame_index == self.frame.value:
            return self.project.current_frame_outputs or {}
        if not self.project.dirty:
            return self.project.results.get(self.frame.value) or {}
        return {}

    def _redraw_safely(self):
        try:
            self._draw()
        except Exception as error:
            self._displayed_frame = None
            self.axes.clear()
            self.figure.canvas.draw_idle()
            self._message(f"Display error: {type(error).__name__}: {error}")

    def _draw(self):
        same_frame = self._displayed_frame == self.frame.value
        previous_xlim, previous_ylim = self.axes.get_xlim(), self.axes.get_ylim()
        if self._selector is not None:
            self._selector.disconnect_events()
        self.axes.clear()
        self.axes.set_xlabel("X (pixels)")
        self.axes.set_ylabel("Y (pixels)")
        source_index = self.project.sequence.frame_indices[self.frame.value]
        low, high = self.contrast.value
        frame_image = self.project.sequence.preview(source_index, low=low, high=high)
        extent = (-0.5, self._width - 0.5, self._height - 0.5, -0.5)
        if frame_image is not None:
            self.axes.imshow(frame_image, origin="upper", extent=extent, interpolation="nearest")
            labels = Image.new("RGBA", (self._width, self._height))
            for object_id, mask in self._visible_masks().items():
                item = self.project.objects.get(object_id)
                if item is None or item["excluded"]:
                    continue
                mask = np.asarray(mask, dtype=bool)
                if mask.shape != (self._height, self._width):
                    raise ValueError("Mask dimensions do not match the image")
                rgba = np.zeros((*mask.shape, 4), dtype=float)
                rgba[mask, :3] = self._color(item)
                rgba[mask, 3] = self.alpha.value
                self.axes.imshow(rgba, origin="upper", extent=extent, interpolation="nearest")
                selected = object_id == self.object.value
                draw_object_label(labels, mask, object_id, item, selected=selected)
                if selected and mask.any() and not mask.all():
                    self.axes.contour(mask, levels=[0.5], colors=["white"], linewidths=1.2)
            self.axes.imshow(np.asarray(labels), origin="upper", extent=extent, interpolation="nearest")
        saved = self.project.concept_prompt
        if saved is not None and saved["frame"] == self.frame.value:
            for box, label in zip(saved["boxes"], saved["box_labels"]):
                left, top, width, height = box
                self.axes.add_patch(Rectangle((left * self._width, top * self._height),
                    width * self._width, height * self._height, fill=False,
                    edgecolor="#39e68c" if label else "#ff5e72", linewidth=2, linestyle="dashed"))
        for entry in self._pending:
            positive = bool(entry["label"])
            color = "#39e68c" if positive else "#ff5e72"
            coordinates = entry["coordinates"]
            if entry["kind"] == "point":
                self.axes.plot(*coordinates, marker="+" if positive else "x", color=color,
                               markersize=12, markeredgewidth=2, linestyle="none")
            else:
                left, top, width, height = coordinates
                self.axes.add_patch(Rectangle((left * self._width, top * self._height),
                    width * self._width, height * self._height, fill=False, edgecolor=color,
                    linewidth=2, linestyle="solid" if positive else "dotted"))
        self.axes.set_xlim(previous_xlim if same_frame else (-0.5, self._width - 0.5))
        self.axes.set_ylim(previous_ylim if same_frame else (self._height - 0.5, -0.5))
        self.axes.set_title(f"Local frame {self.frame.value} · source {source_index}")
        points = sum(entry["kind"] == "point" for entry in self._pending)
        self.pending_label.value = f"Pending: {points} points · {len(self._pending) - points} boxes"
        self._selector = RectangleSelector(self.axes, self._on_rectangle, useblit=False,
                                            button=[1], minspanx=0, minspany=0,
                                            spancoords="data", interactive=False)
        self._selector.set_active(self._can_annotate() and self.tool.value.startswith("box"))
        self.figure.canvas.draw_idle()
        self._displayed_frame = self.frame.value
        if not self._plot_is_widget:
            from IPython.display import clear_output, display
            with self._plot_widget:
                clear_output(wait=True)
                display(self.figure)

    def close(self):
        """Disconnect annotation callbacks and close the figure, leaving the project open."""
        if self._closed:
            return
        self._closed = True
        for widget, callback in self._observers:
            widget.unobserve(callback, names="value")
        for button, callback in self._buttons:
            button.on_click(callback, remove=True)
        self.figure.canvas.mpl_disconnect(self._click_connection)
        if self._selector is not None:
            self._selector.disconnect_events()
        self._set_controls()
        plt.close(self.figure)
