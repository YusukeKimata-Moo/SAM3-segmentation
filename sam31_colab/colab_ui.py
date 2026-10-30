"""Display annotation controls through Colab HTML and Python callbacks."""

import base64
from io import BytesIO
import json
import math
from uuid import uuid4

import numpy as np
from PIL import Image

from .labels import draw_object_label


class ColabTrackingUI:
    """Use show() in Colab; close() leaves the tracking project and its prompts intact."""

    def __init__(self, project, initial_pending=None):
        self.project = project
        self.frame = 0
        self.object_id = None
        self.low, self.high = float(project.prepared["low"]), float(project.prepared["high"])
        self.alpha = 0.4
        saved = project.concept_prompt
        self.concept = saved["text"] if saved is not None else ""
        self.box_update = "add"
        self.name = ""
        self.name_prefix = project.name_prefix
        self.status = "Choose a frame, then enter a concept or add points."
        self._closed = False
        self._busy = False
        self._callback_name = None
        self._output = None
        self.initial_pending = list(initial_pending or [])
        self._root_id = "sam31_" + uuid4().hex

    def _masks(self):
        if self.project.rebuild_required:
            return {}
        if self.project.current_frame_index == self.frame:
            return self.project.current_frame_outputs or {}
        return {} if self.project.dirty else self.project.results.get(self.frame, {})

    def state(self):
        source_index = self.project.sequence.frame_indices[self.frame]
        pixels = self.project.sequence.preview(source_index, low=self.low, high=self.high).copy()
        height, width = self.project.sequence.shape[1:]
        owner = np.zeros((height, width), np.int64)
        shown = {}
        for object_id, mask in sorted(self._masks().items()):
            item = self.project.objects.get(object_id)
            if item is None or item["excluded"]:
                continue
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != owner.shape:
                raise ValueError("Mask dimensions do not match the image")
            visible = mask & (owner == 0)
            owner[visible] = object_id
            pixels[visible] = np.rint((1 - self.alpha) * pixels[visible] + self.alpha * np.asarray(item["color"])).astype(np.uint8)
            shown[object_id] = mask
        image = Image.fromarray(pixels)
        for object_id, mask in shown.items():
            draw_object_label(image, mask, object_id, self.project.objects[object_id],
                              selected=object_id == self.object_id)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        active = [{"id": key, "name": item["name"], "color": item["color"]}
                  for key, item in sorted(self.project.objects.items()) if not item["excluded"]]
        if self.object_id is not None and self.object_id not in [item["id"] for item in active]:
            self.object_id = None
        if self.object_id is not None:
            self.name = self.project.objects[self.object_id]["name"]
        return {"frame": self.frame, "source_index": source_index,
                "frame_count": self.project.sequence.shape[0], "width": width, "height": height,
                "object_id": self.object_id, "objects": active, "name": self.name,
                "concept": self.concept, "name_prefix": self.name_prefix,
                "effective_name_prefix": self.project.effective_name_prefix,
                "box_update": self.box_update, "saved_examples": self.project.concept_prompt,
                "low": self.low, "high": self.high, "alpha": self.alpha,
                "maximum": int(np.iinfo(self.project.sequence.metadata["dtype"]).max),
                "image": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
                "tracked_frames": len(self.project.results),
                "dirty": self.project.dirty, "rebuild_required": self.project.rebuild_required,
                "closed": self._closed, "status": self.status}

    def handle_request(self, action, payload=None):
        if self._closed:
            raise RuntimeError("This panel is closed. Use the latest displayed panel.")
        if self._busy:
            raise RuntimeError("Wait for the current operation to finish.")
        payload = payload or {}
        self._busy = True
        try:
            frame = payload.get("frame", self.frame)
            if isinstance(frame, bool) or not isinstance(frame, int) or not 0 <= frame < self.project.sequence.shape[0]:
                raise ValueError("Frame is outside the selected sequence")
            selected = payload.get("object_id", self.object_id)
            if selected is not None and (isinstance(selected, bool) or not isinstance(selected, int)
                                         or selected not in self.project.objects or self.project.objects[selected]["excluded"]):
                raise ValueError("Select an active object or New object")
            low, high, alpha = [float(payload.get(key, getattr(self, key))) for key in ("low", "high", "alpha")]
            maximum = np.iinfo(self.project.sequence.metadata["dtype"]).max
            if not all(math.isfinite(value) for value in (low, high, alpha)) or not 0 <= low <= high <= maximum or not 0 <= alpha <= 1:
                raise ValueError("Check display limits and opacity")
            if action not in {"frame", "object", "display", "concept", "points", "name", "prefix", "exclude", "track"}:
                raise ValueError("Unknown panel operation")
            if action not in {"frame", "object", "display"} and self.project.rebuild_required:
                raise RuntimeError("Rebuild the project before adding prompts or tracking")
            prefix = payload.get("name_prefix", self.name_prefix)
            if not isinstance(prefix, str):
                raise ValueError("Name prefix must be a string")
            box_update = payload.get("box_update", self.box_update)
            if box_update not in ("add", "replace"):
                raise ValueError("Box update must be add or replace")
            name_changed = payload.get("name_changed", payload.get("name", self.name) != self.name)
            self.frame, self.object_id = frame, selected
            self.low, self.high, self.alpha = low, high, alpha
            self.name_prefix = prefix.strip()
            self.box_update = box_update
            if "concept" in payload:
                self.concept = str(payload["concept"])
            if "name" in payload:
                self.name = str(payload["name"])
            pending = payload.get("pending", [])
            if action == "concept":
                if any(item["kind"] != "box" for item in pending):
                    raise ValueError("Apply or clear pending points first")
                self.project.update_concept(frame, self.concept,
                                            boxes=[item["coordinates"] for item in pending],
                                            box_labels=[item["label"] for item in pending],
                                            mode=self.box_update, name_prefix=self.name_prefix)
                self.object_id, self.name = None, ""
                self.status = "Detection applied. Review the new objects and Track before exporting."
            elif action == "points":
                if any(item["kind"] != "point" for item in pending):
                    raise ValueError("Apply or clear pending boxes first")
                self._save_prefix()
                previous_ids = set(self.project.objects)
                returned_masks = self.project.add_points(frame, selected, [item["coordinates"] for item in pending],
                                                         [item["label"] for item in pending],
                                                         name=self.name if selected is None or name_changed else "")
                added = sorted(set(self.project.objects) - previous_ids)
                if selected is None and added:
                    self.object_id = added[0]
                if self.object_id not in returned_masks or not np.any(returned_masks[self.object_id]):
                    self.status = "Points applied, but no visible mask was returned for this object. Check prompt coordinates and inference before tracking."
                else:
                    self.status = "Points applied. Review the masks."
            elif action == "name":
                if selected is None:
                    raise ValueError("Select an existing object first")
                self.project.rename_object(selected, self.name)
                self.status = "Object name saved."
            elif action == "prefix":
                self.project.set_name_prefix(self.name_prefix, rename_existing=True)
                self.status = "Prefix applied to all active objects and future objects."
            elif action == "exclude":
                if selected is None:
                    raise ValueError("Select an existing object first")
                self.project.remove_object(selected, frame=frame)
                self.object_id, self.name = None, ""
                self.status = "Object excluded. Track again to update results."
            elif action == "track":
                if pending:
                    raise ValueError("Apply or clear pending annotations before tracking")
                self._save_prefix()
                self.project.propagate(start_frame=frame, direction=payload.get("direction", "both"),
                                       progress=lambda completed, total: None)
                self.status = ("Partial coverage. Track both directions before exporting." if self.project.dirty
                               else "Tracking finished. All selected frames are ready for export.")
            elif action in {"frame", "object"}:
                if action == "object" and selected is None:
                    self.name = ""
                self.status = "View updated. Pending annotations were cleared."
            else:
                self.status = "Display updated; model input settings are unchanged."
            return {"ok": True, "state": self.state()}
        except Exception as error:
            self.status = f"{type(error).__name__}: {error}"
            return {"ok": False, "state": self.state()}
        finally:
            self._busy = False

    def _save_prefix(self):
        if self.name_prefix != self.project.name_prefix:
            self.project.set_name_prefix(self.name_prefix)

    def html(self):
        return '''<div id="ROOT" style="font:14px Arial,sans-serif;color:#202124;background:#fff;padding:14px;border:1px solid #ccc;max-width:1100px">
<style>#ROOT label{display:inline-flex;align-items:center;gap:6px;margin:5px}#ROOT button,#ROOT select,#ROOT input{font:inherit;padding:5px}#ROOT button{margin:4px;cursor:pointer}#ROOT canvas{display:block;touch-action:none;cursor:crosshair}#ROOT .row{display:flex;flex-wrap:wrap;align-items:center;gap:6px}#ROOT .viewport{max-height:650px;overflow:auto;border:1px solid #aaa;background:#222}#ROOT .status{white-space:pre-wrap;padding:8px;border:1px solid #ddd;margin-top:8px}#ROOT button:disabled{cursor:wait;opacity:.5}</style>
<b>SAM 3.1 · Annotation and tracking</b>
<p>Frames are zero-based. Click for points; drag for boxes. Zoom with the scale control and scroll to move around.</p>
<div class="row"><label>Frame <input data-role="frame" type="range" min="0" value="0"></label><label>Frame number <input data-role="frame_number" type="number" min="0" value="0" style="width:80px"></label><span data-role="source"></span></div>
<div class="row"><label>Object <select data-role="object"></select></label><label>Tool <select data-role="tool"><option value="point_positive">+ Point (target)</option><option value="point_negative">− Point (background)</option><option value="box_positive">+ Box</option><option value="box_negative">− Box</option></select></label><button data-action="undo">Undo pending</button><button data-action="clear">Clear pending</button><span data-role="pending"></span></div>
<div class="row"><label>Concept <input data-role="concept" type="text" placeholder="e.g. dog"></label><label>Box update <select data-role="box_update" data-mutation><option value="add">Add to saved</option><option value="replace">Replace saved</option></select></label><button data-action="concept" data-mutation>Apply concept / boxes</button><button data-action="points" data-mutation>Apply points</button></div>
<div class="row"><span data-role="saved"></span><span>Saved boxes are dashed. Reapplying detection replaces objects/IDs, discards point corrections and previous tracking; redo Track before export.</span></div>
<div class="row"><label>Name prefix <input data-role="name_prefix" data-mutation type="text" placeholder="Blank: concept or Object"></label><button data-action="prefix" data-mutation>Apply prefix to all</button><span>Names use prefix + object ID. Applying to all replaces individual names.</span></div>
<div class="row"><label>Name <input data-role="name" data-mutation type="text" placeholder="Optional individual name"></label><button data-action="name" data-mutation>Save name</button><button data-action="exclude" data-mutation>Exclude object</button></div>
<div class="row"><label>Display low <input data-role="low" type="number" min="0" style="width:95px"></label><label>Display high <input data-role="high" type="number" min="0" style="width:95px"></label><label>Mask opacity <input data-role="alpha" type="range" min="0" max="1" step="0.05"></label><button data-action="display">Update display</button><label>Zoom <input data-role="zoom" type="range" min="0.1" max="4" step="0.1" value="1"></label><span data-role="zoom_value">100%</span></div>
<div class="viewport"><canvas data-role="canvas"></canvas></div>
<div data-role="legend"></div>
<div class="row"><label>Direction <select data-role="direction"><option value="both">both</option><option value="forward">forward</option><option value="backward">backward</option></select></label><button data-action="track" data-mutation>Track</button><label>Tracked <progress data-role="progress" value="0" max="1"></progress></label><span data-role="coverage"></span></div>
<div class="status" data-role="status" aria-live="polite"></div>
</div>'''.replace("ROOT", self._root_id)

    def script(self, state=None):
        initial = {"state": state or self.state(), "pending": self.initial_pending,
                   "root": self._root_id, "callback": self._callback_name}
        encoded = json.dumps(initial, ensure_ascii=True).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
        return "(() => { const initial = " + encoded + ";\n" + _SCRIPT + "\n})()"

    def _callback(self, action, payload=None):
        from IPython.display import JSON
        return JSON(self.handle_request(action, payload))

    def show(self):
        if self._closed:
            raise RuntimeError("This panel is closed")
        from google.colab import output
        from IPython.display import HTML, display
        if self._callback_name is not None:
            self._output.unregister_callback(self._callback_name)
        self._output = output
        self._root_id = "sam31_" + uuid4().hex
        self._callback_name = "sam31.panel." + uuid4().hex
        output.register_callback(self._callback_name, self._callback)
        display(HTML(self.html()))
        output.eval_js(self.script(), timeout_sec=30)
        self.initial_pending = []

    def refresh(self):
        """Display a fresh panel in the current cell; use the most recently displayed panel."""
        self.name_prefix = self.project.name_prefix
        saved = self.project.concept_prompt
        if saved is not None:
            self.concept = saved["text"]
        self.show()

    def close(self):
        if not self._closed and self._callback_name is not None:
            self._output.unregister_callback(self._callback_name)
        self._closed = True


_SCRIPT = r'''
const root = document.getElementById(initial.root);
if (!root) throw new Error('Annotation panel HTML is missing. Run this cell again.');
const get = role => root.querySelector('[data-role="' + role + '"]');
const canvas = get('canvas'), context = canvas.getContext('2d');
let state = initial.state, pending = initial.pending || [], busy = false, drag = null, image = null;
if(pending.length)get('tool').value=pending[0].kind+'_'+(pending[0].label?'positive':'negative');
const allControls = [...root.querySelectorAll('input,button,select')];
function lock(value) {
  busy = value;
  allControls.forEach(control => control.disabled = value || state.closed || (state.rebuild_required && control.hasAttribute('data-mutation')));
  ['name','exclude'].forEach(action => {root.querySelector('[data-action="'+action+'"]').disabled = value || state.closed || state.rebuild_required || state.object_id === null;});
}
function point(event) {
  const rect = canvas.getBoundingClientRect();
  return [(event.clientX-rect.left)*canvas.width/rect.width, (event.clientY-rect.top)*canvas.height/rect.height];
}
function inside(position) {return position.every(Number.isFinite) && position[0]>=0 && position[0]<state.width && position[1]>=0 && position[1]<state.height;}
function zoom() {
  const scale = Number(get('zoom').value);
  canvas.style.width = state.width*scale+'px'; canvas.style.height = state.height*scale+'px';
  get('zoom_value').textContent = Math.round(scale*100)+'%';
}
function draw() {
  if (!image) return;
  context.clearRect(0,0,canvas.width,canvas.height); context.drawImage(image,0,0);
  const scale = Number(get('zoom').value);
  context.lineWidth = 2/scale;
  const saved=state.saved_examples;
  if(saved && saved.frame===state.frame && saved.boxes.length){
    context.save();context.setLineDash([6/scale,4/scale]);
    saved.boxes.forEach((box,index)=>{
      context.strokeStyle=saved.box_labels[index]?'#39e68c':'#ff5e72';
      const [x,y,w,h]=box;context.strokeRect(x*state.width,y*state.height,w*state.width,h*state.height);
    });
    context.restore();
  }
  pending.forEach(item => {
    context.strokeStyle = item.label ? '#39e68c' : '#ff5e72';
    if (item.kind==='point') {
      const [x,y]=item.coordinates, radius=6/scale;
      context.beginPath(); context.moveTo(x-radius,y); context.lineTo(x+radius,y);
      if(item.label){context.moveTo(x,y-radius);context.lineTo(x,y+radius);}
      else {context.moveTo(x-radius,y-radius);context.lineTo(x+radius,y+radius);}
      context.stroke();
    } else {
      const [x,y,w,h]=item.coordinates;
      context.strokeRect(x*state.width,y*state.height,w*state.width,h*state.height);
    }
  });
  if(drag && drag.end){context.strokeStyle='#ffd54f';context.strokeRect(drag.start[0],drag.start[1],drag.end[0]-drag.start[0],drag.end[1]-drag.start[1]);}
  get('pending').textContent = pending.filter(item=>item.kind==='point').length+' points / '+pending.filter(item=>item.kind==='box').length+' boxes pending';
}
async function render(next) {
  state = next;
  get('frame').max=get('frame_number').max=state.frame_count-1;
  get('frame').value=get('frame_number').value=state.frame;
  get('source').textContent='Source frame '+state.source_index+' / local '+state.frame;
  const object=get('object'); object.replaceChildren(new Option('New object',''));
  state.objects.forEach(item=>object.add(new Option(item.id+': '+item.name,String(item.id))));
  object.value=state.object_id===null?'':String(state.object_id);
  get('name').value=state.name; get('concept').value=state.concept; get('name_prefix').value=state.name_prefix;
  get('box_update').value=state.box_update;
  const saved=state.saved_examples;
  get('saved').textContent=saved?('Saved examples: '+saved.boxes.length+' ('+saved.box_labels.filter(label=>label===1).length+' target / '+saved.box_labels.filter(label=>label===0).length+' background) · reference local frame '+saved.frame):'Saved examples: 0 · no reference frame';
  ['low','high','alpha'].forEach(key=>get(key).value=state[key]);
  get('low').max=get('high').max=state.maximum;
  get('status').textContent=state.status;
  get('coverage').textContent=state.rebuild_required?'Rebuild project required':(state.dirty?'Tracking is incomplete or needs updating':'All selected frames tracked');
  get('progress').max=state.frame_count;get('progress').value=state.tracked_frames;
  get('legend').textContent=state.objects.map(item=>item.id+': '+item.name).join(' · ') || 'No objects yet';
  const loaded = new Image(); await new Promise((resolve,reject)=>{loaded.onload=resolve;loaded.onerror=()=>reject(new Error('Image could not be decoded'));loaded.src=state.image;});
  image=loaded;canvas.width=state.width;canvas.height=state.height;zoom();draw();lock(false);
  if(globalThis.google?.colab?.output?.setIframeHeight) google.colab.output.setIframeHeight(document.documentElement.scrollHeight,true);
}
function payload() {
  return {frame:Number(get('frame_number').value), object_id:get('object').value===''?null:Number(get('object').value),
    concept:get('concept').value, name:get('name').value, name_changed:get('name').value!==state.name,
    name_prefix:get('name_prefix').value, box_update:get('box_update').value, direction:get('direction').value,
    low:Number(get('low').value),high:Number(get('high').value),alpha:Number(get('alpha').value),pending:pending};
}
async function request(action) {
  if(busy || state.closed)return;
  const data=payload(); lock(true);
  get('status').textContent=action==='track'?'Tracking selected frames… Wait for completion.':'Applying…';
  if(action==='track')get('progress').removeAttribute('value');
  try {
    const response=await google.colab.kernel.invokeFunction(initial.callback,[action,data],{});
    const result=response.data['application/json'];
    if(!result || !result.state)throw new Error('Invalid response from Python');
    if(result.ok && ['points','concept','exclude'].includes(action))pending=[];
    await render(result.state);
  } catch(error) {get('status').textContent='Connection/display error: '+error.message+'. Use the latest panel.';lock(false);}
}
get('frame').oninput=()=>{get('frame_number').value=get('frame').value;};
get('frame').onchange=()=>{pending=[];drag=null;request('frame');};
get('frame_number').onchange=()=>{pending=[];drag=null;request('frame');};
get('object').onchange=()=>{pending=[];drag=null;request('object');};
let previousTool=get('tool').value;
get('tool').onchange=()=>{if(previousTool.split('_')[0]!==get('tool').value.split('_')[0])pending=[];previousTool=get('tool').value;drag=null;draw();};
get('zoom').oninput=()=>{zoom();draw();};
root.querySelectorAll('[data-action]').forEach(button=>button.onclick=()=>{
  if(busy)return;
  const action=button.dataset.action;
  if(action==='undo'){pending.pop();draw();}
  else if(action==='clear'){pending=[];drag=null;draw();}
  else if(action==='track' && pending.length){get('status').textContent='Apply or clear pending annotations before tracking.';}
  else request(action);
});
canvas.onpointerdown=event=>{
  if(busy || state.closed || state.rebuild_required || event.button!==0)return;
  const position=point(event);if(!inside(position))return;
  const tool=get('tool').value;
  if(tool.startsWith('point')){pending.push({kind:'point',coordinates:position,label:Number(tool.endsWith('positive'))});draw();}
  else{drag={start:position,end:position};canvas.setPointerCapture(event.pointerId);}
};
canvas.onpointermove=event=>{if(drag){drag.end=point(event);draw();}};
canvas.onpointerup=event=>{
  if(!drag)return;
  const end=point(event),start=drag.start;drag=null;
  if(inside(end)){
    const left=Math.min(start[0],end[0]),top=Math.min(start[1],end[1]),w=Math.abs(end[0]-start[0]),h=Math.abs(end[1]-start[1]);
    if(w>0 && h>0)pending.push({kind:'box',coordinates:[left/state.width,top/state.height,w/state.width,h/state.height],label:Number(get('tool').value.endsWith('positive'))});
  }
  draw();
};
canvas.onpointercancel=()=>{drag=null;draw();};
root.dataset.ready='loading';
return render(state).then(()=>{root.dataset.ready='true';return true;});
'''
