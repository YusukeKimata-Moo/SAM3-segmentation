"""Build the standalone Colab notebook from the included Python helpers."""

from pathlib import Path
import hashlib
import json
import textwrap

import nbformat

ROOT = Path(__file__).resolve().parents[1]
cells = []


def markdown(text):
    cells.append(nbformat.v4.new_markdown_cell(textwrap.dedent(text).strip()))


def code(text, hidden=False):
    cell = nbformat.v4.new_code_cell(textwrap.dedent(text).strip())
    if hidden:
        cell.metadata["cellView"] = "form"
    cells.append(cell)


markdown("""
# SAM 3.1: Image Segmentation and Time-lapse Tracking

Segment **grayscale or RGB TIFF, JPG, or PNG images** with text, foreground/background points,
or rectangles. Track multiple objects forward and backward from any selected frame,
correct their masks, and export segmentation images at the original image size.

**Before you start:** select a Python 3.12 GPU runtime with native BF16 support (A100 preferred;
T4 is not supported here). Request access to [SAM 3.1 weights](https://huggingface.co/facebook/sam3.1).
Text-prompt results depend on image content; point prompts also work without text.
Review masks and object identities before exporting.

Your original images remain unchanged. The model sees 8-bit RGB PNG images. The exported
results contain only segmentation images and a JSON file with prompts and reproduction settings.
This notebook processes **2D images and 2D time-lapse sequences**, with uint8 or uint16
grayscale or RGB TIFF pixels, 8-bit grayscale/RGB JPG/JPEG or PNG, or 16-bit grayscale PNG.
Non-RGB multichannel images and Z stacks are not supported.

**Validation status:** CPU image/session tests accompany the source. GPU inference,
Colab widget behavior, and accuracy on your images must be checked in your runtime.
Start with a short sequence and two objects before processing a longer recording.
""")

markdown("""
## 1. Install the required packages

Use **Runtime > Change runtime type > Runtime Version** to select a version with Python
3.12 (for example, 2026.07 when available), then choose a compatible GPU. Python 3.13
is incompatible with this notebook's NumPy 1.26.4 requirement.

Run this cell once in a fresh GPU runtime. It installs a fixed SAM source revision and
the notebook dependencies without replacing PyTorch. After installation and a fresh-process
import check succeed, the session restarts automatically to clear previously loaded libraries.
A temporary disconnection or unexpected-session-exit notification is expected. Reconnect if
needed, then continue at **section 2**; do not run section 1 again unless reinstalling.
Variables and loaded models are cleared on restart; save your project before reinstalling
during an analysis. Failed installation or import checks stop without restarting.
Model weights are downloaded later.
""")
code('''
import subprocess
import sys

if sys.version_info[:2] != (3, 12):
    raise RuntimeError("Use a Python 3.12 runtime: Runtime > Change runtime type > Runtime Version. Then run section 1 and restart the session.")
import torch
if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
    raise RuntimeError("Select an A100 or another GPU with native BF16 support. T4 is not supported.")
SAM_REVISION = "7f473cfd983d5eb43040ff608f66bff0d7bdcb53"
packages = [
    "setuptools<81", "numpy==1.26.4", "tifffile>=2024.8.30",
    "imagecodecs", "Pillow>=10", "ipywidgets>=8,<9", "ipympl>=0.9,<0.10",
    f"sam3[notebooks] @ git+https://github.com/facebookresearch/sam3.git@{SAM_REVISION}",
]
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--only-binary=numpy", *packages])
subprocess.check_call([sys.executable, "-c",
    "import numpy; import numpy.random; import sklearn; import skimage; import matplotlib; import torchvision; "
    "assert numpy.__version__ == '1.26.4', numpy.__version__; "
    "print('NumPy and image-library imports passed in a fresh Python process.')"])
print("Installation and import checks passed. Restarting the session now; reconnect and run section 2 onward.", flush=True)
try:
    import google.colab
except ImportError:
    print("Outside Colab: restart your Python kernel manually, then run section 2 onward.", flush=True)
else:
    import os
    import signal
    os.kill(os.getpid(), signal.SIGKILL)
''')

markdown("""
## 2. Load the notebook tools

Run the following cell once after restarting. The tools are included in this notebook;
you do not need to upload other Python files or publish a repository.
""")
embedded = {path.name: path.read_text(encoding="utf-8")
            for path in sorted((ROOT / "sam31_colab").glob("*.py"))}
code('''
#@title Load the included tools
import sys
from pathlib import Path
tool_root = Path("/content/sam31_notebook") if Path("/content").is_dir() else Path.cwd() / ".sam31_runtime"
package_dir = tool_root / "sam31_colab"
package_dir.mkdir(parents=True, exist_ok=True)
''' + "\nhelper_sources = " + repr(embedded) + '''
for filename, source in helper_sources.items():
    (package_dir / filename).write_text(source, encoding="utf-8")
if str(tool_root) not in sys.path:
    sys.path.insert(0, str(tool_root))
from sam31_colab.io import ImageSettings, TiffSequence
from sam31_colab.runtime import check_environment, load_predictor
from sam31_colab.engine import TrackingProject
from sam31_colab.ui import TrackingUI
from sam31_colab.colab_ui import ColabTrackingUI
print(check_environment())
''', hidden=True)

markdown("""
## 3. Connect your images

Use a TIFF/OME-TIFF file, a grayscale or RGB JPG/JPEG or PNG file, or a folder of supported
images in natural filename order (for example, `frame2.png` before `frame10.png`).
A folder's files are concatenated in time. Each JPG/PNG file represents one frame.
All files must have matching XY dimensions, grayscale/RGB mode, and bit depth.
Use the file browser to copy the path into `INPUT_PATH` below.

Set `MOUNT_DRIVE=True` to access Google Drive. To upload files from your computer,
set `UPLOAD_FILES=True`; the upload directory will be printed. Upload only images belonging
to the same sequence. Large uploads use runtime disk space and may be lost on disconnect.
""")
code('''
MOUNT_DRIVE = False
UPLOAD_FILES = False

if MOUNT_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
if UPLOAD_FILES:
    from google.colab import files
    from uuid import uuid4
    upload_dir = Path("/content/sam31_inputs") / uuid4().hex[:12]
    upload_dir.mkdir(parents=True, exist_ok=False)
    uploaded = files.upload(target_dir=str(upload_dir))
    print("Input folder:", upload_dir)
    del uploaded
''')

markdown("""
## 4. Choose the input and time range

Indices are **zero-based**; `STOP` is exclusive. For JPG/PNG, leave `AXES` empty (or use
`YX`) and `SERIES=0`, including RGB images. Palette, transparent, and animated JPG/PNG
inputs are unsupported; save a plain grayscale or RGB image before uploading.
JPEG compression can affect boundaries.
For TIFF, leave `AXES` empty to use trustworthy TIFF
metadata. If the reader reports an unknown page axis such as `QYX`, set `AXES="TYX"`
only after confirming that the pages represent time. Grayscale images normally use `YX`
and grayscale time-lapse sequences use `TYX`. For RGB TIFF, S means the three stored color
samples: common orders are `YXS` / `TYXS` or planar `SYX` / `TSYX`. For example, change
an unknown `QYXS` page axis to `TYXS` only if those pages represent time. Keep S in its
stored position. RGB input must be tagged as RGB in the TIFF, with exactly three samples;
RGBA and palette images are not supported.

T means time; tracking follows T only. Use single-channel grayscale or RGB input.
Non-RGB multichannel images (C > 1) and images with multiple Z slices are rejected.
Do not label channels or depth slices as time frames. Singleton C or Z axes in TIFF
metadata are accepted. RGB samples are processed together as one color image.
A selected time stride is recorded in the output; skipping frames can make tracking harder.

For grayscale, leave the intensity limits as `None` for one shared range estimated from
sampled frames (1st to 99.8th percentile), or enter fixed raw pixel values. For RGB, the
default range is 0-255 for uint8 (unchanged colors) or 0-65535 for uint16 (converted to
8-bit RGB). Any custom limits apply equally to R, G, and B for all frames; color samples
are never normalized independently. There is no automatic frame-by-frame normalization.
Empty or uniformly bright inputs may need explicit limits.
""")
code('''
INPUT_PATH = ""  # Enter an image file path or a folder of images from one sequence.
AXES = ""
SERIES = 0
START = 0
STOP = None
STEP = 1
INTENSITY_LOW = None
INTENSITY_HIGH = None

if not INPUT_PATH.strip():
    raise ValueError("Enter INPUT_PATH before running this cell.")
settings = ImageSettings(
    input_path=INPUT_PATH, axes=AXES, series=SERIES,
    start=START, stop=STOP, step=STEP,
    intensity_low=INTENSITY_LOW, intensity_high=INTENSITY_HIGH,
)
sequence = TiffSequence(settings)
print("Selected shape (T, Y, X):", sequence.shape)
print(sequence.metadata)
''')

markdown("""
## 5. Prepare and inspect the model input

This cell creates a new PNG cache with the original XY dimensions. It does not change
the source images. Review the frame order, contrast, and histogram before
continuing. Reduce the selected frame range if runtime RAM or disk is insufficient.
The runtime holds model frames and output masks; long sequences are not split automatically.
""")
code('''
from uuid import uuid4
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

CACHE_ROOT = "/content/sam31_cache"
prepared = sequence.prepare(Path(CACHE_ROOT) / uuid4().hex)
first_source_frame = sequence.frame_indices[0]
preview = sequence.preview(first_source_frame, prepared["low"], prepared["high"])
figure, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].imshow(preview)
axes[0].set_title(f"Model input: source frame {first_source_frame}")
axes[0].set_xlabel("X (pixels)")
axes[0].set_ylabel("Y (pixels)")
raw_first_frame = sequence.raw_frame(first_source_frame)
sample_stride = max(1, preview.shape[0] * preview.shape[1] // 200000)
if sequence.is_rgb:
    for sample_index, sample_name, sample_color in ((0, "Red", "red"), (1, "Green", "green"), (2, "Blue", "blue")):
        axes[1].hist(raw_first_frame[..., sample_index].ravel()[::sample_stride], bins=100,
                     histtype="step", color=sample_color, label=sample_name)
else:
    axes[1].hist(raw_first_frame.ravel()[::sample_stride], bins=100)
axes[1].axvline(prepared["low"], color="red", label="Lower limit")
axes[1].axvline(prepared["high"], color="green", label="Upper limit")
axes[1].set_xlabel("Raw intensity")
axes[1].set_ylabel("Sampled pixel count")
axes[1].legend()
figure.tight_layout()
plt.show()
print("Shared intensity range:", prepared["low"], "to", prepared["high"])
print("Selected source frames:", list(sequence.frame_indices[:20]))
print("Approximate mask RAM per tracked object (GiB):", round(int(np.prod(sequence.shape)) / 2**30, 3))
''')

markdown("""
## 6. Load SAM 3.1

Enter a read-only Hugging Face token in the hidden prompt each time you run this entire
cell. Pasted characters are not displayed; press Enter after pasting. Colab Secrets are
not read by this cell. The token is not included in project files. Model access must be approved
for your account. Surrounding whitespace is removed automatically; embedded whitespace
is rejected before connecting. Never print your token or share an error containing it.
Set `MAX_OBJECTS` to an upper bound for the entire recording, including
objects appearing later. Increasing it may require more GPU memory.

The notebook uses SAM 3.1 weights for both images and time series. FlashAttention 3 and
compilation are disabled for the initial run. Checkpoint revision and environment versions
are saved with results. No SAM2/SAM3 fallback is used.
""")
code('''
import getpass
from importlib.metadata import version, PackageNotFoundError

MAX_OBJECTS = 16
MODEL_REVISION = None  # Paste a saved checkpoint revision to reproduce a previous run.
hf_token = None
try:
    hf_token = getpass.getpass("Paste your Hugging Face read token, then press Enter: ").strip()
    if not hf_token or any(ord(character) <= 32 or ord(character) >= 127 for character in hf_token):
        raise ValueError("The token is empty or contains invalid characters. Run this entire cell again and enter the token.")
    predictor, environment = load_predictor(MAX_OBJECTS, token=hf_token, model_revision=MODEL_REVISION)
finally:
    hf_token = None
environment["packages"] = {}
for package in ("numpy", "tifffile", "Pillow", "ipywidgets", "ipympl", "torchvision", "sam3"):
    try:
        environment["packages"][package] = version(package)
    except PackageNotFoundError:
        environment["packages"][package] = "unavailable"
print("SAM 3.1 is ready.")
print(environment)
''')

markdown("""
## 7. Open the annotation and tracking controls

Choose a frame and start with one of these approaches:

- Enter a short English concept, such as `dog`, `person`, or `car`, and detect candidates. Rectangles
  provide positive or negative visual examples for the concept; they are not guaranteed
  to identify only one object. Use one concept at a time.
- Select a new object and add foreground points. Add background points to exclude nearby
  regions. Apply the points, then select another object to add it.
- Select an existing object to refine its mask with points.

### Boxes versus points

**Boxes are visual examples for detecting matching objects throughout the image, including
outside the boxes.** They do not restrict detection to the enclosed object or region.
Positive boxes show what to find; negative boxes show unwanted examples. A negative box
does not simply erase the pixels inside it. Points, by contrast, select or refine one
object instance; clicking one object does not automatically detect all similar objects.

To detect multiple objects using **boxes alone**:

1. Start a fresh project with the cell below; save previous work before restarting it.
2. Select a representative frame and leave **Concept empty**.
3. Select **+ Box** and drag around one representative object, enclosing the whole object.
4. Optionally add more positive examples or use **− Box** around unwanted objects before applying.
5. Click **Apply concept / boxes**, not Apply points. Review masks and IDs across the entire image.
6. Refine individual objects with points if needed, then choose **Direction = both** and click **Track**.

Text and boxes can be combined when they describe the same kind of object. Results may
contain multiple objects, one object, or none. Set a sufficient **MAX_OBJECTS** in section 6.
### Update examples and redetect

Use **Box update** to choose how the next **Apply concept / boxes** uses your examples:

- **Add to saved**: combine saved examples with pending boxes on the same reference frame.
- **Replace saved**: use only the pending boxes. Redraw the full set to change examples;
  you may choose a different reference frame. To remove all boxes, keep Concept nonempty
  and apply Replace saved without pending boxes.

Saved examples are dashed on their reference frame; their count and reference frame are
shown in the panel. Undo pending and Clear pending affect only unapplied annotations.
You can edit Concept before reapplying, or switch from point-only input to text/boxes.

**Reapplying detection replaces active objects with new IDs and automatic names, clears
previous tracking results, and does not transfer individual names, exclusions, or point
corrections. Name prefix is retained.** Save the current analysis before redetecting if
you want to keep it. Review the new masks, add corrections, and run Track again before export.
No model reload or section 7 restart is needed to update examples.

In Colab, all controls use native HTML inputs and the image uses an HTML canvas.
Click to add points and drag to draw boxes. Adjust Zoom and scroll inside the image to
inspect a region. Undo removes the most recent pending point or box. Display contrast
changes only the preview; use Update display to apply the display limits and opacity.
Text detections are named automatically using the concept and object ID, such as `car 1`.
Set **Name prefix** before applying points or detecting objects to use another shared name;
the prefix persists when selecting New object. Leave it blank to use the concept, or Object
when no concept is set. **Apply prefix to all** renames all active objects and sets the prefix
for future objects, without rerunning inference. Individual names are still editable:
select an object, enter its exact name in **Name**, and click **Save name**. Names appear
beside the ID in the image and exported overlays, and in that object's mask folder name.
Default names keep ID-only image labels and folders. Renaming alone does not require
tracking again. Export to another empty or new folder to save changed names.
Non-Latin image labels require an installed font supporting those characters; use English
or alphanumeric names if a suitable font is unavailable. JSON retains original names.
Add newly appearing objects with new IDs when needed.

Track in **both** directions to cover the selected sequence. A forward run from frame 0
or a backward run from the final frame also covers it. Partial coverage cannot be exported
as a complete result. Add correction points on problem frames and track again.

After tracking finishes, use the **Frame** slider or **Frame number** to inspect any selected frame with
colored masks and object IDs. Adjust **Mask opacity** and **Display range** to inspect
boundaries. Frame numbers refer to the selected sequence; the source index is also shown.
Frames are displayed individually, not as a regularly spaced gallery. During tracking,
the status reads Tracking; frame navigation is available again when tracking finishes.
If a panel is displayed again, use the most recent panel; earlier panels are disconnected.

To start over entirely, save your current project first, then rerun the next cell.
Re-running it discards the live session, not saved exports. To update examples or text
within the current project, use Box update and Apply concept / boxes instead.
""")
code('''
try:
    import google.colab
except ImportError:
    in_colab = False
else:
    in_colab = True
if "annotation_ui" in globals():
    annotation_ui.close()
if "project" in globals():
    project.close()
project = TrackingProject(sequence, predictor, prepared, environment=environment, max_objects=MAX_OBJECTS)
if in_colab:
    annotation_ui = ColabTrackingUI(project)
    annotation_ui.show()
else:
    from IPython import get_ipython
    get_ipython().run_line_magic("matplotlib", "widget")
    annotation_ui = TrackingUI(project)
    display(annotation_ui.widget)
''')

markdown("""
## 8. Save prompts or reopen a saved project

Save the project JSON before disconnecting. Computing source fingerprints can take time for
large images or sequences. It records prompts and source fingerprints,
not GPU memory. Choose a new filename; existing files are not overwritten.
To reopen, select the same input/settings above, create a fresh project in section 7,
and set `REPLAY_PROJECT` below. Replay may recompute saved tracking steps and take time.
Different masks or identities during replay stop the process to avoid editing a wrong object.
Move your JSON and results to Drive or download them before the runtime expires.
""")
code('''
SAVE_PROJECT = ""  # Enter the full output path, including a new .json filename; leave empty to skip saving.
REPLAY_PROJECT = ""  # Enter the full path to an existing saved .json file to replay in a fresh project; otherwise leave empty.
if SAVE_PROJECT and REPLAY_PROJECT:
    raise ValueError("Choose either save or replay in one run.")
if SAVE_PROJECT:
    project.save_project(SAVE_PROJECT)
    print("Saved project:", SAVE_PROJECT)
if REPLAY_PROJECT:
    project.replay_project(REPLAY_PROJECT)
    annotation_ui.refresh()
    print("Project replay complete. Review the masks and object IDs.")
''')

markdown("""
## 9. Export and review

Export only after the full selected sequence has been tracked following your latest
correction. Choose an **empty or new** output folder outside the input directory. Set `OUTPUT_MODE` to:

- `"mask"`: 8-bit binary TIFFs (background 0, foreground 255), one frame per file in
  `masks/object_<ID>_<name>/` for custom names, or `masks/object_<ID>/` for default names.
  Folder names replace characters unsuitable for file paths; JSON retains the original name.
  Each object has its own folder, so overlapping masks are preserved.
  An absent object has an all-zero mask. Object IDs are in folder names, not pixel values.
- `"overlay"`: RGB PNGs in `overlays/`, with colored masks, object IDs, and custom names on the
  displayed grayscale or RGB image. These use 8 bits per color channel and are for visual
  review. At overlapping pixels the lowest object ID is shown.
- `"both"`: both image formats.

Image dimensions are unchanged. Frame filenames count from zero within the selected time
range. Alongside the images, `project.json` records prompts, object IDs and names, image
preprocessing settings, source fingerprints, model versions, tracking history, and export
settings. Its `export.source_frame_indices` list maps each local frame index to the original
sequence index. Keep the same source images to replay the project.

Only the selected segmentation images and `project.json` are written. The JSON is written
last; an export interrupted before it is saved is incomplete. Choose a fresh folder when
retrying. Review masks, missing objects, and ID switches interactively before exporting.
""")
code('''
OUTPUT_DIRECTORY = ""  # Enter the full path to an empty or new output directory, preferably on Drive.
OUTPUT_MODE = "mask"  # Choose "mask", "overlay", or "both".
if not OUTPUT_DIRECTORY.strip():
    raise ValueError("Enter OUTPUT_DIRECTORY before exporting.")
result_directory = project.export(OUTPUT_DIRECTORY, output_mode=OUTPUT_MODE)
print("Results saved to:", result_directory)
''')

markdown("""
## Checks before a larger analysis

1. Confirm T (time) and frame order against your acquisition. Use grayscale or RGB 2D
   images or a 2D time-lapse sequence. Non-RGB multichannel images and Z stacks are
   not supported. For RGB, check that the displayed red, green, and blue colors are correct.
2. Check at least two objects on the first, middle, and final frames, including contact
   or disappearance events. Compare representative masks with manual annotations.
3. Add correction points on a middle frame and track again. Confirm that the correction
   is retained and that old results cannot be exported accidentally.
4. Open the exported images in Fiji and verify image size and frame order. Binary TIFFs
   should be 8-bit with values 0 and 255; their folder names identify objects. Check overlay
   alignment and IDs if exported.
5. Save and replay a short project before relying on replay for a long experiment.

If memory is exhausted, keep your saved project and restart the runtime. Select fewer
frames or objects explicitly; the notebook does not silently lower resolution or split
tracks across independent sessions. CPU tests do not establish GPU accuracy or capacity.

Sources: [SAM 3.1 release](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md),
[official example](https://github.com/facebookresearch/sam3/blob/main/examples/sam3.1_video_predictor_example.ipynb),
[reference SAM2 notebook](https://github.com/YusukeKimata-Moo/SAM2-segmentation).
Weights and upstream code retain Meta's license and model-access conditions.
""")

for position, cell in enumerate(cells):
    cell.id = hashlib.sha256(f"{position}:{cell.source}".encode()).hexdigest()[:12]
notebook = nbformat.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
    "colab": {"name": "SAM3_1_Segmentation.ipynb", "provenance": []},
    "accelerator": "GPU",
})
nbformat.validate(notebook)
destination = ROOT / "SAM3_1_Segmentation.ipynb"
nbformat.write(notebook, destination)
print(f"Built {destination.name}: {len(cells)} cells; {len(embedded)} included helpers")
