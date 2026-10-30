# SAM 3.1 Segmentation and Tracking in Google Colab

[English](README.md) | [日本語](README_ja.md)

Segment grayscale or RGB images with text, foreground/background points, or boxes, then track multiple objects through a 2D time-lapse sequence. Export binary masks, overlay images, and prompt information for replaying an analysis.

Open [SAM3_1_Segmentation.ipynb](SAM3_1_Segmentation.ipynb) in Google Colab using **File > Upload notebook**. No additional source files need to be uploaded.

## Before you start

- Select a **Python 3.12 GPU runtime with native BF16 support**, preferably an A100. T4 GPUs are not supported by this notebook.
- Request access to [facebook/sam3.1](https://huggingface.co/facebook/sam3.1) and accept its model-access conditions.
- Have a Hugging Face read token ready. Enter it only in the hidden prompt in section 6; do not paste it into a code cell.
- Put the images in Google Drive or upload them through section 3. Keep the original images available if you want to replay saved prompts later.

### Getting a Hugging Face token

1. Create an account at [Hugging Face](https://huggingface.co/) and sign in.
2. Visit [facebook/sam3.1](https://huggingface.co/facebook/sam3.1), review the conditions, and request access. Wait for approval if required.
3. Using the same account, open [Settings > Access Tokens](https://huggingface.co/settings/tokens) and select **New token** or **Create new token**.
4. Give the token a descriptive name and select the **Read** role. Write access is not needed for this notebook.
5. Copy the token, store it securely, and paste it into the **hidden input prompt in section 6** when requested.

Creating a token does not grant model access: use a token belonging to an account authorized to access the model. Do not include tokens in documentation, code, or screenshots. See the [official token guide](https://huggingface.co/docs/hub/security-tokens) for details.

## Supported images

| Format | Supported input |
|---|---|
| TIFF / OME-TIFF | 8-bit or 16-bit single-channel grayscale or RGB; single images, time stacks, or frame folders |
| JPG / JPEG | 8-bit grayscale or RGB; one frame per file |
| PNG | 8-bit grayscale or RGB, or 16-bit grayscale; one frame per file |

Only **2D images and 2D time-lapse sequences** are supported. Non-RGB multichannel images and Z stacks are not supported. Palette, transparent/RGBA, animated, and CMYK images are not supported.

For a frame sequence, use one folder per sequence. Images must have matching dimensions, color mode, and bit depth. Filenames are sorted naturally, so `frame2.png` comes before `frame10.png`. Keep annotation masks and unrelated images out of the input folder.

## Setup and image loading

Run the notebook sections in order:

1. **Section 1 - Install dependencies.** Run once. After installation succeeds, the Python session automatically restarts. Reconnect and continue at section 2; do not rerun section 1 after that restart.
2. **Section 2 - Load the notebook tools.** Run after restarting.
3. **Section 3 - Connect your images.** Mount Google Drive or upload your images.
4. **Section 4 - Choose the input and time range.** Set `INPUT_PATH` to the full path of an image file or a folder containing one sequence.
5. **Section 5 - Prepare and inspect the model input.** Check the displayed image, contrast, and frame order before proceeding.
6. **Section 6 - Load the model.** Enter your Hugging Face read token when prompted. Loading can take time.
7. **Section 7 - Open the controls.** Segment objects and run tracking as described below.

### Input settings in section 4

| Parameter | How to set it |
|---|---|
| `INPUT_PATH` | Full path to an image file or a folder of frame images |
| `AXES` | Usually leave empty. For TIFF, override only after confirming the stored dimensions |
| `SERIES` | TIFF series index; use `0` for JPG/PNG |
| `START` | First frame to include, starting from `0` |
| `STOP` | First frame to exclude; `None` includes the remaining frames |
| `STEP` | Frame interval; `1` uses every frame |
| `INTENSITY_LOW`, `INTENSITY_HIGH` | Usually leave as `None`; set raw pixel values to choose a fixed model-input intensity range |

For JPG/PNG, leave `AXES` empty or use `YX`, including RGB images. Common TIFF axes are `YX` for grayscale images and `TYX` for grayscale time stacks. RGB TIFFs use an `S` axis for color samples, such as `YXS` or `TYXS`. Never relabel channels or depth slices as time.

The original images are unchanged. Model input is converted to 8-bit RGB using a shared intensity range across the sequence. Grayscale defaults use sampled percentiles; RGB defaults preserve the full pixel range. If the prepared image looks too dark or saturated, adjust the intensity limits in section 4 and rerun sections 4 onward. Start with a short sequence when checking new data.

## Segmenting objects in section 7

Use **Frame** or **Frame number** to select a frame. Frame numbers start at zero within the selected time range.

### Text prompts

1. Enter a short English concept in **Concept**, such as `dog`, `person`, or `car`.
2. Click **Apply concept / boxes**.
3. Review the detected masks and object IDs.

Text prompts can find multiple matching objects. Detection depends on the object, background, lighting, and object size. You can also draw positive or negative boxes before applying the concept to provide visual examples. Boxes are not guaranteed to select only one object.

Use one concept at a time. You can update the text or example boxes and redetect within the same project using the procedure below. Save the current analysis first if you want to retain it.

### Boxes as examples: detect multiple objects without text

**Boxes and points have different roles in this notebook.** A box is an example of the kind of object to find, not a boundary restricting detection to its interior.

| Prompt | What it does |
|---|---|
| **+ Box** with **Apply concept / boxes** | Supplies a positive example. The model looks for matching objects throughout the image, including outside the box |
| **− Box** with **Apply concept / boxes** | Supplies an example of an unwanted object to help distinguish what should be excluded |
| **+ Point / − Point** with **Apply points** | Selects or refines one object instance. Points do not automatically find all similar objects |

To try detection using boxes alone:

1. Start a fresh project by running section 7. Save previous work first if you want to retain it.
2. Select a frame containing representative objects and leave **Concept** empty.
3. Choose **+ Box** and drag a rectangle enclosing one representative object in full.
4. Before applying, optionally draw more **+ Box** examples or **− Box** examples around unwanted objects.
5. Click **Apply concept / boxes**, not **Apply points**. Inspect the masks and IDs across the image, including outside the boxes.
6. If needed, select individual object IDs and refine them with points. Then choose **Direction = both** and click **Track** to process the sequence.

A box can yield multiple matching objects, one object, or no objects; detection quality depends on the images and examples. Text can also be combined with boxes, provided both describe the same kind of object. A negative box supplies a negative example rather than simply erasing a rectangular area.

### Adding or changing example boxes after detection

You can redetect within the same project, without rerunning section 7 or loading the model again. Saved example boxes are shown as dashed rectangles on their reference frame; the panel also shows their count and reference frame. Pending boxes are the ones you have drawn but not applied yet.

| Box update | How it uses the examples |
|---|---|
| **Add to saved** | Combines the saved boxes with the new pending boxes, then redetects. Add boxes on the same reference frame |
| **Replace saved** | Discards the saved boxes and redetects using only the pending boxes. Use this to redraw, remove, or change examples, or to choose a different reference frame |

1. Save prompts and results first if you want to retain the previous analysis.
2. To add examples, return to the reference frame, draw more positive/negative boxes, and choose **Add to saved**.
3. To change examples, draw the full new set and choose **Replace saved**. You can choose another frame for this. To remove all boxes, clear pending annotations, keep a nonempty **Concept**, and choose **Replace saved**.
4. Click **Apply concept / boxes**. You can also edit **Concept** before applying to change the text used for detection.
5. Review the new objects, add point corrections if needed, and run **Track** again before exporting.

**Redetection replaces the active detection results.** Objects receive new IDs and automatic names; previous individual names, exclusions, and point corrections are not transferred to the new objects. Previous tracking results are cleared. The shared **Name prefix** is retained. This also applies when switching from a point-only analysis to text/box detection. Drawing or clearing pending annotations alone does not perform redetection.

**Clear pending** and **Undo pending** do not remove saved boxes. To remove saved examples, use **Replace saved** as described above. Boxes from different frames cannot be combined using Add to saved. An excessive number of detected candidates can also be limited by `MAX_OBJECTS` in section 6.

### Foreground and background points

1. Select **New object** in **Object**.
2. Choose **+ Point (target)** and click inside the object.
3. Choose **- Point (background)** to mark regions that should be excluded, if needed.
4. Click **Apply points** and review the mask.
5. Select **New object** again to add another object.

To refine an existing object, select its ID, add target/background points, and click **Apply points**. Points can also refine objects first detected using text. **Undo pending** and **Clear pending** affect annotations that have not yet been applied.

### Automatic names and bulk renaming

You do not need to name each object separately. Text detection automatically uses the concept and object ID, so `car` produces `car 1`, `car 2`, and so on. The number is the stable object ID, not a separate counter for each name.

Use **Name prefix** to choose a shared name for newly detected or point-added objects. For example, enter `target` before applying points to create `target 1`, `target 2`, and so on as you add objects. The prefix is retained when selecting **New object** again. Leave it blank to use the concept text, or `Object` if no concept has been set. Objects discovered later during tracking follow the same naming rule.

To rename objects already registered, enter a prefix and click **Apply prefix to all**. This renames every active object, including individually named ones, and sets the prefix for future objects. It works without selecting an existing object and does not require segmentation or tracking to be rerun.

For an individual exception, select the object, enter its exact name in **Name**, and click **Save name**. Names appear with the numeric ID on the preview and exported overlay, for example `1: car 1`. They also appear in the object's mask folder, for example `masks/object_000001_car_1/`, and are retained in the saved JSON. Objects still named `Object 1`, etc. keep the ID-only image label and `object_000001` folder. Renaming does not change segmentation or tracking results.

Characters unsuitable for file paths are replaced when naming folders; the original name is retained in JSON. Frame filenames remain numbered, and a combined overlay stays in `overlays/frame_000000.png`. Renaming does not modify files already exported; export again to another empty or new folder to save the updated names.

Non-Latin names, including Japanese, require an installed system font that supports those characters for image labels. The notebook selects from available fonts; unsupported characters may appear as missing glyphs. Use English/alphanumeric names if a suitable font is unavailable. Original Unicode names are still retained in JSON and sanitized folder names.

### Correcting masks, adding objects, and excluding objects

These operations use the current project; do not rerun section 7 unless you want to start over. They also work on objects initially detected using text.

| What you want to do | Steps in the panel |
|---|---|
| Remove an unwanted protrusion or background region | Select the frame and the existing object ID in **Object**. Place **− Point (background)** on the unwanted region, then click **Apply points** |
| Restore a missing part of an object | Select the frame and existing object ID. Place **+ Point (target)** inside the missing region, then click **Apply points** |
| Add another object | Select a frame where it is visible. Choose **New object**, place at least one **+ Point (target)** inside it, then click **Apply points** |
| Correct a frame where tracking became inaccurate | Move to that frame, select the existing object ID, add target/background points, and click **Apply points** |
| Exclude an unwanted object | Select its ID in **Object**, then click **Exclude object** |

**Apply points** reruns segmentation for the selected frame. Review the updated mask, then set **Direction** to `both` and click **Track** to update results across the selected sequence. Corrections, additions, and exclusions invalidate previous tracking results; complete tracking again before exporting.

Points guide a new prediction rather than directly painting or erasing pixels. A background point can change other parts of the mask; add target points inside regions you want to retain if necessary. Applied points accumulate for that object and frame. **Undo pending** and **Clear pending** cannot remove already-applied points.

**Exclude object applies to the entire analysis**, not just the displayed frame. The excluded object is omitted from newly exported results. There is no undo-exclusion button; save prompts and results before excluding objects if you want to retain the previous analysis. To correct an incorrect region in just one frame, refine the existing object with points instead of excluding it.

When adding an object partway through a sequence, `both` also attempts to track it in earlier frames. Check for incorrect masks before its actual appearance. Changes do not update images already saved to disk; export the revised analysis to a different empty or new folder.

### Display controls

Adjust **Display low**, **Display high**, and **Mask opacity**, then click **Update display**. These controls change the preview, not the images used by the model. Use **Zoom** and scroll inside the image area to inspect boundaries.

## Tracking and reviewing results

Select a starting frame and choose **Direction**:

- `both`: track across the full selected sequence in both directions.
- `forward`: track from the selected frame toward the end.
- `backward`: track from the selected frame toward the beginning.

Click **Track** and wait for it to finish. Then use the frame controls to inspect masks and IDs throughout the sequence. If needed, add correction points on a problem frame and track again. Export requires complete coverage of the selected sequence after the latest correction; `both` is the simplest choice for this.

Review object identities during contact, disappearance, reappearance, splitting, and merging. The notebook does not record split/merge relationships or calculate object measurements.

## Starting over or switching image sequences

The following steps assume the Colab session is still connected and the model is loaded. You do not need to reinstall dependencies or reload the model. If the runtime has been reset, follow the setup instructions again.

### Start over with the same images

1. Save any prompts you want to keep in section 8 and export any completed results in section 9.
2. Rerun **section 7** to create a fresh project. This clears the current prompts and tracking results from the live session, while keeping saved files.
3. Use only the newly displayed panel. Select a frame and begin again with text, points, or boxes.
4. When saving the new analysis, use a new JSON filename in section 8 and an empty or new output folder in section 9.

This procedure retains the input images, time range, and model-input intensity settings. To change those settings, rerun sections 4 and 5 before section 7.

### Switch to a different image sequence

1. Save the current prompts in section 8 and export any completed results in section 9 before switching.
2. If uploading new files, run **section 3** and select the frames belonging to the new sequence. If the images are already available on the mounted Drive, you can skip this step.
3. In **section 4**, change `INPUT_PATH` to the new file or frame folder. Check `AXES`, `SERIES`, `START`, `STOP`, `STEP`, and the intensity limits; settings from the previous sequence are not reset automatically. Run the cell.
4. Run **section 5** to prepare the new input and inspect its appearance and frame order.
5. **Skip section 6** if the loaded model and its configuration are unchanged. Run **section 7** to create a fresh project for the new sequence, then segment and track using the new panel.
6. Choose a new JSON filename in section 8 and an empty or new output folder in section 9. Clear `REPLAY_PROJECT` unless you intend to replay a project saved for this same sequence.

Do not replay prompts from the previous sequence on different source images. If you change model settings such as `MAX_OBJECTS`, rerun section 6 before creating the new project in section 7.

## Saving prompts and replaying an analysis - section 8

- Set `SAVE_PROJECT` to the **full output path including a new `.json` filename**, then run the cell. Leave `REPLAY_PROJECT` empty.
- To replay, first load the same images and preprocessing settings and create a fresh project in section 7. Set `REPLAY_PROJECT` to the **full path of the saved JSON**, leave `SAVE_PROJECT` empty, and run section 8.

Existing JSON files are not overwritten. The saved project contains prompts and reproduction settings, not result images or loaded model memory. Replay reruns the saved operations and can take time; it stops if masks or object identities differ.

## Exporting segmentation images - section 9

Set `OUTPUT_DIRECTORY` to the **full path of an empty or new output folder**, preferably on Google Drive and outside the input folder. A new folder is created automatically. Folders containing files are rejected to avoid overwriting results.

Choose `OUTPUT_MODE`:

| Value | Export |
|---|---|
| `"mask"` | 8-bit binary TIFFs, background `0` and foreground `255`, in a separate `masks/object_<ID>_<name>/` folder for each named object (ID-only folders for default names) |
| `"overlay"` | RGB PNGs with colored masks, object IDs, and custom names in `overlays/` |
| `"both"` | Both image formats |

Images retain the original XY dimensions. Each frame is saved separately. Mask pixel values do not encode object IDs; the folder identifies the object. An absent object has an all-zero mask. Independent binary masks preserve overlaps; overlays display the lowest object ID at overlapping pixels.

The export also contains `project.json` with prompts and reproduction settings. Its `export.source_frame_indices` list maps output frame numbers to the original sequence. The JSON is written last; if export is interrupted, use a different empty folder and retry.

Save to Drive or download your files before the Colab runtime expires. Files stored only under `/content` are temporary.

## Reference

When using SAM 3 or SAM 3.1 in research, cite the paper recommended by the [official repository](https://github.com/facebookresearch/sam3#citing-sam-3):

Carion, N., et al. (2025). **SAM 3: Segment Anything with Concepts.** arXiv:2511.16719. https://doi.org/10.48550/arXiv.2511.16719

[Read the paper](https://arxiv.org/abs/2511.16719). SAM 3.1's Object Multiplex method is described in Appendix H of the updated paper, as noted in the official release notes. State that you used SAM 3.1 and record the checkpoint revision and source-code commit for reproducibility.

## Related resources

- [SAM 3.1 release](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md)
- [Official SAM 3.1 example](https://github.com/facebookresearch/sam3/blob/main/examples/sam3.1_video_predictor_example.ipynb)
- [Reference SAM2 notebook](https://github.com/YusukeKimata-Moo/SAM2-segmentation)

## License

Original notebook code, helpers, and documentation are distributed under the [MIT License](LICENSE). Meta's SAM code and model weights are governed separately by the SAM License and model-access conditions; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
