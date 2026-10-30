"""Check your GPU and load the SAM 3.1 checkpoint."""

from functools import wraps
from importlib import metadata
import inspect
import json
from pathlib import Path
import sys

SAM_REVISION = "7f473cfd983d5eb43040ff608f66bff0d7bdcb53"
MODEL_REPOSITORY = "facebook/sam3.1"
MODEL_FILENAME = "sam3.1_multiplex.pt"


def check_python_version():
    """Require the Python version supported by this notebook's pinned NumPy."""
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            "This notebook requires Python 3.12. In Colab, choose Runtime > Change runtime type > "
            "Runtime Version and select Python 3.12, then rerun section 1 and restart the session."
        )


def verify_sam_revision():
    """Run the installation cell if the installed SAM source does not match this notebook."""
    try:
        direct_url = json.loads(metadata.distribution("sam3").read_text("direct_url.json") or "{}")
        revision = direct_url.get("vcs_info", {}).get("commit_id")
    except (metadata.PackageNotFoundError, ValueError):
        revision = None
    if revision != SAM_REVISION:
        raise RuntimeError("The installed SAM source revision differs. Run section 1, restart the session, and continue at section 2.")
    return revision


def check_environment():
    """Use a CUDA GPU with native BF16 support and a compatible PyTorch build."""
    check_python_version()
    import torch
    from packaging.version import Version

    if Version(torch.__version__.split("+")[0]) < Version("2.7"):
        raise RuntimeError("Use PyTorch 2.7 or later with a matching torchvision build.")
    if not torch.cuda.is_available():
        raise RuntimeError("Select Runtime > Change runtime type > GPU, then reconnect.")
    if Version(torch.version.cuda or "0") < Version("12.6"):
        raise RuntimeError("Use a PyTorch CUDA build of version 12.6 or later.")
    if torch.cuda.get_device_capability()[0] < 8 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This notebook requires native BF16 support. Select an A100 or another compatible GPU; T4 is not supported.")
    verify_sam_revision()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "python": sys.version.split()[0], "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
        "free_vram_gib": round(free_bytes / 2**30, 2),
        "total_vram_gib": round(total_bytes / 2**30, 2),
        "sam_revision": SAM_REVISION,
    }


def adapt_session_arguments(predictor):
    """Keep frame offloading enabled; state offloading is unavailable for this model."""
    original = predictor.model.init_state
    parameters = inspect.signature(original).parameters
    if "offload_state_to_cpu" in parameters or any(
        item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters.values()
    ):
        return predictor

    @wraps(original)
    def initialize(*args, offload_state_to_cpu=False, **kwargs):
        if offload_state_to_cpu:
            raise ValueError("SAM 3.1 does not support CPU offloading of tracker state in this notebook.")
        return original(*args, **kwargs)

    predictor.model.init_state = initialize
    return predictor


def adapt_point_output(predictor):
    """Let SAM merge refined masks when the first point prompt has no cached frame."""
    model = predictor.model
    original = model._build_sam2_output
    if getattr(original, "_first_point_output_fixed", False):
        return predictor

    @wraps(original)
    def build_output(inference_state, frame_idx, refined_obj_id_to_mask=None):
        cached_outputs = inference_state["cached_frame_outputs"]
        if frame_idx in cached_outputs or not refined_obj_id_to_mask:
            return original(inference_state, frame_idx, refined_obj_id_to_mask)

        patched_state = inference_state.copy()
        patched_outputs = cached_outputs.copy()
        patched_outputs[frame_idx] = {}
        patched_state["cached_frame_outputs"] = patched_outputs
        return original(patched_state, frame_idx, refined_obj_id_to_mask)

    build_output._first_point_output_fixed = True
    model._build_sam2_output = build_output
    return predictor


def clean_token(token):
    """Remove surrounding whitespace; enter a token without embedded spaces or control characters."""
    if token is None:
        return None
    if not isinstance(token, str):
        raise ValueError("The Hugging Face token must be text.")
    cleaned = token.strip()
    if not cleaned or any(ord(character) <= 32 or ord(character) >= 127 for character in cleaned):
        raise ValueError("Enter a nonempty Hugging Face token without embedded whitespace or non-ASCII characters.")
    return cleaned


def load_predictor(max_objects=16, token=None, model_revision=None):
    """Download approved weights using your token without storing it in the notebook."""
    if isinstance(max_objects, bool) or not isinstance(max_objects, int) or not 1 <= max_objects <= 256:
        raise ValueError("Choose an object limit between 1 and 256; larger limits require more memory.")
    token = clean_token(token)
    environment = check_environment()
    from huggingface_hub import HfApi, hf_hub_download
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    revision = HfApi().model_info(MODEL_REPOSITORY, revision=model_revision, token=token).sha
    checkpoint = hf_hub_download(
        MODEL_REPOSITORY, MODEL_FILENAME, revision=revision, token=token,
    )
    predictor = build_sam3_multiplex_video_predictor(
        checkpoint_path=checkpoint, max_num_objects=max_objects,
        use_fa3=False, compile=False, warm_up=False, async_loading_frames=False,
    )
    adapt_session_arguments(predictor)
    adapt_point_output(predictor)
    environment.update({"checkpoint_revision": revision, "checkpoint_file": Path(checkpoint).name,
                        "max_objects": max_objects, "flash_attention_3": False, "compile": False})
    return predictor, environment
