# SAM3 Video Segmentation Notebook

This repository contains a Google Colab notebook for performing segmentation of live-cell imaging data using Meta's **Segment Anything Model 3 (SAM 3)**.

## 🚀 Overview

The notebook [`SAM3-video-segmentation_v0.ipynb`](SAM3-video-segmentation_v0.ipynb) allows you to:

1.  Load a sequence of video frames (JPEGs).
2.  Use SAM 3 to segment objects based on text prompts (e.g., "cell").
3.  Visualize and save the segmentation masks.

## 🔑 Prerequisites & Setup (IMPORTANT)

To use SAM 3, you **must** have a HuggingFace account and an access token with permission to access the SAM 3 model weights.

### 1. Get HuggingFace Access Token

1.  **Create an Account**: If you don't have one, sign up at [huggingface.co](https://huggingface.co/).
2.  **Request Access**: Visit the official [Facebook SAM 3 repository](https://huggingface.co/facebook/sam3) and accept the license terms to request access to the model.
3.  **Generate Token**:
    - Go to your [Settings > Access Tokens](https://huggingface.co/settings/tokens).
    - Click **"Create new token"**.
    - Give it a name (e.g., "colab-sam3") and ensure it has **Read** permissions (or "Fine-grained" with access to the repos you need).
    - Copy the token string (it starts with `hf_...`).

### 2. Configure Google Colab

1.  Open the notebook in Google Colab.
2.  In the left sidebar, click on the **Secrets** icon (🔑).
3.  Add a new secret:
    - **Name**: `HF_TOKEN`
    - **Value**: Paste your HuggingFace token here.
4.  Toggle the "Notebook access" switch to **On** for this secret.

## 📂 Usage

1.  **Prepare Input**:

    - Create a folder named `input/` in the same directory as the notebook (or adjust the path in the notebook).
    - Place your video frames in this folder as JPEG files, named sequentially (e.g., `000.jpg`, `001.jpg`, `002.jpg`...).

2.  **Run the Notebook**:
    - Execute the cells in order.
    - **Environment Setup**: Handles library installation and authentication.
    - **Configuration**: Loads the SAM 3 model.
    - **Data Loading**: Loads your frames.
    - **Inference**:
      - Enter a **Text Prompt** (e.g., "cell", "nucleus") to define what you want to segment.
      - The notebook will perform segmentation on the reference frame and then propagate it through the video.
    - **Save Results**: The output masks and merged images will be saved to the `output_mask/` and `output_merged/` directories respectively.

## 📝 Credits

- **Author**: Dr. Yusuke Kimata (ORCID: https://orcid.org/0000-0002-1366-0636)
- **References**:
  - [Meta Research SAM 3](https://github.com/facebookresearch/sam3)
  - [Roboflow Notebook](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-segment-videos-with-segment-anything-3.ipynb)
