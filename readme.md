# TikTok Tech Jam

This repository contains code for the TikTok Tech Jam project.

## Installation

1. Install dependencies:
    Create your venv and install the dependencies
    ```bash
    pip install -r requirements.txt
    ```

## Running the Python File

To run the main Python file:
```bash
streamlit run privacy_gallery_yolo_und18.py
```

## Params to change if needed:

### 1. in ```is_minor``` function 

``` p_minor_threshold ``` = how much probability mass must be in the “under 18” buckets to count as a minor.

If p_minor_threshold = 0.60 → classified as minor ✅

If p_minor_threshold = 0.70 → classified as adult ❌ (since 0.63 < 0.70)

I used 0.40 to blur uncertain cases (safer but more false positive, some adults could be blurred)