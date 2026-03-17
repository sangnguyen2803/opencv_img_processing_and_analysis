


# opencv_img_processing_and_analysis

This repository contains multiple projects related to image processing, computer vision, and video analysis using OpenCV and Python. Each subproject is self-contained and focuses on a specific application or technique. Below you will find detailed descriptions, requirements, and usage instructions for each project.

---

## Table of Contents

- [Grain-Counting](#grain-counting)
- [HTR-pipeline (Handwritten Text Recognition)](#htr-pipeline-handwritten-text-recognition)
- [SIFT-RANSAC Case Studies](#sift-ransac-case-studies)
- [Video Analysis - Optical Flow](#video-analysis---optical-flow)
- [General Setup](#general-setup)
- [License](#license)

---

## Grain-Counting

**Folder:** `Grain-Counting/Counting Rice Grains-20260316/`

Scripts and resources for counting rice grains in images using image processing techniques. This project demonstrates segmentation, contour detection, and counting of rice grains from static images. 

**Typical workflow:**
- Prepare input images of rice grains.
- Run the provided scripts to segment and count grains.
- Review output images and count statistics.

> See the project folder for scripts and sample data.

---

## HTR-pipeline (Handwritten Text Recognition)

**Folder:** `HTR-pipeline/`

Pipeline for handwritten character recognition, including segmentation, feature extraction, model training, and benchmarking.

### Features
- Segments individual characters from images.
- Extracts features for machine learning models.
- Trains and evaluates SVM and Random Forest classifiers.
- Includes benchmarking and augmentation utilities.

### Project Structure
- `main.py`: Entry point for running the pipeline.
- `src/`: Source code for all major components:
  - `augment.py`: Data augmentation helpers
  - `features.py`: Feature extraction
  - `segmentation.py`: Character segmentation
  - `train.py`: Model training
  - `recognize.py`: Inference/recognition
  - `benchmark.py`: Evaluation and benchmarking
- `models/`: Pre-trained models (`rf_model.joblib`, `svm_model.joblib`, `svm_scaler.joblib`)
- `data/`: Character images, datasets, and raw data
- `segmented_chars/`: Output of segmentation routines
- `benchmark_results.csv`, `dataset_features.csv`: Example outputs

### Requirements
- Python 3.8+
- `numpy`, `scikit-learn`, `joblib`, `opencv-python`, `pillow`, `pandas`, `matplotlib`, `scikit-image`

### Installation
```bash
pip install opencv-python numpy scikit-image scikit-learn pandas joblib matplotlib
```

### Usage
Prepare your data under `data/`, then run:
```bash
python main.py
```
Check script headers and the `src/` folder for more options and arguments.

### Benchmarking
Run `src/benchmark.py` to evaluate model accuracy and feature extraction. Results are saved to `benchmark_results.csv`.

---

## SIFT-RANSAC Case Studies

**Folder:** `SIFT-RANSAC/`

Scripts and case studies demonstrating SIFT feature detection, description, and RANSAC-based matching for image analysis.

### Contents
- `case_study1_p1.py`, `case_study1_p2.py`, `case_study2.py`, `case_study3.py`: Python scripts for each case study
- `images/`: Input images
- `case_study1_output/`: Output results for case study 1
- `reports/`: PDF reports and documentation

### Prerequisites
- Python 3.8+
- `opencv-contrib-python` (for SIFT), `numpy`, `matplotlib`, `scikit-image`

### Quick Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate
python -m pip install --upgrade pip
pip install opencv-contrib-python numpy matplotlib scikit-image
```

### How to Run
```powershell
python .\case_study1_p1.py
python .\case_study1_p2.py
python .\case_study2.py
python .\case_study3.py
```
Scripts read images from `images/` and output results to respective folders.

---

## Video Analysis - Optical Flow

**Folder:** `Video Analysis - Optical Flow/`

Project for analyzing tennis videos using optical flow and motion analysis to detect and rank effective play shots.

### Key Features
- Shot boundary detection using mean absolute frame difference
- Per-shot feature extraction (court ratio, player motion, etc.)
- Ranking of approach-to-net shots
- Output includes CSV summaries, visualizations, and individual shot video clips

### Project Structure
- `report.ipynb`: Main analysis notebook
- `video_files/`: Input video files
- `plots/`: Visualizations (e.g., shot boundary detection)
- `results/`: CSV files with shot summaries and rankings
- `shots_output/`: Individual video clips for each detected shot

### Main Functions (in notebook/scripts)
- `mean_abs_diff_between_frames()`, `detect_shot_boundaries()`, `plot_meandiff_func()`
- `estimate_court_hue()`, `court_ratio_from_hue()`, `analyze_shot()`
- `centroid_track_mog2()`, `features_maxadvance_and_center()`, `zscore()`, `sigmoid()`

### Requirements
- Python 3.x
- `opencv-python`, `numpy`, `pandas`, `matplotlib`

---

## General Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/opencv_img_processing_and_analysis.git
   ```
2. **Navigate to the desired project folder** and follow the specific README or instructions inside each project for setup and usage.
3. **Install dependencies** as described above for each project.

---

## License
This repository is for educational and research purposes. See individual project folders for more details.

## Author
- [Your Name]

---
For more details, refer to the README files inside each project folder.
