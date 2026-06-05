# Stereo Vision System for 3D Reconstruction Using SIFT Features

## Overview

This project focuses on the implementation of a simple stereo vision system for three-dimensional scene reconstruction. The objective is to recover the 3D coordinates of points observed from two different camera viewpoints and generate a 3D point cloud representation of the scene.

The experiment is based on a controlled setup composed of three boxes with known dimensions placed on a planar surface. Using stereo image acquisition, camera calibration, feature matching, and triangulation techniques, the system reconstructs the spatial structure of the scene.

This project was developed as part of the **Computer Vision** course in the **Master's Program in Visual Computing (MIV)** at USTHB.

---

## Objectives

The main goals of this project are:

* Understand the principles of stereo vision and 3D reconstruction.
* Perform accurate camera calibration.
* Acquire stereo image pairs using controlled camera translation.
* Detect and match visual features using the SIFT algorithm.
* Estimate the 3D coordinates of matched points through triangulation.
* Generate and visualize a 3D point cloud of the reconstructed scene.
* Validate the reconstructed dimensions using known object measurements.

---

## Experimental Setup

The experimental scene consists of:

* Three boxes with different known dimensions.
* A planar support surface.
* A calibrated camera.
* Two images captured after a translational camera movement.

The dimensions and positions of the boxes are known beforehand, allowing quantitative evaluation of reconstruction accuracy.

---

## Methodology

### 1. Scene Preparation

A controlled 3D scene is created using three boxes of different sizes placed on a flat surface.

Requirements:

* Known dimensions for all objects.
* Stable lighting conditions.
* Minimal shadows and reflections.
* Fixed object positions during acquisition.

This setup provides a reliable environment for stereo reconstruction experiments.

---

### 2. Camera Calibration

Camera calibration is performed to estimate the intrinsic parameters necessary for 3D reconstruction.

The calibration process computes:

* Camera matrix.
* Focal length.
* Principal point.
* Distortion coefficients.

Typical calibration workflow:

* Capture multiple images of a checkerboard pattern.
* Detect calibration corners.
* Estimate intrinsic and extrinsic parameters.
* Evaluate reprojection error.

Calibration accuracy directly influences depth estimation quality.

---

### 3. Stereo Image Acquisition

Two images of the scene are captured while maintaining the same camera orientation and applying a translation parallel to the observation plane.

Acquisition constraints:

* Pure translational motion.
* Constant focal length.
* Fixed exposure settings.
* Significant overlap between images.

The resulting image pair forms the basis of the stereo vision system.

---

### 4. SIFT Feature Detection

Feature extraction is performed using the Scale-Invariant Feature Transform (SIFT).

For each image, SIFT detects:

* Keypoint locations.
* Characteristic scales.
* Dominant orientations.
* Feature descriptors.

Advantages of SIFT include:

* Scale invariance.
* Rotation invariance.
* Robustness to illumination changes.
* High matching reliability.

---

### 5. Feature Matching

SIFT descriptors extracted from both images are matched to identify corresponding points.

The matching procedure includes:

* Nearest-neighbor descriptor search.
* Euclidean distance comparison.
* Lowe's ratio test.
* Outlier rejection.

Only reliable correspondences are retained for reconstruction.

---

### 6. Epipolar Geometry Estimation

The geometric relationship between both camera views is modeled through:

* Fundamental Matrix (F).
* Essential Matrix (E).

These matrices:

* Describe stereo geometry.
* Constrain point correspondences.
* Eliminate incorrect matches.
* Improve triangulation accuracy.

Epipolar geometry is a critical step in stereo vision pipelines.

---

### 7. 3D Point Reconstruction

The 3D coordinates of matched points are estimated through triangulation.

For every pair of corresponding image points:

* Projection rays are computed.
* Their spatial intersection is estimated.
* A 3D point is reconstructed.

The output consists of:

* X coordinate.
* Y coordinate.
* Z coordinate (depth).

This process generates the three-dimensional structure of the scene.

---

### 8. Point Cloud Generation

All reconstructed points are merged into a single 3D point cloud.

The point cloud represents:

* The geometric shape of the boxes.
* Their relative positions.
* Depth variations within the scene.

Point cloud density depends on:

* Number of SIFT features.
* Matching accuracy.
* Scene texture richness.

---

### 9. 3D Visualization

The reconstructed point cloud is visualized using dedicated 3D visualization software.

Possible visualization tools include:

* Open3D
* MeshLab
* CloudCompare
* MATLAB
* Matplotlib 3D

Visualization allows qualitative inspection of reconstruction quality and depth estimation.

---

## Experimental Evaluation

The stereo vision system is evaluated using:

* Number of detected keypoints.
* Number of valid correspondences.
* Reprojection error.
* Reconstruction accuracy.
* Depth estimation precision.
* Comparison between reconstructed and real object dimensions.

The known dimensions of the boxes provide a reference for quantitative validation.

---

## Expected Results

The implemented system should successfully produce:

* Reliable SIFT feature correspondences.
* Accurate camera calibration parameters.
* Correct stereo geometry estimation.
* Precise 3D point triangulation.
* A coherent 3D point cloud representation.

The reconstructed scene should preserve:

* Relative object positions.
* Object dimensions.
* Spatial depth information.

---

## Technologies Used

* Python
* OpenCV
* NumPy
* SIFT
* Stereo Vision
* Epipolar Geometry
* Triangulation
* Open3D
* MeshLab
* Matplotlib

---



---

## Key Contributions

* Complete stereo vision reconstruction pipeline.
* Camera calibration and distortion correction.
* SIFT-based feature extraction and matching.
* Epipolar geometry estimation.
* 3D point triangulation.
* Point cloud generation and visualization.
* Validation using known object dimensions.

---

## Future Work

Potential extensions include:

* Dense stereo matching techniques.
* Real-time stereo reconstruction.
* Multi-view reconstruction systems.
* Bundle adjustment optimization.
* Deep learning-based feature matching.
* Surface reconstruction from point clouds.
* 3D object recognition and scene understanding.

---

## Author

**Ghemmour Lehna**  
Master's Student in Visual Computing (MIV)  
University of Science and Technology Houari Boumediene (USTHB)  
Academic Year 2025–2026
