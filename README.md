# Overview

### 1. Line Fitting
Showing the use of RANSAC (RANdom SAmple Consensus) for robust model fitting. Working on the simple case of 2D line estimation.
### 2. Multi-View Stereo
Given a collection of images with known camera parameters, multi-view stereo (MVS) describes the task of reconstructing the dense geometry of the observed scene.
Trying to solve the multi-view stereo problem with deep learning. First, we estimate the depth maps for each view of the scene. Then we reconstruct a point cloud with all the depth maps and the filtering techniques.

## Environment Setup

To install and activate the environment, run:
```
conda env create -f env.yaml
conda activate cv-mvs
pip install plyfile open3d
```

## Line Fitting

<img width="400" alt="ransac" src="https://user-images.githubusercontent.com/47776895/144763238-cd0097bc-452c-4196-9ae8-25836b4ba011.png">
