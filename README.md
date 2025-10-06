# 3D Depth Estimation

This project estimates the **3D distance (depth)** of cars from 2D images using the **KITTI dataset**. It combines object detection with geometric modeling based on intrinsic camera parameters to approximate the real-world distance of vehicles from a monocular camera.

---

## Overview

The system uses a **YOLOv8x** model for detecting cars and a **geometrical approach** for calculating their distances from the camera.  
Each detected bounding box is processed to estimate depth using the intrinsic matrix from the KITTI calibration data.  
This approach simulates how autonomous vehicles can perceive depth without stereo or LiDAR sensors. 

## Methodology

### 1. Object Detection
- **Model:** YOLOv8x (Ultralytics)
- **Input:** KITTI RGB images  
- **Output:** 2D bounding boxes of detected cars  

### 2. Depth Estimation
- The **bottom midpoint** of each detected bounding box is used as the reference point.  
- The **intrinsic camera matrix** transforms this 2D point into a 3D direction vector.  
- The vector is extended to intersect with a **ground plane** at a fixed height (≈ 1.65 m).  
- The **Euclidean distance** between the camera and this intersection point gives the estimated depth.

### 3. Matching Detections with Ground Truth
- Each detection is matched to a ground truth box using **Intersection over Union (IoU)**.  
- Detections with low IoU are treated as false positives.  

---

## Results  

### Successful Cases
- Accurate predictions when:
  - Cars are fully visible and not occluded.  
  - Lighting conditions are stable.  
  - Ground is relatively flat.  
- Predicted distances closely match ground truth values.

### Problematic Cases
- **Occlusions:** Partial visibility leads to incomplete bounding boxes.  
- **Detection Errors:** False positives or missed detections impact depth accuracy.  
- **Lighting Issues:** Shadows and low light reduce detection quality.

---

## Dataset

- **Dataset:** KITTI  
  - Includes RGB images, calibration data, and ground truth labels.  
  - A subset is used for testing and validation.  

---

## Conclusion

- This project shows how 2D detections and camera geometry can work together for 3D depth estimation.
- Although results are promising, real-world challenges like occlusion, uneven terrain, and lighting variations still affect accuracy.
