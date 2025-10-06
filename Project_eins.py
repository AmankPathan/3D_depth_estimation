from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import csv

# Load the YOLO model
model = YOLO('yolov8x.pt')

# Paths
image_dir = r'C:\Users\Aman\Desktop\1st sem\Computer Vision\Projects\KITTI_Selection\images'  # Replace with the path to your image directory
ground_truth_dir = r'C:\Users\Aman\Desktop\1st sem\Computer Vision\Projects\KITTI_Selection\labels'  # Replace with the path to your ground truth directory
output_dir = r'C:\Users\Aman\Desktop\1st sem\Computer Vision\Projects\KITTI_Selection\p_r_output'  # Directory to save annotated images
calib_dir = r'C:\Users\Aman\Desktop\1st sem\Computer Vision\Projects\KITTI_Selection\calib'  # Path to the calibra

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load images from a directory
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

def read_intrinsic_matrix(calib_file):
    with open(calib_file, 'r') as f:
        lines = f.readlines()
        K = np.array([list(map(float, line.strip().split())) for line in lines])
    return K

def calculate_distance(midpoint, K, camera_height):
    midpoint_homogeneous = np.array([midpoint[0], midpoint[1], 1.0])
    direction_vector = np.linalg.inv(K).dot(midpoint_homogeneous)
    direction_vector /= direction_vector[2]
    scale = -camera_height / direction_vector[1]
    ground_point = direction_vector * scale
    distance = np.linalg.norm(ground_point[:3])
    return distance

def calculate_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    iou = interArea / float(box1Area + box2Area - interArea)
    return iou

# CSV file to save results
csv_file = os.path.join(output_dir, 'car_distances.csv')
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image Name', 'Car ID', 'YOLO Distance (m)', 'GT Distance (m)', 'Precision', 'Recall'])

    for image_path in image_files:
        image = cv2.imread(image_path)

        if image is None:
            print(f"Image file not found or could not be read: {image_path}")
            continue

        image_name = os.path.splitext(os.path.basename(image_path))[0]
        calib_file = os.path.join(calib_dir, f'{image_name}.txt')

        if not os.path.exists(calib_file):
            print(f"Calibration file not found for image: {image_path}")
            continue

        K = read_intrinsic_matrix(calib_file)
        camera_height = 1.65

        results = model(image)

        ground_truth_path = os.path.join(ground_truth_dir, f'{image_name}.txt')
        if os.path.exists(ground_truth_path):
            with open(ground_truth_path, 'r') as f:
                ground_truth_boxes = [line.strip().split() for line in f.readlines()]
        else:
            ground_truth_boxes = []

        gt_boxes = []
        gt_distances = []
        for gt_box in ground_truth_boxes:
            if gt_box[0] == 'Car':
                x1, y1, x2, y2, gt_distance = map(float, gt_box[1:])
                gt_boxes.append([x1, y1, x2, y2])
                gt_distances.append(gt_distance)

        for gt_box in gt_boxes:
            x1, y1, x2, y2 = gt_box
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        TP, FP, FN = 0, 0, len(gt_boxes)
        car_id = 0

        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                if model.names[class_id] == 'car':
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    midpoint = ((x1 + x2) / 2, y2)
                    yolo_distance = calculate_distance(midpoint, K, camera_height)

                    gt_distance = None
                    for i, gt_box in enumerate(gt_boxes):
                        if calculate_iou([x1, y1, x2, y2], gt_box) > 0.5:
                            gt_distance = gt_distances[i]
                            TP += 1
                            FN -= 1
                            break
                    else:
                        FP += 1

                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

                    # YOLO label placement (above or below depending on space)
                    yolo_text_y = y1 - 20 if y1 - 20 > 10 else y1 + 20  # Ensure text is not outside the image
                    cv2.putText(image, f'{yolo_distance:.2f}m', (x1, yolo_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    # GT label placement (fixed 30 pixels below YOLO label)
                    gt_text_y = yolo_text_y + 30  # Fixed offset from YOLO label
                    if gt_distance is not None:
                        cv2.putText(image, f'{gt_distance:.2f}m', (x1, gt_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    writer.writerow([os.path.basename(image_path), car_id, yolo_distance, gt_distance, None, None])
                    car_id += 1

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        text_image = np.zeros((50, image.shape[1], 3), dtype=np.uint8)
        text = f'File: {os.path.basename(image_path)}, Precision: {precision:.2f}, Recall: {recall:.2f}'
        cv2.putText(text_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        combined_image = np.vstack((text_image, image))
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, combined_image)

        combined_image_rgb = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)
        plt.imshow(combined_image_rgb)
        plt.axis('off')
        plt.show()