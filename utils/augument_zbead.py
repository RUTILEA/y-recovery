import os
import json
import cv2
import numpy as np
import random

# Constants
dataset_path = "/workspace/data"
# ok_images_path = os.path.join(dataset_path, "crop_image", "Z_500_2")
ng_images_path = os.path.join(dataset_path, "substance", "single", "Z_bead2_coco")
annotations_file_path = os.path.join(ng_images_path, "annotations.json")
output_dir = os.path.join(dataset_path, "dataset", "Z_bead4", "train")
os.makedirs(output_dir, exist_ok=True)

# Anomaly insertion parameters
# Z
# insertion_x_range = (22, 1000)
# insertion_y_range = (400, 620)
# # bead
# insertion_x_range = (0, 240)
insertion_x_range = (30, 220)
insertion_y_range = (0, 190)
# oblieque1
# insertion_x_range = (50, 690)
# insertion_y_range = (50, 250)
# oblieque2
# insertion_x_range = (50, 1050)
# insertion_y_range = (20, 20)

min_intensity_difference = -255
max_insertion_attempts = 10

def load_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

def init_json():
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{
            "skelton": [],
            "keypoints": [],
            "color": "#FF0000",
            "supercategory": "substance",
            "id": 1,
            "name": "substance",
            "keypoints_colors": []
        }]
    }

    return coco_data

def is_suitable_location(image, anomaly, x, y):
    height, width = image.shape[:2]
    if x + anomaly.shape[1] > width or y + anomaly.shape[0] > height:
        return False

    surrounding_area = image[y:y + anomaly.shape[0], x:x + anomaly.shape[1]]
    flag = np.mean(anomaly) - np.mean(surrounding_area) > min_intensity_difference

    return flag


def crop_image(img):
    height, width = img.shape[:2]
    center_x, center_y = width//2, height//2
    left_bead = img[center_y-250//2+30: \
                    center_y-250//2+220, \
                    center_x-1000//2+80: \
                    center_x-1000//2+320]
    right_bead = img[center_y-250//2+30: \
                    center_y-250//2+220, \
                    center_x-1000//2+680: \
                    center_x-1000//2+920]
    return [left_bead, right_bead]

def extract_anomaly(image, annotation):
    x, y, w, h = annotation["bbox"]
    x = int(x); y = int(y); w = int(w); h = int(h)
    anomaly = image[y:y + h, x:x + w]
    return anomaly, x, y

def insert_anomaly(image, anomaly, h, j):
    cnt = 0
    bboxes = []
    for _ in range(100):
        x = random.randint(*insertion_x_range)
        y = h-(1024//2-250//2+30)
        if is_suitable_location(image, anomaly, x, y):
            image[y:y + anomaly.shape[0], x:x + anomaly.shape[1]] = anomaly
            bboxes.append((x, y, x + anomaly.shape[1], y + anomaly.shape[0]))
            cnt += 1
        if cnt == 20:
            return image, bboxes
    print(f"Failed to insert anomaly {j}.")
    return image, bboxes
    

def main():
    import glob
    annotations = load_json(annotations_file_path)
    n = len(annotations["images"])
    coco_data = init_json()
    
    for i in range(n):
        image_path = os.path.join(ng_images_path, annotations["images"][i]["file_name"])
        image = cv2.imread(image_path)
        annotation = annotations["annotations"][i]
        anomaly, pos_x, pos_h = extract_anomaly(image, annotation)
        crop_images = crop_image(image)
        for j, crop_img in enumerate(crop_images):
            if j == 0 and (pos_x < 1024//2-1000//2+80 or pos_x > 1024//2-1000//2+320): continue
            if j == 1 and (pos_x < 1024//2-1000//2+680 or pos_x > 1024//2-1000//2+920): continue
            new_image, bboxes = insert_anomaly(crop_img, anomaly, pos_h, j)
            if j == 0:
                bboxes.append((pos_x-(1024//2-1000//2+80), pos_h, pos_x-(1024//2-1000//2+80) + anomaly.shape[1], pos_h + anomaly.shape[0]))
            elif j == 1:
                bboxes.append((pos_x-(1024//2-1000//2+680), pos_h, pos_x-(1024//2-1000//2+680) + anomaly.shape[1], pos_h + anomaly.shape[0]))
            if new_image is None:
                continue

            output_filename = f"{2*i+j}.png"
            cv2.imwrite(os.path.join(output_dir, output_filename), new_image)

            image_info = {
                    "file_name": output_filename,
                    "height": new_image.shape[0],
                    "width": new_image.shape[1],
                    "id": 2*i+j
                }
            coco_data["images"].append(image_info)
            for k, bbox in enumerate(bboxes):
                x1, y1, x2, y2 = bbox
                
                annotation = {
                    "num_keypoints": 0,
                    "keypoints": [],
                    "segmentation": [[x1, y1, x1, y2, x2, y2, x2, y1]],
                    "iscrowd": 0,
                    "area": (x2 - x1) * (y2 - y1),
                    "image_id": 2*i+j,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "category_id": 1,
                    "id": (2*i+j) * 100 + k,  # Unique ID for each annotation
                    "attributes": {},
                    "rotation": 0
                }
                coco_data["annotations"].append(annotation)
    
    with open(os.path.join(output_dir, "annotations.json"), "w") as file:
        json.dump(coco_data, file, indent=2)

    
if __name__ == "__main__":
    main()