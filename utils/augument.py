import os
import json
import cv2
import numpy as np
import random

# Constants
dataset_path = "/workspace/data"
ok_images_path = os.path.join(dataset_path, "crop_image", "oblique2")
ng_images_path = os.path.join(dataset_path, "substance", "single", "oblique2_coco")
annotations_file_path = os.path.join(ng_images_path, "annotations.json")
output_dir = os.path.join(dataset_path, "dataset", "oblique_crop3", "train")
os.makedirs(output_dir, exist_ok=True)

# Anomaly insertion parameters
# Z
# insertion_x_range = (22, 1000)
# insertion_y_range = (400, 620)
# # bead
# insertion_x_range = (0, 240)
# insertion_y_range = (0, 190)
# oblieque1
# insertion_x_range = (50, 690)
# insertion_y_range = (50, 250)
# oblieque2
insertion_x_range = (50, 1050)
insertion_y_range = (10, 250)

min_intensity_difference = 5
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

def extract_anomalies(annotations, ng_images_path):
    anomalies = []
    for annotation in annotations["annotations"]:
        image_info = next((image for image in annotations["images"] if image["id"] == annotation["image_id"]), None)
        image_path = os.path.join(ng_images_path, image_info["file_name"])
        image = cv2.imread(image_path)
        if image is not None:
            x, y, w, h = annotation["bbox"]
            x = int(x); y = int(y); w = int(w); h = int(h)
            anomaly = image[y:y + h, x:x + w]
            # cv2.rectangle(image, (x, y), (x+w, y+h), (125, 0, 0), 1)
            # cv2.imwrite("/workspace/check.png", image)
            # cv2.imwrite("/workspace/check2.png", image[y:y + h, x:x + w])
            # if image_info["id"] == 24:
            #     exit()
            anomalies.append(anomaly)
    print(f"Extracted {len(anomalies)} anomalies.")
    return anomalies

def is_suitable_location(image, anomaly, x, y):
    height, width = image.shape[:2]
    # print(anomaly.shape, x, y, width, height)
    if x + anomaly.shape[1] > width or y + anomaly.shape[0] > height:
        return False

    surrounding_area = image[y:y + anomaly.shape[0], x:x + anomaly.shape[1]]
    flag = np.mean(anomaly) - np.mean(surrounding_area) > min_intensity_difference
    return flag

def insert_anomaly(image, anomalies):
    cnt = 0
    bboxes = []
    # for anomaly in random.sample(anomalies, len(anomalies)):
    for anomaly in random.choices(anomalies, k=100):
        x = random.randint(*insertion_x_range)
        y = random.randint(*insertion_y_range)
        if is_suitable_location(image, anomaly, x, y):
            image[y:y + anomaly.shape[0], x:x + anomaly.shape[1]] = anomaly
            bboxes.append((x, y, x + anomaly.shape[1], y + anomaly.shape[0]))
            cnt += 1
        if cnt == 20:
            return image, bboxes
    return None, None

def process_image(image_file, anomalies, index):
    image_path = os.path.join(ok_images_path, image_file)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_file}")
        return None, None

    modified_image, bboxes = insert_anomaly(image, anomalies)
    if modified_image is None:
        print(f"Failed to insert anomaly in image: {image_file}")
        return None, None

    return modified_image, bboxes

def main():
    annotations = load_json(annotations_file_path)
    anomalies = extract_anomalies(annotations, ng_images_path)

    coco_data = init_json()

    for index, image_file in enumerate(os.listdir(ok_images_path)):
        print(image_file)
        modified_image, bboxes = process_image(image_file, anomalies, index)
        if modified_image is None:
            continue

        output_filename = f"{index}.png"
        cv2.imwrite(os.path.join(output_dir, output_filename), modified_image)

        image_info = {
                "file_name": output_filename,
                "height": modified_image.shape[0],
                "width": modified_image.shape[1],
                "id": index
            }
        coco_data["images"].append(image_info)
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            
            annotation = {
                "num_keypoints": 0,
                "keypoints": [],
                "segmentation": [[x1, y1, x1, y2, x2, y2, x2, y1]],
                "iscrowd": 0,
                "area": (x2 - x1) * (y2 - y1),
                "image_id": index,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "category_id": 1,
                "id": index * 100 + i,  # Unique ID for each annotation
                "attributes": {},
                "rotation": 0
            }

            coco_data["annotations"].append(annotation)

    with open(os.path.join(output_dir, "annotations.json"), "w") as file:
        json.dump(coco_data, file, indent=2)

if __name__ == "__main__":
    main()