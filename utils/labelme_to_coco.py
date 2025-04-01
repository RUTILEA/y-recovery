import os
import json
import glob
from PIL import Image
import numpy as np
from shapely.geometry import Polygon
import cv2

def create_category_dict(labels):
    category_dict = {}
    categories = []
    for i, label in enumerate(labels):
        category = {
            'id': i + 1,
            'name': label,
            'supercategory': label
        }
        category_dict[label] = i + 1
        categories.append(category)
    return category_dict, categories

def create_annotation_info(ann_id, image_id, category_id, segmentation, area, bbox, iscrowd):
    annotation_info = {
        'id': ann_id,
        'image_id': image_id,
        'category_id': category_id,
        'segmentation': segmentation,
        'area': area,
        'bbox': bbox,
        'iscrowd': iscrowd
    }
    return annotation_info

def create_image_info(image_id, file_name, image_size):
    image_info = {
        'id': image_id,
        'file_name': file_name,
        'width': image_size[0],
        'height': image_size[1]
    }
    return image_info

def convert_labelme_to_coco(labelme_folder, output_file):
    label_files = glob.glob(os.path.join(labelme_folder, '*.json'))
    coco_output = {
        'images': [],
        'annotations': [],
        'categories': []
    }

    category_set = {"substance": 1}
    image_id = 0
    ann_id = 0

    for label_file in label_files:
        print(label_file)
        with open(label_file, 'r') as f:
            label_data = json.load(f)

        # Image info
        # image_path = label_data['imagePath'][9:]
        image_path = label_data['imagePath']
        img = Image.open(os.path.join(labelme_folder, image_path))
        image_info = create_image_info(image_id, image_path, img.size)
        coco_output['images'].append(image_info)

        # Annotations
        for shape in label_data['shapes']:
            points = shape['points']
            (x1, y1), (x2, y2) = points
            x1 = int(x1); x2 = int(x2); y1 = int(y1); y2 = int(y2)
            # image_w = img.width; image_h = img.height
            # rw = image_w/1024; rh = image_h/85; x0 = -4; y0 = 4
            # x1 = int(x1*rw)+x0; x2 = int(x2*rw)+x0; y1 = int(y1*rh)+y0; y2 = int(y2*rh)+y0
            # import cv2
            image = np.array(img)
            # cv2.rectangle(image, (x1, y1), (x2, y2), (125, 0, 0), 1)
            # cv2.imwrite("/workspace/check.png", image)
            label = shape['label']
            polygon = Polygon([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            segmentation = [list(np.asarray(polygon.exterior.coords).ravel())]

            # Compute area, bbox, and create annotation
            area = polygon.area
            bbox = polygon.bounds
            # print(bbox)
            bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
            annotation_info = create_annotation_info(
                ann_id, image_id, category_set[label], segmentation, area, bbox, iscrowd=0
            )
            coco_output['annotations'].append(annotation_info)
            ann_id += 1

        image_id += 1

    # Categories
    category_dict, categories = create_category_dict(["substance"])
    coco_output['categories'] = categories

    # Save to output file
    with open(output_file, 'w') as output_json_file:
        json.dump(coco_output, output_json_file, indent=4)

if __name__ == '__main__':
    labelme_folder = '/workspace/data/substance/single/oblique2_json'
    output_file = '/workspace/data/substance/single/oblique2_coco/annotations.json'
    convert_labelme_to_coco(labelme_folder, output_file)
    print(f'COCO JSON saved to {output_file}')