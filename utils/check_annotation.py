import os
import json
import glob
from PIL import Image
import numpy as np
import cv2

data_dir = "/workspace/data/substance/single/Z_bead_json/"
output_dir = "/workspace/data/substance/single/Z_bead_annotation_result/"
os.makedirs(output_dir, exist_ok=True)

label_files = glob.glob(os.path.join(data_dir, '*.json'))

for label_file in label_files:
    with open(label_file, 'r') as f:
        label_data = json.load(f)

        image_path = label_data['imagePath']
        img = Image.open(os.path.join(data_dir, image_path))
        image_w = img.width; image_h = img.height
        image = np.array(img)
        for shape in label_data['shapes']:
            points = shape['points']
            (x1, y1), (x2, y2) = points
            # rw = image_w/1024; rh = image_h/85; x0 = -4; y0 = 2
            x1 = int(x1); x2 = int(x2); y1 = int(y1); y2 = int(y2)
            # x1 = int(x1*rw)+x0; x2 = int(x2*rw)+x0; y1 = int(y1*rh)+y0; y2 = int(y2*rh)+y0
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (125, 0, 0), 1)
        cv2.imwrite(output_dir + image_path.split("/")[-1], image)
        print(image_path)

