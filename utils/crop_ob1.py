import glob
import cv2
import numpy as np
import os


def crop_image(image, bbox):
    x_min, y_min, x_max, y_max = bbox
    crop_image = image[y_min:y_max, x_min:x_max]
    return crop_image
    
def main(input_dir):
    print("start")
    image_list = glob.glob(input_dir + "/*.tif")
    
    height = 250; width = 740
    for filename in image_list:
        img = cv2.imread(filename)
        h, w = img.shape[:2]
        center_x, center_y = w//2, h//2
        crop_cell_img = img[:, center_x - width // 2:center_x + width // 2]
        print(output_dir + filename.split('/')[-1])
        cv2.imwrite(output_dir + filename.split('/')[-1], crop_cell_img)
    return

if __name__ == "__main__":
    input_dir = "/workspace/data/OK_image/oblieque1_545"
    output_dir = "/workspace/data/crop_image/oblieque1/"
    os.makedirs(output_dir, exist_ok=True)
    main(input_dir)