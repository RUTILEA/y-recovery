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
    
    for filename in image_list:
        img = cv2.imread(filename)
        h, w = img.shape[:2]
        center_x, _ = w//2, h//2
        crop_left_img = img[:, center_x - 1500:center_x - 400]
        crop_center_img = img[:, center_x - 550:center_x + 550]
        crop_right_img = img[:, center_x + 400:center_x + 1500]
        print(output_dir + filename.split('/')[-1])
        cv2.imwrite(output_dir + "left_" + filename.split('/')[-1], crop_left_img)
        cv2.imwrite(output_dir + "center" + filename.split('/')[-1], crop_center_img)
        cv2.imwrite(output_dir + "right" + filename.split('/')[-1], crop_right_img)
    return

if __name__ == "__main__":
    input_dir = "/workspace/data/OK_image/oblieque2_small"
    output_dir = "/workspace/data/crop_image/oblieque2/"
    os.makedirs(output_dir, exist_ok=True)
    main(input_dir)