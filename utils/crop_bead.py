import glob
import cv2
import numpy as np

def detect_contours(image : np.ndarray) -> list[int]:
    # Detect the contours of the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 150, 250)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = float('-inf'), float('-inf')
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x_min, y_min = min(x_min, x), min(y_min, y)
        x_max, y_max = max(x_max, x + w), max(y_max, y + h)
    return [x_min, y_min, x_max, y_max]

def detect_center(contour) -> list[int]:
    center_x = (contour[0] + contour[2]) // 2
    center_y = (contour[1] + contour[3]) // 2
    return [center_x, center_y]

def create_bbox(center, height, width):
    x_min = center[0] - width // 2
    x_max = center[0] + width // 2
    y_min = center[1] - height // 2
    y_max = center[1] + height // 2
    return [x_min, y_min, x_max, y_max]

def crop_image(image, bbox):
    x_min, y_min, x_max, y_max = bbox
    crop_image = image[y_min:y_max, x_min:x_max]
    return crop_image

def crop_bead_area(image):
    left_bead = image[30:220, 80:320]
    right_bead = image[30:220, 680:920]
    return [left_bead, right_bead]
    
def main(input_dir):
    print("start")
    image_list = glob.glob(input_dir + "/*.tif")
    
    height = 250; width = 1000
    for filename in image_list:
        if int(filename[-8:-4]) < 230 or int(filename[-8:-4]) > 245:
            continue
        img = cv2.imread(filename)
        countour = detect_contours(img)
        # center = detect_center(countour)
        center = [img.shape[1] // 2, img.shape[0] // 2]
        bbox = create_bbox(center, height, width)
        crop_cell_img = crop_image(img, bbox)
        # cv2.imwrite(output_dir + filename.split('/')[-1], crop_cell_img)
        bead_img = crop_bead_area(crop_cell_img)
        cv2.imwrite(output_dir + "left_" + filename.split('/')[-1], bead_img[0])
        cv2.imwrite(output_dir + "right_" + filename.split('/')[-1], bead_img[1])
    return

if __name__ == "__main__":
    import os
    output_dir = "/workspace/data/crop_image/Z_new/"
    for cell_dir in os.listdir("/workspace/data/NG_data"):
        print(cell_dir)
        input_dir = f"/workspace/data/NG_data/{cell_dir}/[Z軸]/"
        input_dir = glob.escape(input_dir) 
        # output_dir = f"/workspace/data/crop_image/{cell_dir}/"
        os.makedirs(output_dir, exist_ok=True)
        main(input_dir)
        input_dir = "/workspace/data/NG_data/1GP170529A0214_正極_20170708_180638/[Z軸]/"
    # input_dir = glob.escape(input_dir) 
    # output_dir = "/workspace/data/crop_image/Z_new/"
    # os.makedirs(output_dir, exist_ok=True)
    # main(input_dir)