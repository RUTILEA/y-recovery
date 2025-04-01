import cv2

image_path = "/workspace/data/NG_data/1GP170529A0214_正極_20170708_180638/[オブリーク2]/"
image_name = "オブリーク2_正極_1GP170529A0214_0503.tif"
output_dir = "/workspace/"
img = cv2.imread(image_path+image_name)
cv2.drawMarker(img, (1571, 105), (255, 0, 0), markerSize=20)
cv2.imwrite(output_dir + "check.png", img)

image_path = "/workspace/data/NG_data/1GP170529A0214_正極_20170708_180638/[Z軸]/"
image_name = "Z軸_正極_1GP170529A0214_0146.tif"
output_dir = "/workspace/"
img = cv2.imread(image_path+image_name)
cv2.drawMarker(img, (517, 500), (255, 0, 0), markerSize=20)
cv2.imwrite(output_dir + "check_Z.png", img)