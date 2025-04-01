import os
import glob
import shutil

# input_dir = "/workspace/data/GYHD10200/high_resolution/高解像度NGデータ溶接ビード部/*"
# dirs = glob.glob(input_dir)

# ref = "/workspace/data/GSYuasa_annotation_v3/extract/oblieque2/"

# output_dir = "/workspace/data/GSYuasa_annotation_v3/extract/oblieque2/"
# os.makedirs(output_dir, exist_ok=True)


# refnames = []

# refnames += [f for f in os.listdir(ref) if f.endswith((".png", ".jpg", ".jpeg", ".json", ".tif"))]

# for dir in dirs:
#     # print(dir)
#     filenames = [f for f in os.listdir(dir+"/[オブリーク2]/") if f.endswith((".png", ".jpg", ".jpeg", ".tif"))]
#     for i, filename in enumerate(filenames):
#         # print(filename)
#         for refname in refnames:
#             if filename[:-4] == refname[:-5]:
#                 print(filename)
#                 shutil.copy(dir+"/[オブリーク2]/"+filename, output_dir+filename)
            
json_path = "/workspace/data/substance/single/Z_bead_json/*json"
json_list = glob.glob(json_path)

image_path = "/workspace/data/substance/single/Z_bead_json/*.tif"
image_list = glob.glob(image_path)

cnt = 0
for json in json_list:
    for image in image_list:
        if json.split("/")[-1][:-5] == image.split("/")[-1][:-4]:
            break
    else:
        print(image)
        cnt += 1
        print(cnt)
        os.remove(json)

# import json

# labelme_folder = '/workspace/data/GSYuasa_annotation_v3/extract/oblieque2'
# label_files = glob.glob(os.path.join(labelme_folder, '*.json'))


# for label_file in label_files:
#     with open(label_file, 'r') as f:
#         label_data = json.load(f)
#     for shape in label_data['shapes']:
#             label = shape['label']
#             if label != 'substance':
#                 print(label, label_file)
#                 os.remove(label_file)

# file_path = "/workspace/data/GSYuasa_annotation_v3/extract/oblieque1/"
# file_list = glob.glob(file_path+"*")

# for file in file_list:
#     print(file, file_path + file.split("/")[-1][10:])
#     os.rename(file, file_path + file.split("/")[-1][10:])
    