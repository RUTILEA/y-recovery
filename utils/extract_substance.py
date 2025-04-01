import os
import glob
import shutil


# # extract one NG image by one workpiece
# input_dir = "/workspace/data/GYHD10200/high_resolution/高解像度NGデータ溶接ビード部/*"
# input_dir = "/workspace/data/GYHD10200/high_resolution/NGData/*"
# dirs = glob.glob(input_dir)

# NG_dir = "/workspace/data/GYHD10176/低解像度/TIFF low resolution_NG_NG partial extraction/*"
# # NG_dir = "/workspace/data/GYHD10176/低解像度/TIFF low resolution_NG bead section_NG portion extracted and annotated/*"
# refs = glob.glob(NG_dir)

# output_dir = "/workspace/data/development_data/substance/single/oblieque1_json/"
# os.makedirs(output_dir, exist_ok=True)

# refnames = []
# for ref in refs:
#     refnames += [f for f in os.listdir(ref) if f.endswith((".png", ".jpg", ".jpeg", ".tif", ".json"))]

# for dir in dirs:
#     print(dir)
#     filenames = [f for f in os.listdir(dir + "/[オブリーク1]/") if f.endswith((".png", ".jpg", ".jpeg", ".tif"))]
#     for i, filename in enumerate(filenames):
#         if filename in refnames:
#             shutil.copy(dir+"/[オブリーク1]/"+filename, output_dir+filename)

# extract annotation data matching images
# json_path = "/workspace/data/substance/single/oblieque2_json/*.json"
# json_list = glob.glob(json_path)

# image_path = "/workspace/data/substance/single/oblieque2_difficult_json/*.tif"
# image_list = glob.glob(image_path)

# output_dir = "/workspace/data/substance/single/oblieque2_difficult_json/"

# cnt = 0
# for image in image_list:
#     for json in json_list:
#         if json.split("/")[-1][:-5] == image.split("/")[-1][:-4]:
#             print(output_dir+json.split("/")[-1])
#             shutil.copy(json, output_dir+json.split("/")[-1])
#             break
#     else:
#         # os.remove(image)
#         pass

# # extract annotation data matching images
# json_path = "/workspace/data/substance/single/oblieque2_difficult_json/*.json"
# json_list = glob.glob(json_path)
# for json in json_list:
#     print(json)
#     os.remove(json)

# # remain substance
# import json

# labelme_folder = "/workspace/data/development_data/substance/single/Z_bead_json/"
# label_files = glob.glob(os.path.join(labelme_folder, '*.json'))

# for label_file in label_files:
#     with open(label_file, 'r') as f:
#         label_data = json.load(f)
#     for shape in label_data['shapes']:
#             label = shape['label']
#             print(label)
#             if label != 'substance':
#                 print(label, label_file)
#                 print(label_file[:-5]+".tif")
#                 os.remove(label_file)
#                 os.remove(label_file[:-5]+".tif")
                    


