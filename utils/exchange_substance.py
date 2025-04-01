import os
import glob
import shutil
import json

input_dir = "/workspace/data/GYHD10200/high_resolution/高解像度NGデータ溶接ビード部/*"
# input_dir = "/workspace/data/GYHD10200/high_resolution/NGData/*"
dirs = glob.glob(input_dir)

# NG_dir =  "/workspace/data/development_data/substance/single/oblieque1_json/*.json"
# # NG_dir = "/workspace/data/GYHD10176/低解像度/TIFF low resolution_NG bead section_NG portion extracted and annotated/*"
refnames = []
# refnames = [ref.split('/')[-1][:-5]+'.tif' for ref in refnames]
# print(len(refnames))
labelme_folder = "/workspace/data/development_data/substance/single/oblieque1_json/"
label_files = glob.glob(os.path.join(labelme_folder, '*.json'))

for label_file in label_files:
    with open(label_file, 'r') as f:
        label_data = json.load(f)
    refnames.append(label_data['imagePath'])

output_dir = "/workspace/data/development_data/substance/single/oblieque1_json/"
os.makedirs(output_dir, exist_ok=True)

cnt = 0
for dir in dirs:
    filenames = [f for f in os.listdir(dir + "/[オブリーク1]/") if f.endswith((".png", ".jpg", ".jpeg", ".tif"))]
    # print(len(filenames))
    for i, filename in enumerate(filenames):
        if filename in refnames:
            print(filename)
            shutil.copy(dir+"/[オブリーク1]/"+filename, output_dir+filename)
            cnt += 1
print(cnt)

# extract one NG image by one workpiece
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
#     refnames += [f for f in os.listdir(ref) if f.endswith((".png", ".jpg", ".jpeg", ".tif"))]
# print(len(refnames))

# for dir in dirs:
#     print(dir)
#     filenames = [f for f in os.listdir(dir + "/[オブリーク1]/") if f.endswith((".png", ".jpg", ".jpeg", ".tif"))]
#     for i, filename in enumerate(filenames):
#         if filename in refnames:
#             shutil.copy(dir+"/[オブリーク1]/"+filename, output_dir+filename)


# substract number
# json_path = "/workspace/data/development_data/substance/single/oblieque1_json/*.json"
# json_list = glob.glob(json_path)

# for file in json_list:
#     os.rename(file, file[:-9] + f"{int(file[-9:-5])-1:04}" + '.json')            


# # substract substance
# import json
# labelme_folder = "/workspace/data/development_data/substance/single/oblieque1_json/"
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
#                 # os.remove(label_file)
#                 os.remove(label_file[:-5]+".tif")

# labelme_folder = "/workspace/data/development_data/substance/single/oblieque1_json/"
# label_files = glob.glob(os.path.join(labelme_folder, '*.json'))

# for label_file in label_files:
#     with open(label_file, 'r') as f:
#         label_data = json.load(f)
#         file = label_data['imagePath']
#         label_data['imagePath'] = file[:-8] + f"{int(file[-8:-4])-1:04}" + '.tif'

#     with open(label_file, 'w') as f:
#         json.dump(label_data, f, indent=4, ensure_ascii=False)

# import json
# labelme_folder = "/workspace/data/development_data/substance/single/oblieque1_json/"
# label_files = glob.glob(os.path.join(labelme_folder, '*.json'))

# cnt = 0 
# for label_file in label_files:
#     filename = label_file.split('/')[-1]
#     print(filename)
#     with open(label_file, 'r') as f:
#         label_data = json.load(f)
#     imagename = label_data['imagePath']
#     print(imagename[10:])
#     if filename[:-5] == imagename[10:-4]:
#         cnt += 1
# print(cnt)

# json_path = "/workspace/data/development_data/substance/single/oblieque1_json/*.json"
# json_list = glob.glob(json_path)

# image_path = "/workspace/data/development_data/substance/single/oblieque1_json/*.tif"
# image_list = glob.glob(image_path)

# output_dir = "/workspace/data/development_data/substance/single/oblieque1_json/"

# cnt = 0
# for image in image_list:
#     for json in json_list:
#         print(json.split("/")[-1][:-5], image.split("/")[-1][10:-4])
#         # Z
#         # if json.split("/")[-1][:-5] == image.split("/")[-1][:-4]:
#         # ob1
#         if json.split("/")[-1][:-5] == image.split("/")[-1][10:-4]:
#             print(output_dir+json.split("/")[-1])
#             # shutil.copy(json, output_dir+json.split("/")[-1])
#             break
#     else:
#         os.remove(image)
#         pass

