import os
import glob
import shutil

input_dir = "/workspace/data/GYHD10200/high_resolution/高解像度良品データ/*"
dirs = glob.glob(input_dir)

output_dir = "/workspace/data/GYHD10200/OK_images/oblieque2/"
os.makedirs(output_dir, exist_ok=True)


for dir in dirs:
    # print(dir)
    filenames = [f for f in os.listdir(dir+"/[オブリーク2]/") if f.endswith((".png", ".jpg", ".jpeg", ".tif"))]
    for i, filename in enumerate(filenames):
        if int(filename[-7:-4]) < 400 or 624 < int(filename[-7:-4]):
            continue
        # print(dir+"/[Z軸]/"+filename)
        shutil.copy(dir+"/[オブリーク2]/"+filename, output_dir+filename)
        # if filename[:-4] in refnames:
        #     print(filename)
        #     shutil.copy(dir+"/[Z軸]/"+filename, output_dir+filename)
        # if filename in refnames:
        #     print(filename)
        #     shutil.copy(dir+"/[Z軸]/"+filename, output_dir+filename)
            
        
