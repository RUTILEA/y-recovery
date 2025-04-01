import glob
import os

file_path = "/workspace/data/substance/single/oblieque2_json/"
file_list = glob.glob(file_path+"*.tif")

for file in file_list:
    if len(file.split("/")[-1]) < 25:
        continue
    print(file, file_path + file.split("/")[-1][10:])
    os.rename(file, file_path + file.split("/")[-1][10:])