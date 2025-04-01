import os
import shutil
import pandas as pd
import glob

# Excelファイルの読み込み
excel_file = "/workspace/data/ng_cell_serials.xlsx"
# data = pd.read_excel(excel_file)
data = pd.read_excel(excel_file, sheet_name="NGデータビード部")
# print(data.columns)


# N列が"異物不良"でB列のシリアル番号とNGファイル番号と極を取得
# serials = data.loc[data['分類'] == '異物不良', 'シリアル'].dropna().astype(str).tolist()
# serials += data.loc[data['分類'] == 'ビード部不良', 'シリアル'].dropna().astype(str).tolist()
# ng_numbers = data.loc[data['分類'] == '異物不良', 'NG場所オブリーク2'].dropna().astype(str).tolist()
# ng_numbers += data.loc[data['分類'] == 'ビード部不良', 'NG場所オブリーク2'].dropna().astype(str).tolist()

filtered_data = data[data['分類'].isin(['異物不良', 'ビード部不良'])]
# Extract serial numbers and NG location numbers
serials = filtered_data['シリアル'].dropna().astype(str).tolist()
ng_numbers = filtered_data['NG場所Z軸'].dropna().astype(str).tolist()

# # コピー元とコピー先のディレクトリ
# source_directory = "/workspace/data/NG_data"  # フォルダ全体が格納されているディレクトリ
source_directory = "/workspace/data/substance/single/Z_json"  # フォルダ全体が格納されているディレクトリ
destination_directory = "/workspace/data/substance/single/Z_bead_json"  # コピー先ディレクトリ


# # シリアル番号とNGファイル番号のリストを作成
serial_ng_list = list(zip(serials, ng_numbers))
for serial, ng_number in serial_ng_list:
    # print(serial, ng_number)
    # ng_numberを4桁にする
    # ng_number = str(int(ng_number)-1)
    ng_number = ng_number.zfill(4)
    # # file_dir = f"/{serial}*{pole}*/*Z*/*{ng_number}.tif"
    # # # *を使ったfile_dirと一致するファイルを検索
    # # file_path = glob.glob(source_directory+file_dir)
    # # if len(file_path) == 0:
    # #     continue
    try:
        file_path = source_directory + f"/{serial[3:]}_{ng_number}.tif"
        shutil.copy(file_path, destination_directory)
        file_path = source_directory + f"/{serial[3:]}_{ng_number}.json"
        shutil.copy(file_path, destination_directory)    
    except FileNotFoundError:
        print(f"File not found: {serial}_{ng_number}.tif")
        continue
    
    

# print("Copy completed.")
