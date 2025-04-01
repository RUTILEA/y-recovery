import pandas as pd
import glob

# Excelファイルのパスを指定
excel_file_path = "/workspace/data/ng_cell_serials.xlsx"  # あなたのExcelファイルのパスに変更してください
check_folder_path = "/workspace/data/substance/single/oblieque2_json/*.tif"

# Excelファイルを読み込む
xls = pd.ExcelFile(excel_file_path)

# "NGデータ"シートのデータを読み込む
data = pd.read_excel(xls, sheet_name='NGデータビード部')
filtered_data = data[data['分類'].isin(['異物不良', 'ビード部不良'])]
# Extract serial numbers and NG location numbers
serials = filtered_data['シリアル'].dropna().astype(str).tolist()

# "NGデータビード部"シートのデータを読み込む
# df_ng_data_bead = pd.read_excel(xls, sheet_name='NGデータビード部')

# "シリアル"列からシリアル番号を抽出し、リストに変換
# serial_numbers_ng_data_bead = df_ng_data_bead[df_ng_data_bead['分類'] == '異物不良']['シリアル'].dropna().tolist()

# 両方のリストを結合
# combined_serial_numbers = serial_numbers_ng_data + serial_numbers_ng_data_bead

files_list = glob.glob(check_folder_path)
print(len(files_list))

cnt = 0
for num in serials:
    for file in files_list:
        # print(num, file.split("/")[-1][:-9])
        if num == file.split("/")[-1][:-9]:
            break
    else:
        
        cnt += 1
        print(cnt, num)