import os
import shutil
import pandas as pd
import glob

excel_file = "/workspace/data/ng_cell_serials.xlsx"
# data = pd.read_excel(excel_file)
data = pd.read_excel(excel_file, sheet_name="NGデータビード部")