import os
import json
import numpy as np
import glob

json_paths = glob.glob(f'data/results/OK_data*/results.json')
# print(json_paths, len(json_paths))

json_results = {}
for json_path in json_paths:
    json_results |= json.load(open(json_path))
    
name_results = {os.path.basename(name).split('_')[0]: False for name in json_results.keys()}
# print(name_results, len(name_results))

for name, result in json_results.items():
    name_results[os.path.basename(name).split('_')[0]] = name_results[os.path.basename(name).split('_')[0]] or result

OK_NG_results = [result for name, result in name_results.items()]
print(f"{OK_NG_results.count(False)}/{len(OK_NG_results)}")


