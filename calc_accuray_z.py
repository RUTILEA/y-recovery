import os
import json
import numpy as np
import glob

correct_data = json.load(open("complete_correct_data.json"))
text_paths = glob.glob(f'data/results/NG_data_*/Zaxis/*.txt')
# print(text_paths, len(text_paths))
text_data_list = [json.load(open(text_path)) for text_path in text_paths]
# print(text_data_list[0], len(text_data_list))
detection_results_NG_data = {os.path.basename(os.path.splitext(text_path)[0]): text_data for text_path, text_data in zip(text_paths, text_data_list)}
# print(detection_results_NG_data, len(text_data_list))
# exit()

correctly_detected_count = 0
for data_name, data_value in correct_data.items():
    data_value["xy"]
    if f"{data_value['pole']}_{data_name}_{data_value['z']}" not in detection_results_NG_data:
        print(f"{data_name} {data_value['z']}")
        continue
    is_correctly_detected = False
    for box1 in detection_results_NG_data[f"{data_value['pole']}_{data_name}_{data_value['z']}"]:
        for box2 in data_value["xy"]:
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            area_of_intersection = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0])) * max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
            IOU = area_of_intersection / (area1 + area2 - area_of_intersection)
            if not (box1[2] < box2[0] or box2[2] < box1[0] or box1[3] < box2[1] or box2[3] < box1[1]) and IOU > 0.01:
                correctly_detected_count += 1
                is_correctly_detected = True
                break
        if is_correctly_detected:
            break
    if not is_correctly_detected:
        print(f"{data_name} {data_value['z']}")

print(f"{correctly_detected_count} / {len(correct_data)}")
print(len([box for boxes in detection_results_NG_data.values() for box in boxes]))





