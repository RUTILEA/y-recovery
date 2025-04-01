import json

def fix_coco_annotations(json_path):
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    for idx, ann in enumerate(coco_data['annotations']):
        if 'bbox' not in ann:
            x1, y1, _, y2, x2, _, _, _ = ann["segmentation"][0]
            ann['bbox'] = [x1, y1, x2 - x1, y2 - y1]  # 1から始まる一意のIDを割り当てる

    with open(json_path, 'w') as f:
        json.dump(coco_data, f, indent=4)

# 例: "annotations.json" を修正
fix_coco_annotations("/workspace/data/GYHD10200/dataset/train/annotations.json")