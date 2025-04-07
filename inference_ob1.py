import cv2
import numpy as np
import os
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import glob
import json
from PIL import Image
import pandas as pd

class InspectorOblique1:
    def __init__(self, input_dir, weight_dir, weight_list, output_dir, gpu_id):
        self.input_dir = input_dir
        self.weight_dir = weight_dir
        self.weight_list = weight_list
        self.output_dir = output_dir
        self.crop_size = {'width': 740, 'height': 256}
        self._set_predictor(weight_list, gpu_id)
        
    def _setup_cfg(self, gpu_id, weight_path, threshold=0.3):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.MODEL.WEIGHTS = weight_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        cfg.DATASETS.TEST = ("val2", "my_dataset_val3",)
        cfg.MODEL.DEVICE = f'cuda:{gpu_id}'
        return cfg
        
    def _set_predictor(self, weight_list, gpu_id):
        self.predictor = []
        for weight, thresh in weight_list:
            weight_path = os.path.join(self.weight_dir, weight)
            cfg = self._setup_cfg(gpu_id, weight_path, thresh)
            p = DefaultPredictor(cfg)
            self.predictor.append(p)
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        
    def read_image(self, filename):
        image_path = os.path.join(self.input_dir, filename) 
        img = cv2.imread(image_path)
        return img
    
    def extract_annotation(self, label_data, height, width):
        for shape in label_data['shapes']:
            points = shape['points']
            (x1, y1), (x2, y2) = points
            x1, y1, x2, y2 = int((x1*width)/1024), int((y1*height)/85), int((x2*width)/1024), int((y2*height)/85)
            center_x, _ = width // 2, height // 2
            P = [(x1-(center_x-370), y1), (x2-(center_x-370), y2)]
        return P
    
    def extract_annotation2(self, label_data, height, width):
        for shape in label_data['shapes']:
            points = shape['points']
            (x1, y1), (x2, y2) = points
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            center_x, _ = width // 2, height // 2
            P = [(x1-(center_x-370), y1), (x2-(center_x-370), y2)]
        return P
    
    def crop_images(self, img):
        _, w = img.shape[:2]
        center_x = w//2
        crop_img = img[:, center_x - self.crop_size['width'] // 2:center_x + self.crop_size['width'] // 2]
        return crop_img
    
    def save_image(self, img, outputs, filename):
        v = Visualizer(img[:, :, ::-1], self.metadata, scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        output_path = os.path.join(self.output_dir, filename)
        out_image = out.get_image()
        cv2.imwrite(output_path, out_image[:, :, ::-1])
    
    def is_substance(self, boxes, point):
        for box in boxes:
            result_x1, result_y1, result_x2, result_y2 = box
            (x1, y1), (x2, y2) = point
            if result_x1 < x2 and result_x2 > x1 and result_y1 < y2 and result_y2 > y1:
                return True
        return False
    
    def write_csv(self, results):
        # 出力ファイルのパス
        output_csv = os.path.join(self.output_dir, "results.csv")
        new_df = pd.DataFrame(results, columns=['filename', '0001999'])
        # 既存のCSVファイルがあるかチェック
        if os.path.exists(output_csv):
            # 既存データを読み込む
            existing_df = pd.read_csv(output_csv)
            # **新しい列があれば既存データに追加**
            for col in new_df.columns:
                if col not in existing_df.columns:
                    existing_df[col] = None  # 追加列にNaNを入れる
            # **データを結合**
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            # 新規作成
            combined_df = new_df
        # CSVに保存（上書き）
        combined_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        
    
    def check_inspect(self):
        input_files = glob.glob(os.path.join(self.input_dir, '*.json'))
        
        cnt = 0
        results = []
        for label_file in input_files:
            with open(label_file, 'r') as f:
                label_data = json.load(f)
                filename = label_data['imagePath']
            img = self.read_image(filename)
            height, width = img.shape[:2]
            crop_image = self.crop_images(img)
            point = self.extract_annotation2(label_data, height, width)
            
            detect = False
            outputs = self.predictor[0](crop_image)
            isinstance = outputs["instances"]
            boxes = isinstance.pred_boxes.tensor.cpu().numpy()
            self.save_image(crop_image, outputs, filename)
            ok = self.is_substance(boxes, point)
            if ok: detect = True
            results.append([filename, 'o' if detect else 'x'])
            
            if detect:
                print(f"detected: {filename}")
                cnt += 1
            else:
                print(f"not detected: {filename}")
                pass
        # df = pd.DataFrame(results, columns=['filename', '0000999'])
        # df.to_csv(os.path.join(self.output_dir, "results.csv"), index=False)
        print(f"detect count: {cnt}/{len(input_files)}")
        
    def convert_coordinate(self, boxes, height, width):
        new_boxes = []
        center_x, _ = width // 2, height // 2
        
        ### ここから修正
        for box in boxes:
            box[0] += center_x - 3*(self.crop_image_size['width']//2) + self.buffer
            box[2] += center_x - 3*(self.crop_image_size['width']//2) + self.buffer
            new_boxes.append(box)
        return new_boxes
        
    def inspect(self, img, save=False, saveID=None):
        crop_img = self.crop_images(img) 
        outputs = self.predictor[0](crop_img)
        isinstance = outputs["instances"]
        boxes = isinstance.pred_boxes.tensor.cpu().numpy()
        boxes = self.convert_coordinate(boxes, img.shape[0], img.shape[1])
        if len(boxes) != 0 and save:
            self.save_image(img, outputs, saveID+'.png')
        return boxes
        



if __name__ == "__main__":
    input_dir = "/workspace/data/substance/single/oblique1_difficult_json2"   
    weight_dir = "/workspace/weights/oblique1/"
    weight_list = [("model_main.pth", 0.8)]
    # weight_list = [("model_main.pth", 0.8), ("model_sub.pth", 0.8)]
    output_dir = "/workspace/data/results/oblique1/difficult_model_0001999"
    os.makedirs(output_dir, exist_ok=True)
    
    # for weight in weights_list:
    #     weight_path = os.path.join(weight_dir, weight)
    #     inspector = Inspector_oblique1(input_dir, weight_path, output_dir)
    #     print(f"weight: {weight}")
        # inspector.check_inspect()
    inspector = InspectorOblique1(input_dir, weight_dir, weight_list, output_dir)
    inspector.check_inspect()
