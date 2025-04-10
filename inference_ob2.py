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

class InspectorOblique2:
    def __init__(self, input_dir, weight_dir, weight_list, output_dir, gpu_id):
        self.input_dir = input_dir
        self.weight_dir = weight_dir
        self.weight_list = weight_list
        self.output_dir = output_dir
        self.crop_image_size = {'width': 1100, 'height': 256}
        self.buffer = 150
        self._set_predictor(weight_list, gpu_id)
        
    def _setup_cfg(self, gpu_id, weight_path, threshold):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.MODEL.WEIGHTS = weight_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.2
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

    def black_boxes(self, image, boxes):
        new_boxes = np.empty((0, 4), int)
        buff = 5
        min_brightness = 80
        min_brightness_difference = -10
        height, width = image.shape[:2]
        for box in boxes:
            x1, y1, x2, y2 = box; x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if x2 <= x1 or y2 <= y1:
                continue
            
            # 周囲のエリアよりも明るいかどうかを平均値により判断する
            surrounding_area = image[max(0, y1-buff):y1, x1:x2].reshape(-1, 3)
            surrounding_area = np.concatenate([surrounding_area, image[y2:min(height, y2+buff), x1:x2].reshape(-1, 3)], axis=0)
            surrounding_area = np.concatenate([surrounding_area, image[y1:y2, max(0, x1-buff):x1].reshape(-1, 3)], axis=0)
            surrounding_area = np.concatenate([surrounding_area, image[y1:y2, x2:min(width, x2+buff)].reshape(-1, 3)], axis=0)
            anomaly = image[y1:y2, x1:x2]
            if surrounding_area.shape[0] == 0:
                flag1 = True
            else:
                flag1 = int(np.mean(anomaly)) - int(np.mean(surrounding_area)) > min_brightness_difference
            
            # 周囲のエリアよりも明るいかどうかを中央値により判断する（ノイズの影響を受けにくい）
            surrounding_area_averages = []
            if image[max(0, y1-buff):y1, x1:x2].reshape(-1, 3).shape[0] != 0:
                surrounding_area_averages.append(int(np.mean(image[max(0, y1-buff):y1, x1:x2].reshape(-1, 3))))
            if image[y2:min(height, y2+buff), x1:x2].reshape(-1, 3).shape[0] != 0:
                surrounding_area_averages.append(int(np.mean(image[y2:min(height, y2+buff), x1:x2].reshape(-1, 3))))
            if image[y1:y2, max(0, x1-buff):x1].reshape(-1, 3).shape[0] != 0:
                surrounding_area_averages.append(int(np.mean(image[y1:y2, max(0, x1-buff):x1].reshape(-1, 3))))
            if image[y1:y2, x2:min(width, x2+buff)].reshape(-1, 3).shape[0] != 0:
                surrounding_area_averages.append(int(np.mean(image[y1:y2, x2:min(width, x2+buff)].reshape(-1, 3))))
            if len(surrounding_area_averages) == 0:
                flag2 = True
            else:
                flag2 = int(np.mean(anomaly)) - np.median(surrounding_area_averages) > min_brightness_difference
            
            # 最低輝度を上回っているかどうかを判断する
            flag3 = int(np.max(anomaly)) > min_brightness
            flag = flag1 and flag2 and flag3
            
            if flag:
                new_boxes = np.concatenate([new_boxes, box.reshape(1, 4)], axis=0)
        return new_boxes

    def read_image(self, filename):
        image_path = os.path.join(self.input_dir, filename) 
        img = cv2.imread(image_path)
        return img
    
    def extract_annotation(self, label_data, height, width):
        for shape in label_data['shapes']:
            points = shape['points']
            (x1, y1), (x2, y2) = points
            center_x, _ = width // 2, height // 2
            P = [[(0, 0), (0, 0)] for _ in range(3)]
            if x1 < center_x - 400:
                P[0] = [(x1-(center_x-1500), y1), (min(x2-(center_x-1500), 1100), y2)]
            if x1 < center_x + 550 or x2 > center_x - 550:
                P[1] = [(max(x1-(center_x-550), 0), y1), (min(x2-(center_x-550), 1100), y2)]
            if x2 > center_x + 400:
                P[2] = [(max(x1-(center_x+400), 0), y1), (x2-(center_x+400), y2)]
        return P
    
    def crop_images(self, img):
        height, width = img.shape[:2]
        center_x, _ = width//2, height//2
        crop_left_img = img[:, center_x - 3*(self.crop_image_size['width']//2) + self.buffer:center_x - (self.crop_image_size['width']//2) + self.buffer]
        crop_center_img = img[:, center_x - self.crop_image_size['width']//2:center_x + self.crop_image_size['width']//2]
        crop_right_img = img[:, center_x + self.crop_image_size['width']//2 - self.buffer:center_x + 3*(self.crop_image_size['width']//2) - self.buffer]
        crop_images = [crop_left_img, crop_center_img, crop_right_img]
        return crop_images
    
    def save_image(self, img, outputs, filename):
        v = Visualizer(img[:, :, ::-1], self.metadata, scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        output_path = os.path.join(self.output_dir, filename)
        out_image = out.get_image()
        cv2.imwrite(output_path, out_image[:, :, ::-1])
    
    def is_substance(self, boxes, point):
        (x1, y1), (x2, y2) = point
        for box in boxes:
            if self.check_overlap(box, [x1, y1, x2, y2]):
                return True
        return False
    
    def check_overlap(self, box1, box2):
        if box1[2] <= box2[0] or box2[2] <= box1[0]:
            return False
        if box1[3] <= box2[1] or box2[3] <= box1[1]:
            return False
        return True
    
    def convert_coordinate(self, boxes, height, width, position=0): # position: 0: left, 1: center, 2: right 
        new_boxes = []
        center_x, _ = width // 2, height // 2
        if position == 0:
            for box in boxes:
                box[0] += (center_x - 3*(self.crop_image_size['width']//2) + self.buffer)
                box[2] += (center_x - 3*(self.crop_image_size['width']//2) + self.buffer)
                new_boxes.append(box)
        elif position == 1:
            for box in boxes:
                box[0] += (center_x - self.crop_image_size['width']//2)
                box[2] += (center_x - self.crop_image_size['width']//2)
                new_boxes.append(box)
        else:
            for box in boxes:
                box[0] += (center_x + self.crop_image_size['width']//2 - self.buffer)
                box[2] += (center_x + self.crop_image_size['width']//2 - self.buffer)
                new_boxes.append(box)
            
        return new_boxes
        
    
    def check_inspect(self):
        input_files = glob.glob(os.path.join(self.input_dir, '*.json'))
        
        cnt = 0
        for label_file in input_files:
            with open(label_file, 'r') as f:
                label_data = json.load(f)
                filename = label_data['imagePath']
            img = self.read_image(filename)
            crop_images = self.crop_images(img)
            height, width = img.shape[:2]
            points = self.extract_annotation(label_data, height, width)
            
            detect = False
            for k in range(len(self.predictor)):
                for i, (crop_img, point) in enumerate(zip(crop_images, points)):
                    outputs = self.predictor[k](crop_img)
                    isinstance = outputs["instances"]
                    boxes = isinstance.pred_boxes.tensor.cpu().numpy()
                    self.save_image(crop_img, outputs, filename.split('.')[0]+f'_{3*k+i}'+'.png')
                    ok = self.is_substance(boxes, point)
                    if ok: detect = True
                
            if detect:
                print(f"detected: {filename}")
                cnt += 1
            else:
                print(f"not detected: {filename}")
                pass
        print(f"detect count: {cnt}/{len(input_files)}")

    def save_image2(self, img, detected_areas, filename):
        img_copy = img.copy()
        for box in [box for areas in detected_areas for box in areas]:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 1)  # 緑色の矩形を描画
        output_path = os.path.join(self.output_dir, filename)
        cv2.imwrite(output_path, img_copy)
        
    def inspect(self, img, save=False, saveID=None):
        detected_areas = []
        crop_images = self.crop_images(img) 
        for k in range(len(self.predictor)):
            for i, crop_img in enumerate(crop_images):
                outputs = self.predictor[k](crop_img)
                isinstance = outputs["instances"]
                boxes = isinstance.pred_boxes.tensor.cpu().numpy()
                boxes = self.convert_coordinate(boxes, img.shape[0], img.shape[1], i)
                # 大きいboxを除外
                new_boxes = np.empty((0, 4), int)
                for box in boxes:
                    if (box[2] - box[0]) * (box[3] - box[1]) > 25 * 25:
                        continue
                    new_boxes = np.concatenate([new_boxes, box.reshape(1, 4)], axis=0)
                # 黒いboxを除外
                new_boxes = self.black_boxes(img, new_boxes)
                detected_areas.append(new_boxes)
                if save: self.save_image(crop_img, outputs, saveID+f'_{3*k+i}'+'.png')
        if len(detected_areas) != 0:
            self.save_image2(img, detected_areas, saveID+'.png')
            
        return detected_areas
    
    def inspect_one_cell(self):
        input_path = glob.escape(self.input_dir)
        input_files = glob.glob(os.path.join(input_path, "*.png"))
        for filename in input_files:
            slice_number = int(filename[-8:-4])
            if slice_number < 400 or 624 < slice_number: continue
            fileID = f"{filename.split('_')[-3]}_{filename.split('_')[-2]}"
            img = self.read_image(filename)
            self.inspect(img, save=True, saveID=f"{fileID}_{filename[-8:-4]}")
            print("slice number: ", slice_number)
        


if __name__ == "__main__":
    import time
    start = time.time()
    input_dir = "/workspace/data/substance/single/oblique2_json"   
    weight_dir = "/workspace/weights/oblique2"
    # weights_list = [("model_main.pth", 0.1), ("model_sub.pth", 0.23)]
    weights_list = [("model_main.pth", 0.1), ("model_sub.pth", 0.3), ("model_small.pth", 0.9)]
    output_dir = "/workspace/data/results/oblique2/difficult_crop4"
    os.makedirs(output_dir, exist_ok=True)
    inspector = InspectorOblique2(input_dir, weight_dir, weights_list, output_dir)
    inspector.check_inspect()
    
    # input_dir = "/workspace/data/NG_data/1GP170529A0214_正極_20170708_180638/[oblique2]"   
    # weights_dir = "/workspace/weights/oblique2/"
    # weights_list = [("model_main.pth", 0.1), ("model_sub.pth", 0.2)]
    # output_dir = "/workspace/data/results/oblique2/170529A0214P_test"
    # os.makedirs(output_dir, exist_ok=True)
    # inspector = InspectorOblique2(input_dir, weights_dir, weights_list, output_dir)
    # inspector.inspect_one_cell()
    print(f"elapsed time: {(time.time() - start)/60:.2f}min")
