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

class InspectorZaxis:
    def __init__(self, input_dir, weight_dir, weight_list, output_dir, gpu_id):
        self.input_dir = input_dir
        self.weight_dir = weight_dir
        self.weight_list = weight_list
        self.output_dir = output_dir
        self._set_predictor(weight_list, gpu_id)
        self.cell_size = {'width': 1000, 'height': 250}
        self.bead_area = {'left': [80, 30, 320, 220], 'right': [680, 30, 920, 220]}  # x1, y1, x2, y2
        self._set_exclusion_area()
        
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
    
    def _set_exclusion_area(self):
        self.exclusion_area_positive = [
            (0, 0, 0, 1024, 390, 256),  # 上端
            (0, 635, 0, 1024, 1024, 256),  # 下端
            (0, 300, 0, 80, 700, 256),  # 左端
            (945, 300, 0, 1024, 700, 256),  # 右端
            (490, 490, 0, 535, 535, 40),  # 中央のムラ
            (20, 600, 0, 40, 620, 45),  # 四隅の出っ張り
            (20, 410, 0, 40, 430, 45),  # 四隅の出っ張り
            (990, 410, 0, 1010, 430, 45),  # 四隅の出っ張り
            (990, 600, 0, 1010, 620, 45),  # 四隅の出っ張り
            (60, 440, 50, 965, 475, 83),  # 上の黒い線
            (60, 550, 30, 965, 585, 60),  # 下の黒い線
            (60, 535, 61, 965, 600, 83),  # 下の黒い線
            (405, 400, 82, 410, 580, 95),  # 中央の左側の一時的な白い領域**
            (455, 400, 82, 460, 580, 95),  # 中央の左側の一時的な白い領域**
            (560, 435, 82, 565, 615, 95),  # 中央の右側の一時的な白い領域**
            (620, 435, 82, 625, 615, 95),  # 中央の右側の一時的な白い領域**
            (495, 405, 78, 525, 620, 166),  # 中央の白い点*
            (355, 405, 90, 375, 620, 180),  # 中央左側の白い点
            (645, 405, 90, 665, 620, 180),  # 中央右側の白い点
            (95, 405, 78, 105, 620, 166),  # 左部左側の白い点*
            (295, 405, 78, 305, 620, 166),  # 左部右側の白い点*
            (720, 405, 78, 730, 620, 166),  # 右部左側の白い点*
            (920, 405, 78, 930, 620, 166),  # 右部右側の白い点*
            (460, 445, 98, 475, 570, 180),  # 中央すぐ左の白い点*
            (540, 445, 98, 555, 570, 180),  # 中央すぐ右の白い点*
            (390, 470, 90, 405, 550, 165),  # 中央左側中央の白い点**
            (615, 470, 90, 630, 550, 165),  # 中央右側中央の白い点**
            (375, 405, 98, 400, 415, 180),  # 中央左側上部の白い点
            (375, 600, 98, 400, 620, 180),  # 中央左側下部の白い点
            (620, 405, 94, 645, 415, 180),  # 中央右側上部の白い点
            (620, 610, 98, 645, 620, 180),  # 中央右側下部の白い点
            (305, 405, 90, 315, 620, 166),  # 左部右端の白い点*
            (710, 405, 90, 720, 620, 166),  # 右部左端の白い点*
            (390, 415, 102, 420, 420, 125),  # ギザギザ (1,1))/(4,4)
            (390, 470, 102, 420, 475, 125),  # ギザギザ (2,1))/(4,4)
            (390, 520, 102, 420, 525, 125),  # ギザギザ (3,1))/(4,4)
            (390, 575, 102, 420, 585, 125),  # ギザギザ (4,1))/(4,4)
            (445, 415, 102, 475, 420, 125),  # ギザギザ (1,2))/(4,4)
            (445, 470, 102, 475, 475, 125),  # ギザギザ (2,2))/(4,4)
            (445, 520, 102, 475, 525, 125),  # ギザギザ (3,2))/(4,4)
            (445, 575, 102, 475, 585, 125),  # ギザギザ (4,2))/(4,4)
            (550, 440, 102, 580, 445, 125),  # ギザギザ (1,3))/(4,4)
            (550, 495, 102, 580, 500, 125),  # ギザギザ (2,3))/(4,4)
            (550, 545, 102, 580, 550, 125),  # ギザギザ (3,3))/(4,4)
            (550, 595, 102, 580, 605, 125),  # ギザギザ (4,3))/(4,4)
            (605, 440, 102, 635, 445, 125),  # ギザギザ (1,4))/(4,4)
            (605, 495, 102, 635, 500, 125),  # ギザギザ (2,4))/(4,4)
            (605, 545, 102, 635, 550, 125),  # ギザギザ (3,4))/(4,4)
            (605, 595, 102, 635, 605, 125),  # ギザギザ (4,4))/(4,4)
            (425, 470, 100, 440, 585, 133),  # 中央の右側の一時的な白い領域**
            (585, 495, 100, 600, 605, 133),  # 中央の右側の一時的な白い領域**
            (405, 400, 140, 410, 580, 165),  # 中央の左側の一時的な白い領域**
            (455, 400, 140, 460, 580, 165),  # 中央の左側の一時的な白い領域**
            (560, 435, 140, 565, 615, 165),  # 中央の右側の一時的な白い領域**
            (620, 435, 140, 625, 615, 165),  # 中央の右側の一時的な白い領域**
            (120, 405, 142, 145, 620, 165),  # 右部ビード部の境界**
            (880, 405, 142, 905, 620, 165),  # 左部ビード部の境界**
            (110, 400, 157, 340, 440, 209),  # 左部ビード部上
            (110, 580, 157, 340, 620, 209),  # 左部ビード部下
            (110, 400, 157, 120, 620, 209),  # 左部ビード部左
            (300, 400, 157, 340, 620, 209),  # 左部ビード部右
            (685, 400, 157, 915, 440, 209),  # 右部ビード部上
            (685, 580, 157, 915, 620, 209),  # 右部ビード部下
            (685, 400, 157, 725, 620, 209),  # 右部ビード部左
            (905, 400, 157, 915, 620, 209),  # 右部ビード部右
            (725, 440, 157, 745, 460, 209),  # 右側ビード部左上
            (130, 450, 157, 140, 575, 190),  # 左部白色の棒*
            (885, 450, 157, 895, 575, 190),  # 右部白色の棒*
            # (80, 395, 230, 990, 405, 240),  # 物体上部の境界**
            # (80, 620, 230, 990, 630, 240),  # 物体下部の境界**
            (630, 500, 210, 655, 525, 265),  # 画面右部の白丸**
            (490, 490, 191, 535, 535, 256),  # 中央の白点
        ]

        self.exclusion_area_negative = [
            (0, 0, 0, 1024, 390, 256),  # 上端
            (0, 635, 0, 1024, 1024, 256),  # 下端
            (0, 300, 0, 40, 700, 256),  # 左端
            (985, 300, 0, 1024, 700, 256),  # 右端
            (490, 490, 0, 535, 535, 60),  # 中央のムラ
            (20, 600, 0, 40, 620, 45),  # 四隅の出っ張り
            (20, 410, 0, 40, 430, 45),  # 四隅の出っ張り
            (990, 410, 0, 1010, 430, 45),  # 四隅の出っ張り
            (990, 600, 0, 1010, 620, 45),  # 四隅の出っ張り
            (495, 405, 82, 525, 620, 165),  # 中央の白い点*
            (355, 405, 90, 375, 620, 180),  # 中央左側の白い点
            (645, 405, 90, 665, 620, 180),  # 中央右側の白い点
            (95, 405, 82, 105, 620, 165),  # 左部左側の白い点*
            (295, 405, 82, 305, 620, 165),  # 左部右側の白い点
            (720, 405, 82, 730, 620, 165),  # 右部左側の白い点
            (920, 405, 82, 930, 620, 165),  # 右部右側の白い点*
            (460, 460, 98, 475, 570, 165),  # 中央すぐ左の白い点*
            (540, 460, 98, 555, 570, 165),  # 中央すぐ右の白い点*
            (375, 405, 98, 400, 415, 165),  # 中央左側上部の白い点
            (375, 600, 98, 400, 620, 165),  # 中央左側下部の白い点
            (620, 405, 94, 645, 415, 165),  # 中央右側上部の白い点
            (620, 610, 98, 645, 620, 165),  # 中央右側下部の白い点
            (305, 405, 98, 320, 620, 165),  # 左部右端の白い点*
            (705, 405, 98, 720, 620, 165),  # 右部左端の白い点*
            (385, 415, 110, 415, 425, 140),  # ギザギザ (1,1))/(4,4)
            (385, 470, 110, 415, 480, 140),  # ギザギザ (2,1))/(4,4)
            (385, 520, 110, 415, 530, 140),  # ギザギザ (3,1))/(4,4)
            (385, 575, 110, 415, 585, 140),  # ギザギザ (4,1))/(4,4)
            (445, 415, 110, 475, 425, 140),  # ギザギザ (1,2))/(4,4)
            (445, 470, 110, 475, 480, 140),  # ギザギザ (2,2))/(4,4)
            (445, 520, 110, 475, 530, 140),  # ギザギザ (3,2))/(4,4)
            (445, 575, 110, 475, 585, 140),  # ギザギザ (4,2))/(4,4)
            (545, 440, 110, 575, 450, 140),  # ギザギザ (1,3))/(4,4)
            (545, 495, 110, 575, 505, 140),  # ギザギザ (2,3))/(4,4)
            (545, 545, 110, 575, 555, 140),  # ギザギザ (3,3))/(4,4)
            (545, 595, 110, 575, 605, 140),  # ギザギザ (4,3))/(4,4)
            (605, 440, 110, 635, 450, 140),  # ギザギザ (1,4))/(4,4)
            (605, 495, 110, 635, 505, 140),  # ギザギザ (2,4))/(4,4)
            (605, 545, 110, 635, 555, 140),  # ギザギザ (3,4))/(4,4)
            (605, 600, 110, 635, 610, 140),  # ギザギザ (4,4))/(4,4)
            (355, 595, 166, 395, 620, 180),  # 中央左側下部の白い点**
            (110, 400, 157, 340, 440, 209),  # 左部ビード部上
            (110, 580, 157, 340, 620, 209),  # 左部ビード部下
            (110, 400, 157, 120, 620, 209),  # 左部ビード部左
            (300, 400, 157, 340, 620, 209),  # 左部ビード部右
            (685, 400, 157, 915, 440, 209),  # 右部ビード部上
            (685, 580, 157, 915, 620, 209),  # 右部ビード部下
            (685, 400, 157, 725, 620, 209),  # 右部ビード部左
            (905, 400, 157, 915, 620, 209),  # 右部ビード部右
            (125, 450, 166, 145, 575, 195),  # 左部白色の棒*
            (880, 450, 166, 900, 575, 195),  # 右部白色の棒*
            # (80, 395, 230, 990, 405, 240),  # 物体上部の境界
            # (80, 620, 230, 990, 630, 240),  # 物体下部の境界
            (490, 490, 191, 535, 535, 256),  # 中央の白点
        ]
        
    def black_boxes(self, image, boxes):
        new_boxes = np.empty((0, 4), int)
        buff = 5
        min_brightness = 80
        min_brightness_difference = -5
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
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            P = [(x1, y1), (x2, y2)]
        return P
    
    def extract_bead_annotation(self, label_data, height, width, right=False):
        for shape in label_data['shapes']:
            points = shape['points']
            (x1, y1), (x2, y2) = points
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            center_x, center_y = width // 2, height // 2
            if right:
                P = [(x1-(center_x-self.cell_size['width']//2+self.bead_area['right'][0]), \
                    (y1-(center_y-self.cell_size['height']//2+self.bead_area['right'][1]))), \
                    (x2-(center_x-self.cell_size['width']//2+self.bead_area['right'][0]),\
                    (y2-(center_y-self.cell_size['height']//2+self.bead_area['right'][1])))]
            else:
                P = [(x1-(center_x-self.cell_size['width']//2+self.bead_area['left'][0]), \
                    (y1-(center_y-self.cell_size['height']//2+self.bead_area['left'][1]))), \
                    (x2-(center_x-self.cell_size['width']//2+self.bead_area['left'][0]),\
                    (y2-(center_y-self.cell_size['height']//2+self.bead_area['left'][1])))]
                
        return P
    
    def crop_images(self, img):
        height, width = img.shape[:2]
        center_x, center_y = width//2, height//2
        left_bead = img[center_y-self.cell_size['height']//2+self.bead_area['left'][1]: \
                        center_y-self.cell_size['height']//2+self.bead_area['left'][3], \
                        center_x-self.cell_size['width']//2+self.bead_area['left'][0]: \
                        center_x-self.cell_size['width']//2+self.bead_area['left'][2]]
        right_bead = img[center_y-self.cell_size['height']//2+self.bead_area['right'][1]: \
                        center_y-self.cell_size['height']//2+self.bead_area['right'][3], \
                        center_x-self.cell_size['width']//2+self.bead_area['right'][0]: \
                        center_x-self.cell_size['width']//2+self.bead_area['right'][2]]
        return [left_bead, right_bead]

    
    def save_image(self, img, outputs, filename):
        v = Visualizer(img[:, :, ::-1], self.metadata, scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        output_path = os.path.join(self.output_dir, filename)
        out_image = out.get_image()
        cv2.imwrite(output_path, out_image[:, :, ::-1])

    def save_image2(self, img, detected_areas, filename):
        img_copy = img.copy()
        for box in detected_areas:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 1)  # 緑色の矩形を描画
        output_path = os.path.join(self.output_dir, filename)
        cv2.imwrite(output_path, img_copy)
    
    def is_substance(self, boxes, point):
        (x1, y1), (x2, y2) = point
        for box in boxes:
            if self.check_overlap(box, [x1, y1, x2, y2]):
                return True
        return False
    
    def remove_boxes(self, image, boxes, z_index, is_positive_part=True):
        new_boxes = np.empty((0, 4), int)
        for box in boxes:
            should_exclude = False
            for x1, y1, z1, x2, y2, z2 in self.exclusion_area_positive if is_positive_part else self.exclusion_area_negative:
                if z1 <= z_index <= z2 and self.check_overlap(box, [x1, y1, x2, y2]):
                    # 除外領域に重なっている場合は除外
                    should_exclude = True
                    break
            if (box[2] - box[0]) * (box[3] - box[1]) > 30 * 30:
                # 面積が大きすぎる場合は除外
                should_exclude = True
                break
            elif (is_positive_part and z_index >= 240) or (not is_positive_part and z_index >= 245):
                # z_indexが大きい場合にビード部の縁が消えつつある時に小さな白丸が現れ、過検出されるため削除
                is_overdetection = False
                areas = [
                    [105, 455, 115, 570],  # 左ビード左
                    [105, 485, 115, 540],  # 左ビード左
                    [305, 455, 315, 570],  # 左ビード右
                    [305, 485, 315, 540],  # 左ビード右
                    [710, 455, 720, 570],  # 右ビード左
                    [710, 485, 720, 540],  # 右ビード左
                    [912, 455, 922, 570],  # 右ビード右
                    [912, 485, 922, 540],  # 右ビード右
                ]
                for area in areas:
                    if self.check_overlap(box, area) and np.mean(image[area[1]:area[3], area[0]:area[2]].mean()) < 160:  # 170 <= x < 180
                        should_exclude = True
                        break
            
            if not should_exclude:
                new_boxes = np.concatenate([new_boxes, box.reshape(1, 4)], axis=0)
                
        new_boxes = self.black_boxes(image, new_boxes)
        return new_boxes
    
    def check_overlap(self, box1, box2):
        if box1[2] <= box2[0] or box2[2] <= box1[0]:
            return False
        if box1[3] <= box2[1] or box2[3] <= box1[1]:
            return False
        return True

    def merge_boxes(self, boxes):
        new_boxes = np.empty((0, 4), int)
        for b1 in boxes:
            for b2 in new_boxes:
                if self.check_overlap(b1, b2):
                    break
            else:
                new_boxes = np.vstack([new_boxes, b1])
        return new_boxes

    def convert_coordinate(self, boxes, height, width, right):
        new_boxes = np.empty((0, 4), int)
        center_x, center_y = width // 2, height // 2
        if right:
            for box in boxes:
                box[0] += center_x - self.cell_size['width']//2 + self.bead_area['right'][0]
                box[1] += center_y - self.cell_size['height']//2 + self.bead_area['right'][1]
                box[2] += center_x - self.cell_size['width']//2 + self.bead_area['right'][0]
                box[3] += center_y - self.cell_size['height']//2 + self.bead_area['right'][1]
                new_boxes = np.concatenate([new_boxes, box.reshape(1, 4)], axis=0)  
        else:
            for box in boxes:
                box[0] += center_x - self.cell_size['width']//2 + self.bead_area['left'][0]
                box[1] += center_y - self.cell_size['height']//2 + self.bead_area['left'][1]
                box[2] += center_x - self.cell_size['width']//2 + self.bead_area['left'][0]
                box[3] += center_y - self.cell_size['height']//2 + self.bead_area['left'][1]
                new_boxes = np.concatenate([new_boxes, box.reshape(1, 4)], axis=0)
        return new_boxes
                
    
    def check_inspect_one_model(self, model_idx=0):
        input_files = glob.glob(os.path.join(self.input_dir, '*.json'))
        
        cnt = 0
        for label_file in input_files:
            with open(label_file, 'r') as f:
                label_data = json.load(f)
                filename = label_data['imagePath'][9:]
            img = self.read_image(filename)
            height, width = img.shape[:2]
            point = self.extract_annotation(label_data, height, width)
            
            detect = False
            output = self.predictor[0](img) # main
            isinstance = output["instances"]
            boxes = isinstance.pred_boxes.tensor.cpu().numpy()
            self.save_image(img, output, filename.split('.')[-1] + "_main.png")
            ok = self.is_substance(boxes, point)
            if ok: detect = True
        
            if detect:
                print(f"detected: {filename}")
                cnt += 1
            else:
                print(f"not detected: {filename}")
                pass
        print(f"detect count: {cnt}/{len(input_files)}")
        
    
    def check_inspect(self):
        input_files = glob.glob(os.path.join(self.input_dir, '*.json'))
        
        cnt = 0
        for label_file in input_files:
            with open(label_file, 'r') as f:
                label_data = json.load(f)
                filename = label_data['imagePath'][9:] # 要確認
            img = self.read_image(filename)
            height, width = img.shape[:2]
            point = self.extract_annotation(label_data, height, width)
            
            slice_number = int(filename[-8:-4])
            detect = False
            for i in range(2):
                outputs = self.predictor[i](img)
                isinstance = outputs["instances"]
                boxes = isinstance.pred_boxes.tensor.cpu().numpy()
                boxes =self.remove_boxes(img, boxes, slice_number)
                self.save_image(img, outputs, filename.split('/')[-1].split('.')[0] + f"_{i}" + ".png")
                ok = self.is_substance(boxes, point)
                if ok: detect = True
                
            
            if slice_number < 185 or slice_number > 253:
                if detect:
                    print(f"detected: {filename}")
                    cnt += 1
                else:
                    print(f"not detected: {filename}")
                continue
            
            # bead部
            crop_imgs = self.crop_images(img)
            for i, bead_img in enumerate(crop_imgs):
                point = self.extract_bead_annotation(label_data, height, width, right=i)
                outputs = self.predictor[2](bead_img)
                isinstance = outputs["instances"]
                boxes = isinstance.pred_boxes.tensor.cpu().numpy()
                self.save_image(bead_img, outputs, filename.split('/')[-1].split('.')[0] + f"_{i+2}" + ".png")
                ok = self.is_substance(boxes, point)
                if ok: detect = True
            
            if detect:
                print(f"detected: {filename}")
                cnt += 1
            else:
                print(f"not detected: {filename}")
                pass
        print(f"detect count: {cnt}/{len(input_files)}")
        
    def inspect(self, img, slice_number, save=False, saveID=None, is_positive_part=True):
        height, width = img.shape[:2]
        detected_areas = np.empty((0, 4), int)
        
        for i in range(2):
            outputs = self.predictor[i](img)
            isinstance = outputs["instances"]
            boxes = isinstance.pred_boxes.tensor.cpu().numpy()
            detected_areas = np.concatenate([detected_areas, boxes], axis=0)
            if save: self.save_image(img, outputs, saveID + f"_{i}" + ".png")
                
        if 185 <= slice_number <= 253:
            # bead部
            crop_imgs = self.crop_images(img)
            for i, bead_img in enumerate(crop_imgs):
                outputs = self.predictor[2](bead_img)
                isinstance = outputs["instances"]
                boxes = isinstance.pred_boxes.tensor.cpu().numpy()
                boxes = self.convert_coordinate(boxes, height, width, i)
                detected_areas = np.concatenate([detected_areas, boxes], axis=0)
                if save: self.save_image(bead_img, outputs, saveID + f"_{i+2}" + ".png")

        detected_areas = self.remove_boxes(img, detected_areas, slice_number, is_positive_part)
        detected_areas = self.merge_boxes(detected_areas)
        if len(detected_areas) != 0:
            self.save_image2(img, detected_areas, saveID + ".png")
            open(os.path.join(self.output_dir, saveID + ".txt"), "w").write(json.dumps(detected_areas.tolist()))

        return detected_areas
    
    def inspect_one_cell(self):
        input_path = glob.escape(self.input_dir)
        input_files = glob.glob(os.path.join(input_path, "*.png"))
        for filename in input_files:
            fileID = f"{filename.split('_')[-3]}_{filename.split('_')[-2]}"
            slice_number = int(filename[-8:-4])
            img = self.read_image(filename)
            boxes = self.inspect(img, slice_number, save=True, saveID=f"{fileID}_{slice_number}")
            print("detected: ", len(boxes))
        

if __name__ == "__main__":
    import time
    start = time.time()
    # input_dir = "/workspace/data/NG_data/1GP170529A0214_正極_20170708_180638/[Z軸]"   
    # weights_dir = "/workspace/weights/Zaxis/"
    # weights_list = [("model_main.pth", 0.63), ("model_thin.pth", 0.8), ("model_bead.pth", 0.1)]
    # # weights_list = ["model_main.pth", "model_thin.pth", "model_bead.pth"]
    # output_dir = "/workspace/data/results/Zaxis/170529P_test"
    # os.makedirs(output_dir, exist_ok=True)
    # inspector = InspectorZaxis(input_dir, weights_dir, weights_list, output_dir)
    # inspector.inspect_one_cell()
    
    input_dir = "/workspace/data/substance/single/Z_json"   
    weights_dir = "/workspace/weights/Zaxis/"
    weights_list = [("model_main.pth", 0.63), ("model_thin.pth", 0.8), ("model_bead.pth", 0.1)]
    output_dir = "/workspace/data/results/Zaxis/ng_data_test"
    os.makedirs(output_dir, exist_ok=True)
    inspector = InspectorZaxis(input_dir, weights_dir, weights_list, output_dir)
    inspector.check_inspect()
    
    print(f"elapsed time: {(time.time() - start)/60:.2f}min")
    
    