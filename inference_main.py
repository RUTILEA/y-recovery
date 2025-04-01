from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from inference_Zaxis import InspectorZaxis
from inference_ob1 import InspectorOblique1
from inference_ob2 import InspectorOblique2
import glob
import os
import json
import numpy as np

class InferenceMain:
    def __init__(self, input_dir, weight_dir, weights_list, output_dir):
        self.input_dir = input_dir
        self.weight_dir = weight_dir
        self.output_dir = output_dir
        self._set_predictor(weights_list)
        
    
    def _set_predictor(self, weights_list):
        input_dir = os.path.join(self.input_dir, "[Z軸]")
        weights_dir = os.path.join(self.weight_dir, "Zaxis")
        output_dir = os.path.join(self.output_dir, "Zaxis")
        os.makedirs(output_dir, exist_ok=True)
        self.z_inspector = InspectorZaxis(input_dir, weights_dir, weights_list['Zaxis'], output_dir)
        
        input_dir = os.path.join(self.input_dir, "[オブリーク1]")
        weights_dir = os.path.join(self.weight_dir, "oblique1")
        output_dir = os.path.join(self.output_dir, "oblique1")
        os.makedirs(output_dir, exist_ok=True)
        self.ob1_inspector = InspectorOblique1(input_dir, weights_dir, weights_list['oblique1'], output_dir)
        
        input_dir = os.path.join(self.input_dir, "[オブリーク2]")
        weights_dir = os.path.join(self.weight_dir, "oblique2")
        output_dir = os.path.join(self.output_dir, "oblique2")
        os.makedirs(output_dir, exist_ok=True)
        self.ob2_inspector = InspectorOblique2(input_dir, weights_dir, weights_list['oblique2'], output_dir)
    
    def extract_index(self, filename):
        num = filename.split('_')[-1][:4]
        return int(num)
        
    def convert_Z_to_oblique1(self, boxes, z_index):
        P = []
        for box in boxes:
            xl, yl, xr, yr = map(int, box)
            center_x, center_y = (xl + xr) // 2, (yl + yr) // 2
            x_oblique = 3101 - (center_y)*(3101/1024)
            y_oblique = 256 - z_index
            index_oblique = center_x
            P.append((index_oblique, x_oblique, y_oblique))
        return boxes
    
    def convert_Z_to_oblique2(self, boxes, z_index):
        P = []
        for box in boxes:
            xl, yl, xr, yr = map(int, box)
            center_x, center_y = (xl + xr) // 2, (yl + yr) // 2
            x_oblique = center_x * (3101/1024)
            y_oblique = 256 - z_index
            index_oblique = center_y
            P.append((index_oblique, x_oblique, y_oblique))
        return P
    
    def merge_boxes(self, boxes):
        new_boxes = np.empty((0, 4), int)
        for b1 in boxes:
            for b2 in new_boxes:
                if self.check_overlap(b1, b2):
                    break
            else:
                new_boxes = np.vstack([new_boxes, b1])
        return new_boxes
    
    def check_overlap(self, box1, box2):
        if box1[2] <= box2[0] or box2[2] <= box1[0]:
            return False
        if box1[3] <= box2[1] or box2[3] <= box1[1]:
            return False
        return True
    
    
    
    def inspect(self, filename, fileID):
        slice_number = int(filename[-8:-4])
        img = self.z_inspector.read_image(filename)
        z_index = self.extract_index(filename)
        boxes_z = self.z_inspector.inspect(img, slice_number, save=True, saveID=f"{fileID}_{slice_number}")
        boxes_z = self.merge_boxes(boxes_z)
        print(f"z_index: {z_index}, boxes: {boxes_z}")
        
        ob2_boxes = self.convert_Z_to_oblique2(boxes_z, z_index)
        for p in ob2_boxes:
            idx = p[0]; x = p[1]; y = p[2]
            filename = 'オブリーク2_' + fileID + f"_{idx:04}.tif"
            if not os.path.exists(os.path.join(self.ob2_inspector.input_dir, filename)):
                # print(os.path.join(self.ob2_inspector.input_dir, filename))
                continue
            else:
                pass
            img_ob2 = self.ob2_inspector.read_image(filename)
            boxes_ob2 = self.ob2_inspector.inspect(img_ob2, save=True, saveID=f"{fileID}_{idx}")
            for boxes in boxes_ob2:
                boxes = self.merge_boxes(boxes)
                for box in boxes:
                    center_x, center_y = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
                    if abs(center_x - x) < 20 and abs(center_y - y) < 20:
                        print(f"detected! ob2_index:{idx}, z_x:{x}, z_y:{y}")
                        print('detect_area:', center_x, center_y)
                        # del img; del img_ob2 
                        # import gc
                        # gc.collect() 
                        return True
                        
        ob1_boxes = self.convert_Z_to_oblique1(boxes_z, z_index)
        for p in ob1_boxes:
            idx = p[0]; x = p[1]; y = p[2]
            # for i in range(-1, 5):
            filename = 'オブリーク1_' + fileID + f"_{idx:04}.tif"
            print(filename)
            if not os.path.exists(os.path.join(self.ob1_inspector.input_dir, filename)):
                # print(f"not found: {filename}")
                continue
            else:
                # print(f"found: {filename}")
                pass
            img_ob1 = self.ob1_inspector.read_image(filename)
            boxes_ob1 = self.ob1_inspector.inspect(img_ob1, save=True, saveID=f"{fileID}_{idx}")
            for boxes in boxes_ob1:
                boxes = self.merge_boxes(boxes)
                for box in boxes:
                    center_x, center_y = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
                    if abs(center_x - x) < 20 and abs(center_y - y) < 20:
                        # print(f"detected! z_index:{z_index}, z_x:{x}, z_y:{y}")
                            return True
        return False
        
        
    def inspect_single_image(self):
        input_dir = os.path.join(self.input_dir, "[Z軸]")
        input_dir = glob.escape(input_dir)
        input_files = glob.glob(os.path.join(input_dir, "*.json"))
        cnt = 0
        for label_data in input_files:
            with open(label_data, 'r') as f:
                label_data = json.load(f)
                filename = label_data['imagePath'][9:]
            fileID = filename[:11]
            is_detected = self.inspect(filename, fileID)
            if is_detected:
                cnt += 1
                print(f"detected: {filename}")
            else:
                print(f"not detected: {filename}")
        print(f"detect count: {cnt}/{len(input_files)}")
            
                
    def inspect_one_cell(self, input_dir=None):
        if input_dir is None: input_dir = self.input_dir
        input_path = glob.escape(self.z_inspector.input_dir)
        input_files = glob.glob(os.path.join(input_path, "*.tif"))
        for filename in input_files:
            fileID = f"{filename.split('_')[-3]}_{filename.split('_')[-2]}"
            is_detected = self.inspect(filename, fileID)
            print(f"filename: {filename.split('/')[-1]}, detected: {is_detected}")
            if is_detected:
                return True
        return False
    
    def inspect_cells(self):
        cnt = 0
        input_dirs = os.listdir(self.input_dir)
        for input_dir in input_dirs:
            cell_path = os.path.join(self.input_dir, input_dir)
            cellID = f"{input_dir.split('_')[0]}_{input_dir.split('_')[1]}"
            self.z_inspector.input_dir = os.path.join(cell_path, "[Z軸]")
            # self.ob1_inspector.input_dir = os.path.join(cell_path, "[オブリーク1]")
            self.ob2_inspector.input_dir = os.path.join(cell_path, "[オブリーク2]")
            is_detected = self.inspect_one_cell(self.input_dir)
            if is_detected:
                cnt += 1
                print(f"detected: {cellID}")
            else:
                print(f"not detected: {cellID}")
                
        print(f"detect count: {cnt}/{len(input_dirs)}")
            
    
if __name__ == '__main__':
    import time
    start = time.time()
    input_dir = "/workspace/data/sample_test_data"   
    weights_dir = "/workspace/weights/"
    weights_list = {'Zaxis': [("model_main.pth", 0.63), ("model_thin.pth", 0.8), ("model_bead.pth", 0.1)],\
                    'oblique1': [("model_main.pth", 0.8)],\
                    'oblique2': [("model_main.pth", 0.1), ("model_sub.pth", 0.3), ("model_small.pth", 0.9)]}
    
    output_dir = "/workspace/data/results/sample_test_data"
    inference = InferenceMain(input_dir, weights_dir, weights_list, output_dir)
    inference.inspect_cells()
    
    # input_dir = "/workspace/data/substance/single/all"   
    # weights_dir = "/workspace/weights/"
    # weights_list = {'Zaxis': [("model_main.pth", 0.63), ("model_thin.pth", 0.8), ("model_bead.pth", 0.1)],\
    #                 'oblique1': [("model_main.pth", 0.8)],\
    #                 'oblique2': [("model_main.pth", 0.1), ("model_sub.pth", 0.2)]}
    # output_dir = "/workspace/data/results/main/single_all"
    # inference = InferenceMain(input_dir, weights_dir, weights_list, output_dir)
    # inference.inspect_single_image()
    
    print(f"elapsed time: {(time.time() - start)/60:.2f}min")
