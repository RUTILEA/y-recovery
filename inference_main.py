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
from tqdm import tqdm

class InferenceMain:
    def __init__(self, input_dir, weight_dir, weights_list, output_dir, gpu_id):
        self.input_dir = input_dir
        self.weight_dir = weight_dir
        self.output_dir = output_dir
        self._set_predictor(weights_list, gpu_id)
        
    
    def _set_predictor(self, weights_list, gpu_id):
        input_dir = os.path.join(self.input_dir, "[Z軸]")
        weights_dir = os.path.join(self.weight_dir, "Zaxis")
        output_dir = os.path.join(self.output_dir, "Zaxis")
        os.makedirs(output_dir, exist_ok=True)
        self.z_inspector = InspectorZaxis(input_dir, weights_dir, weights_list['Zaxis'], output_dir, gpu_id)
        
        input_dir = os.path.join(self.input_dir, "[oblique1]")
        weights_dir = os.path.join(self.weight_dir, "oblique1")
        output_dir = os.path.join(self.output_dir, "oblique1")
        os.makedirs(output_dir, exist_ok=True)
        self.ob1_inspector = InspectorOblique1(input_dir, weights_dir, weights_list['oblique1'], output_dir, gpu_id)
        
        input_dir = os.path.join(self.input_dir, "[oblique2]")
        weights_dir = os.path.join(self.weight_dir, "oblique2")
        output_dir = os.path.join(self.output_dir, "oblique2")
        os.makedirs(output_dir, exist_ok=True)
        self.ob2_inspector = InspectorOblique2(input_dir, weights_dir, weights_list['oblique2'], output_dir, gpu_id)
    
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
            index_oblique_min = int(xl)
            index_oblique_max = int(xr) + 1
            P.append((index_oblique_min, index_oblique_max, x_oblique, y_oblique))
        return P
    
    def convert_Z_to_oblique2(self, boxes, z_index):
        P = []
        for box in boxes:
            xl, yl, xr, yr = map(int, box)
            center_x, center_y = (xl + xr) // 2, (yl + yr) // 2
            x_oblique = center_x * (3101/1024)
            y_oblique = 256 - z_index
            index_oblique_min = int(yl)
            index_oblique_max = int(yr) + 1
            P.append((index_oblique_min, index_oblique_max, x_oblique, y_oblique))
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
        z_filename = filename
        img = self.z_inspector.read_image(filename)
        z_index = self.extract_index(filename)
        is_positive_part = os.path.basename(filename).split('_')[1] == '正極'
        boxes_z = self.z_inspector.inspect(img, slice_number, save=False, saveID=f"{fileID}_{slice_number}", is_positive_part=is_positive_part)
        # print(f"filename: {filename}, is_positive_part: {is_positive_part} z_index: {z_index}, boxes: {boxes_z}")
        # return len(boxes_z) > 0
        
        is_detected = False
        detected_boxes = []
        detected_box_indeces = set()
        ob2_boxes = self.convert_Z_to_oblique2(boxes_z, z_index)
        ob2_result = {}
        for i, p in enumerate(ob2_boxes):
            if i in detected_box_indeces:
                continue
            # idx = p[0]; x = p[1]; y = p[2]
            idx_min, idx_max, x, y = p
            for idx in range(idx_min, idx_max+1):
                filename = 'oblique2_' + fileID + f"_{idx:04}.png"
                if not os.path.exists(os.path.join(self.ob2_inspector.input_dir, filename)):
                    # print(os.path.join(self.ob2_inspector.input_dir, filename))
                    continue
                else:
                    pass
                if filename not in ob2_result:
                    img_ob2 = self.ob2_inspector.read_image(filename)
                    boxes_ob2 = self.ob2_inspector.inspect(img_ob2, save=False, saveID=f"{fileID}_{z_index}_{idx}")
                    ob2_result[filename] = boxes_ob2
                else:
                    boxes_ob2 = ob2_result[filename]
                for boxes in boxes_ob2:
                    boxes = self.merge_boxes(boxes)
                    for box in boxes:
                        center_x, center_y = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
                        if abs(center_x - x) < 20 and abs(center_y - y) < 20:
                        # if True:
                            if i in detected_box_indeces:
                                continue
                            # print(f"detected! ob2_index:{idx}, z_x:{x}, z_y:{y}")
                            # print('detect_area:', center_x, center_y)
                            # return True
                            is_detected = True
                            detected_boxes.append(["ob2", (idx, x, y), boxes_z[i].tolist()])
                            detected_box_indeces.add(i)
                            break
                if i in detected_box_indeces:
                    break
                        
        # ob1_boxes = self.convert_Z_to_oblique1(boxes_z, z_index)
        # ob1_result = {}
        # for i, p in enumerate(ob1_boxes):
        #     if i in detected_box_indeces:
        #         continue
        #     # idx = p[0]; x = p[1]; y = p[2]
        #     idx_min, idx_max, x, y = p
        #     for idx in range(idx_min, idx_max+1):
        #         filename = 'oblique1_' + fileID + f"_{idx:04}.png"
        #         # print(filename)
        #         if not os.path.exists(os.path.join(self.ob1_inspector.input_dir, filename)):
        #             # print(f"not found: {filename}")
        #             continue
        #         else:
        #             # print(f"found: {filename}")
        #             pass
        #         if filename not in ob1_result:
        #             img_ob1 = self.ob1_inspector.read_image(filename)
        #             boxes_ob1 = self.ob1_inspector.inspect(img_ob1, save=True, saveID=f"{fileID}_{z_index}_{idx}")
        #             ob1_result[filename] = boxes_ob1
        #         else:
        #             boxes_ob1 = ob1_result[filename]
        #         for box in boxes_ob1:
        #             center_x, center_y = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
        #             if abs(center_x - x) < 20 and abs(center_y - y) < 20:
        #             # if True:
        #                 if i in detected_box_indeces:
        #                     continue
        #                 # print(f"detected! z_index:{z_index}, z_x:{x}, z_y:{y}")
        #                 # return True
        #                 is_detected = True
        #                 detected_boxes.append(["ob1", (idx, x, y), boxes_z[i].tolist()])
        #                 detected_box_indeces.add(i)
        #                 break
        #         if i in detected_box_indeces:
        #             break

        # print(detected_boxes)
        if len(detected_boxes) != 0:
            self.save_image(img, [box[2] for box in detected_boxes], z_filename)
            json.dump(detected_boxes, open(os.path.join(self.output_dir, f"{fileID}_{z_index}.json"), 'w'))
        # return False
        return is_detected

    def save_image(self, img, boxes_z, z_filename):
        import cv2
        output_image = img.copy()
        os.makedirs(os.path.join(self.output_dir, "detected"), exist_ok=True)
        for box in boxes_z:
            xl, yl, xr, yr = map(int, box)
            cv2.rectangle(output_image, (xl, yl), (xr, yr), (0, 255, 0), 1)
        output_path = os.path.join(self.output_dir, "detected", os.path.basename(z_filename))
        cv2.imwrite(output_path, output_image)
        
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
        input_files = glob.glob(os.path.join(input_path, "*.png"))
        
        # NGデータの場合のみ、不良がある層のみに制限
        data = json.load(open("complete_correct_data.json"))
        data_name = os.path.basename(os.path.dirname(input_path)).split('_')[0]
        valid_input_files = sorted([file for file in input_files if data_name not in data or file.endswith(f"{data[data_name]['pole']}_{data_name}_{data[data_name]['z'].zfill(4)}.png")])
        # print(valid_input_files)
        
        detection_result = False
        for filename in valid_input_files:
            fileID = f"{filename.split('_')[-3]}_{filename.split('_')[-2]}"
            # print('name:', filename, 'id:', fileID)
            is_detected = self.inspect(filename, fileID)
            # print(f"filename: {filename.split('/')[-1]}, detected: {is_detected}")
            if is_detected:
                detection_result = True
                return True
        return detection_result
        # return False

    def inspect_cells(self):
        cnt = 0
        input_dirs = [folder for folder in os.listdir(self.input_dir) if folder != ".gitkeep"]
        results = {input_dir: None for input_dir in input_dirs}
        for input_dir in tqdm(input_dirs):
            cell_path = os.path.join(self.input_dir, input_dir)
            cellID = f"{input_dir.split('_')[0]}_{input_dir.split('_')[1]}"
            self.z_inspector.input_dir = os.path.join(cell_path, "[Z軸]")
            self.ob1_inspector.input_dir = os.path.join(cell_path, "[oblique1]")
            self.ob2_inspector.input_dir = os.path.join(cell_path, "[oblique2]")
            is_detected = self.inspect_one_cell(self.input_dir)
            results[input_dir] = is_detected
            if is_detected:
                cnt += 1
                # print(f"detected: {cellID}")
            else:
                pass
                # print(f"not detected: {cellID}")
                
        print(f"detect count: {cnt}/{len(input_dirs)}")
        with open(os.path.join(self.output_dir, "results.json"), 'w') as f:
            json.dump(results, f, indent=4)
            
    
if __name__ == '__main__':
    import time
    start = time.time()
    input_dir = "/workspace/data/NG_data_B"
    weights_dir = "/workspace/weights/"
    # Zaxis(model_main.pth) < 0.64, (model_thin.pth) < 0.69, (model_bead.pth) < 0.13
    # oblique1(model_main.pth) < ?
    # oblique2(model_main.pth) < 0.12, (model_sub.pth) < 0.33, (model_small.pth) < 1
    weights_list = {'Zaxis': [("model_main.pth", 0.63), ("model_thin.pth", 0.68), ("model_bead.pth", 0.12)],\
                    'oblique1': [("model_main.pth", 1)],\
                    'oblique2': [("model_main.pth", 0.11), ("model_sub.pth", 0.32), ("model_small.pth", 0.99)]}
    
    output_dir = "/workspace/data/results/NG_data_B"
    gpu_id = 4
    
    import sys
    if len(sys.argv) == 4:
        input_dir, output_dir, gpu_id = sys.argv[1:]
        print(input_dir, output_dir, gpu_id)
        # exit()
    
    # input_dir, output_dir, gpu_id = "/workspace/data/OK_data1", "/workspace/data/results/OK_data1", 0
    # input_dir, output_dir, gpu_id = "/workspace/data/OK_data2", "/workspace/data/results/OK_data2", 0
    # input_dir, output_dir, gpu_id = "/workspace/data/OK_data3", "/workspace/data/results/OK_data3", 0
    # input_dir, output_dir, gpu_id = "/workspace/data/OK_data4", "/workspace/data/results/OK_data4", 3
    # input_dir, output_dir, gpu_id = "/workspace/data/OK_data5", "/workspace/data/results/OK_data5", 3
    # input_dir, output_dir, gpu_id = "/workspace/data/OK_data6", "/workspace/data/results/OK_data6", 3
    # input_dir, output_dir, gpu_id = "/workspace/data/OK_data7", "/workspace/data/results/OK_data7", 4
    # input_dir, output_dir, gpu_id = "/workspace/data/OK_data8", "/workspace/data/results/OK_data8", 4
    # input_dir, output_dir, gpu_id = "/workspace/data/OK_data9", "/workspace/data/results/OK_data9", 4
    # input_dir, output_dir, gpu_id = "/workspace/data/OK_data10", "/workspace/data/results/OK_data10", 5
    # input_dir, output_dir, gpu_id = "/workspace/data/OK_data11", "/workspace/data/results/OK_data11", 5
    # input_dir, output_dir, gpu_id = "/workspace/data/OK_data12", "/workspace/data/results/OK_data12", 5
    # input_dir, output_dir, gpu_id = "/workspace/data/OK_data13", "/workspace/data/results/OK_data13", 0
    
    # input_dir, output_dir, gpu_id = "/workspace/data/NG_data_A1", "/workspace/data/results/NG_data_A1", 0
    # input_dir, output_dir, gpu_id = "/workspace/data/NG_data_A2", "/workspace/data/results/NG_data_A2", 0
    # input_dir, output_dir, gpu_id = "/workspace/data/NG_data_A3", "/workspace/data/results/NG_data_A3", 0
    # input_dir, output_dir, gpu_id = "/workspace/data/NG_data_A4", "/workspace/data/results/NG_data_A4", 3
    # input_dir, output_dir, gpu_id = "/workspace/data/NG_data_A5", "/workspace/data/results/NG_data_A5", 3
    # input_dir, output_dir, gpu_id = "/workspace/data/NG_data_A6", "/workspace/data/results/NG_data_A6", 3
    # input_dir, output_dir, gpu_id = "/workspace/data/NG_data_B1", "/workspace/data/results/NG_data_B1", 4
    # input_dir, output_dir, gpu_id = "/workspace/data/NG_data_B2", "/workspace/data/results/NG_data_B2", 4
    # input_dir, output_dir, gpu_id = "/workspace/data/NG_data_B3", "/workspace/data/results/NG_data_B3", 4
    # input_dir, output_dir, gpu_id = "/workspace/data/NG_data_B4", "/workspace/data/results/NG_data_B4", 5
    # input_dir, output_dir, gpu_id = "/workspace/data/NG_data_B5", "/workspace/data/results/NG_data_B5", 5
    # input_dir, output_dir, gpu_id = "/workspace/data/NG_data_B6", "/workspace/data/results/NG_data_B6", 5
    
    
    # input_dir, output_dir, gpu_id = "/workspace/data/OK_data", "/workspace/data/results/OK_data", 0
    # input_dir, output_dir, gpu_id = "/workspace/data/NG_data_A", "/workspace/data/results/NG_data_A", 3
    # input_dir, output_dir, gpu_id = "/workspace/data/NG_data_B", "/workspace/data/results/NG_data_B", 4
    
    inference = InferenceMain(input_dir, weights_dir, weights_list, output_dir, gpu_id)
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
