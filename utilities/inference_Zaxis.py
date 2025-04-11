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

from .config import logger


class InspectorZaxis:
    def __init__(self, input_dir, weight_dir, weight_list, output_dir):
        self.input_dir = input_dir
        self.weight_dir = weight_dir
        self.weight_list = weight_list
        self.output_dir = output_dir
        self._set_predictor(weight_list)
        self.cell_size = {"width": 1000, "height": 250}
        self.bead_area = {
            "left": [80, 30, 320, 220],
            "right": [680, 30, 920, 220],
        }  # x1, y1, x2, y2
        self._set_exclusion_area()

    def _setup_cfg(self, gpu_id, weight_path, threshold):
        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
            )
        )
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.MODEL.WEIGHTS = weight_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        cfg.DATASETS.TEST = (
            "val2",
            "my_dataset_val3",
        )
        cfg.MODEL.DEVICE = f"cuda:{gpu_id}"
        return cfg

    def _set_predictor(self, weight_list):
        self.predictor = []
        for weight, thresh in weight_list:
            weight_path = os.path.join(self.weight_dir, weight)
            cfg = self._setup_cfg(0, weight_path, thresh)
            p = DefaultPredictor(cfg)
            self.predictor.append(p)
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

    def _set_exclusion_area(self):
        self.exclusion_area = [
            (20, 600, 0, 40, 620, 40),
            (20, 410, 0, 40, 430, 40),
            (990, 410, 0, 1010, 430, 40),
            (990, 600, 0, 1010, 620, 40),
            (495, 405, 82, 505, 620, 165),
            (515, 405, 82, 525, 620, 165),
            (355, 405, 90, 370, 620, 165),
            (645, 405, 90, 665, 620, 165),
            (295, 405, 90, 315, 620, 165),
            (705, 405, 90, 730, 620, 165),
            (920, 405, 90, 930, 620, 165),
            (95, 405, 90, 105, 620, 165),
            (460, 460, 110, 475, 570, 165),
            (540, 460, 110, 555, 570, 165),
        ]
        # (80, 395, 230, 990, 405, 240), (80, 620, 230, 990, 630, 240)]

    def black_boxes(self, image, boxes):
        new_boxes = np.empty((0, 4), int)
        buff = 10
        height, width = image.shape[:2]
        for box in boxes:
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            surrounding_area = image[max(0, y1 - buff) : y1, x1:x2].reshape(-1, 3)
            surrounding_area = np.concatenate(
                [
                    surrounding_area,
                    image[y2 : min(height, y2 + buff), x1:x2].reshape(-1, 3),
                ],
                axis=0,
            )
            surrounding_area = np.concatenate(
                [surrounding_area, image[y1:y2, max(0, x1 - buff) : x1].reshape(-1, 3)],
                axis=0,
            )
            surrounding_area = np.concatenate(
                [
                    surrounding_area,
                    image[y1:y2, x2 : min(width, x2 + buff)].reshape(-1, 3),
                ],
                axis=0,
            )
            anomaly = image[y1:y2, x1:x2]
            flag = np.mean(anomaly) - np.mean(surrounding_area) > -10
            if flag:
                new_boxes = np.concatenate([new_boxes, box.reshape(1, 4)], axis=0)
        return new_boxes

    def read_image(self, filename):
        image_path = os.path.join(self.input_dir, filename)
        img = cv2.imread(image_path)
        return img

    def extract_annotation(self, label_data, height, width):
        for shape in label_data["shapes"]:
            points = shape["points"]
            (x1, y1), (x2, y2) = points
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            P = [(x1, y1), (x2, y2)]
        return P

    def extract_bead_annotation(self, label_data, height, width, right=False):
        for shape in label_data["shapes"]:
            points = shape["points"]
            (x1, y1), (x2, y2) = points
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            center_x, center_y = width // 2, height // 2
            if right:
                P = [
                    (
                        x1
                        - (
                            center_x
                            - self.cell_size["width"] // 2
                            + self.bead_area["right"][0]
                        ),
                        (
                            y1
                            - (
                                center_y
                                - self.cell_size["height"] // 2
                                + self.bead_area["right"][1]
                            )
                        ),
                    ),
                    (
                        x2
                        - (
                            center_x
                            - self.cell_size["width"] // 2
                            + self.bead_area["right"][0]
                        ),
                        (
                            y2
                            - (
                                center_y
                                - self.cell_size["height"] // 2
                                + self.bead_area["right"][1]
                            )
                        ),
                    ),
                ]
            else:
                P = [
                    (
                        x1
                        - (
                            center_x
                            - self.cell_size["width"] // 2
                            + self.bead_area["left"][0]
                        ),
                        (
                            y1
                            - (
                                center_y
                                - self.cell_size["height"] // 2
                                + self.bead_area["left"][1]
                            )
                        ),
                    ),
                    (
                        x2
                        - (
                            center_x
                            - self.cell_size["width"] // 2
                            + self.bead_area["left"][0]
                        ),
                        (
                            y2
                            - (
                                center_y
                                - self.cell_size["height"] // 2
                                + self.bead_area["left"][1]
                            )
                        ),
                    ),
                ]

        return P

    def crop_images(self, img):
        height, width = img.shape[:2]
        center_x, center_y = width // 2, height // 2
        left_bead = img[
            center_y
            - self.cell_size["height"] // 2
            + self.bead_area["left"][1] : center_y
            - self.cell_size["height"] // 2
            + self.bead_area["left"][3],
            center_x
            - self.cell_size["width"] // 2
            + self.bead_area["left"][0] : center_x
            - self.cell_size["width"] // 2
            + self.bead_area["left"][2],
        ]
        right_bead = img[
            center_y
            - self.cell_size["height"] // 2
            + self.bead_area["right"][1] : center_y
            - self.cell_size["height"] // 2
            + self.bead_area["right"][3],
            center_x
            - self.cell_size["width"] // 2
            + self.bead_area["right"][0] : center_x
            - self.cell_size["width"] // 2
            + self.bead_area["right"][2],
        ]
        return [left_bead, right_bead]

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

    def remove_boxes(self, image, boxes, z_index):
        new_boxes = np.empty((0, 4), int)
        for box in boxes:
            for x1, y1, z1, x2, y2, z2 in self.exclusion_area:
                if z1 <= z_index <= z2 and self.check_overlap(box, [x1, y1, x2, y2]):
                    break
            else:
                new_boxes = np.concatenate([new_boxes, box.reshape(1, 4)], axis=0)
        new_boxes = self.black_boxes(image, new_boxes)
        return new_boxes

    def check_overlap(self, box1, box2):
        if box1[2] <= box2[0] or box2[2] <= box1[0]:
            return False
        if box1[3] <= box2[1] or box2[3] <= box1[1]:
            return False
        return True

    def convert_coordinate(self, boxes, height, width, right):
        new_boxes = np.empty((0, 4), int)
        center_x, center_y = width // 2, height // 2
        if right:
            for box in boxes:
                box[0] += (
                    center_x - self.cell_size["width"] // 2 + self.bead_area["right"][0]
                )
                box[1] += (
                    center_y
                    - self.cell_size["height"] // 2
                    + self.bead_area["right"][1]
                )
                box[2] += (
                    center_x - self.cell_size["width"] // 2 + self.bead_area["right"][0]
                )
                box[3] += (
                    center_y
                    - self.cell_size["height"] // 2
                    + self.bead_area["right"][1]
                )
                new_boxes = np.concatenate([new_boxes, box.reshape(1, 4)], axis=0)
        else:
            for box in boxes:
                box[0] += (
                    center_x - self.cell_size["width"] // 2 + self.bead_area["left"][0]
                )
                box[1] += (
                    center_y - self.cell_size["height"] // 2 + self.bead_area["left"][1]
                )
                box[2] += (
                    center_x - self.cell_size["width"] // 2 + self.bead_area["left"][0]
                )
                box[3] += (
                    center_y - self.cell_size["height"] // 2 + self.bead_area["left"][1]
                )
                new_boxes = np.concatenate([new_boxes, box.reshape(1, 4)], axis=0)
        return new_boxes

    def check_inspect_one_model(self, model_idx=0):
        input_files = glob.glob(os.path.join(self.input_dir, "*.json"))

        cnt = 0
        for label_file in input_files:
            with open(label_file, "r") as f:
                label_data = json.load(f)
                filename = label_data["imagePath"][9:]
            img = self.read_image(filename)
            height, width = img.shape[:2]
            point = self.extract_annotation(label_data, height, width)

            detect = False
            output = self.predictor[0](img)  # main
            isinstance = output["instances"]
            boxes = isinstance.pred_boxes.tensor.cpu().numpy()
            self.save_image(img, output, filename.split(".")[-1] + "_main.png")
            ok = self.is_substance(boxes, point)
            if ok:
                detect = True

            if detect:
                print(f"detected: {filename}")
                cnt += 1
            else:
                print(f"not detected: {filename}")
                pass
        print(f"detect count: {cnt}/{len(input_files)}")

    def check_inspect(self):
        input_files = glob.glob(os.path.join(self.input_dir, "*.json"))

        cnt = 0
        for label_file in input_files:
            with open(label_file, "r") as f:
                label_data = json.load(f)
                filename = label_data["imagePath"][9:]  # 要確認
            img = self.read_image(filename)
            height, width = img.shape[:2]
            point = self.extract_annotation(label_data, height, width)

            slice_number = int(filename[-8:-4])
            detect = False
            for i in range(2):
                outputs = self.predictor[i](img)
                isinstance = outputs["instances"]
                boxes = isinstance.pred_boxes.tensor.cpu().numpy()
                boxes = self.remove_boxes(img, boxes, slice_number)
                self.save_image(
                    img,
                    outputs,
                    filename.split("/")[-1].split(".")[0] + f"_{i}" + ".png",
                )
                ok = self.is_substance(boxes, point)
                if ok:
                    detect = True

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
                self.save_image(
                    bead_img,
                    outputs,
                    filename.split("/")[-1].split(".")[0] + f"_{i+2}" + ".png",
                )
                ok = self.is_substance(boxes, point)
                if ok:
                    detect = True

            if detect:
                print(f"detected: {filename}")
                cnt += 1
            else:
                print(f"not detected: {filename}")
                pass
        print(f"detect count: {cnt}/{len(input_files)}")

    def inspect(self, img, slice_number, save=False, saveID=None):
        height, width = img.shape[:2]
        detected_areas = np.empty((0, 4), int)

        for i in range(2):
            outputs = self.predictor[i](img)
            isinstance = outputs["instances"]
            boxes = isinstance.pred_boxes.tensor.cpu().numpy()

            detected_areas = np.concatenate([detected_areas, boxes], axis=0)
            detected_areas = self.remove_boxes(img, detected_areas, slice_number)
            if save and len(boxes) > 0:
                self.save_image(img, outputs, saveID + f"_{i}" + ".png")

                logger.info(f"saved: defect detected on Zaxis at {saveID}_{i}.png")
                logger.info("*" * 20)

        if slice_number < 185 or slice_number > 253:
            return detected_areas

        # bead部
        crop_imgs = self.crop_images(img)
        for i, bead_img in enumerate(crop_imgs):
            outputs = self.predictor[2](bead_img)
            isinstance = outputs["instances"]
            boxes = isinstance.pred_boxes.tensor.cpu().numpy()
            boxes = self.convert_coordinate(boxes, height, width, i)
            detected_areas = np.concatenate([detected_areas, boxes], axis=0)
            # if save:
            #     self.save_image(bead_img, outputs, saveID + f"_{i+2}" + ".png")
        # return detected areas and confidence score
        return detected_areas

    def inspect_one_cell(self):
        input_path = glob.escape(self.input_dir)
        input_files = glob.glob(os.path.join(input_path, "*.png"))
        for filename in input_files:
            fileID = f"{filename.split('_')[-3]}_{filename.split('_')[-2]}"
            slice_number = int(filename[-8:-4])
            img = self.read_image(filename)
            boxes = self.inspect(
                img, slice_number, save=True, saveID=f"{fileID}_{slice_number}"
            )
            print("detected: ", len(boxes))
