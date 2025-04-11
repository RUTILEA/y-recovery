import os
import glob
import json
import time
import argparse

from detectron2.data import MetadataCatalog

import numpy as np
from tqdm import tqdm


from utilities.inference_Zaxis import InspectorZaxis
from utilities.inference_ob1 import InspectorOblique1
from utilities.inference_ob2 import InspectorOblique2

from utilities.config import logger, load_config


class InferenceMain:
    def __init__(self, input_dir, weight_dir, weights_list, output_dir):
        self.input_dir = input_dir
        self.weight_dir = weight_dir
        self.output_dir = output_dir
        self._set_predictor(weights_list)
        # Create metadata for visualization
        self.metadata = MetadataCatalog.get("__unused")
        self.metadata.thing_classes = ["defect"]  # Set your class names

    def _set_predictor(self, weights_list):
        input_dir = os.path.join(self.input_dir, "[Z軸]")
        weights_dir = os.path.join(self.weight_dir, "Zaxis")
        output_dir = os.path.join(self.output_dir, "Zaxis")
        os.makedirs(output_dir, exist_ok=True)
        self.z_inspector = InspectorZaxis(
            input_dir, weights_dir, weights_list["Zaxis"], output_dir
        )

        input_dir = os.path.join(self.input_dir, "[oblique1]")
        weights_dir = os.path.join(self.weight_dir, "oblique1")
        output_dir = os.path.join(self.output_dir, "oblique1")
        os.makedirs(output_dir, exist_ok=True)
        self.ob1_inspector = InspectorOblique1(
            input_dir, weights_dir, weights_list["oblique1"], output_dir
        )

        input_dir = os.path.join(self.input_dir, "[oblique2]")
        weights_dir = os.path.join(self.weight_dir, "oblique2")
        output_dir = os.path.join(self.output_dir, "oblique2")
        os.makedirs(output_dir, exist_ok=True)
        self.ob2_inspector = InspectorOblique2(
            input_dir, weights_dir, weights_list["oblique2"], output_dir
        )

    def extract_index(self, filename):
        num = filename.split("_")[-1][:4]
        return int(num)

    def convert_Z_to_oblique1(self, boxes, z_index):
        P = []
        for box in boxes:
            xl, yl, xr, yr = map(int, box)
            center_x, center_y = (xl + xr) // 2, (yl + yr) // 2
            x_oblique = 3101 - (center_y) * (3101 / 1024)
            y_oblique = 256 - z_index
            index_oblique = center_x
            P.append((index_oblique, x_oblique, y_oblique))
        return P

    def convert_Z_to_oblique2(self, boxes, z_index):
        P = []
        for box in boxes:
            xl, yl, xr, yr = map(int, box)
            center_x, center_y = (xl + xr) // 2, (yl + yr) // 2
            x_oblique = center_x * (3101 / 1024)
            y_oblique = 256 - z_index
            index_oblique = center_y
            P.append((index_oblique, x_oblique, y_oblique))
        return P

    def merge_boxes(self, boxes):
        new_boxes = np.empty((0, 4), int)
        for b1 in boxes:
            # Ensure b1 is a valid bounding box with 4 elements
            if not isinstance(b1, (list, np.ndarray)) or len(b1) != 4:
                logger.warning(f"Invalid box skipped: {b1}")
                continue
            for b2 in new_boxes:
                if self.check_overlap(b1, b2):
                    break
            else:
                new_boxes = np.vstack([new_boxes, b1])  # Add valid box to new_boxes
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
        boxes_z = self.z_inspector.inspect(
            img, slice_number, save=True, saveID=f"{fileID}_{slice_number}"
        )
        boxes_z = self.merge_boxes(boxes_z)

        ob2_boxes = self.convert_Z_to_oblique2(boxes_z, z_index)

        for p in ob2_boxes:
            idx = p[0]
            x = p[1]
            y = p[2]
            filename = "oblique2_" + fileID + f"_{idx:04}.png"
            if not os.path.exists(os.path.join(self.ob2_inspector.input_dir, filename)):
                continue
            img_ob2 = self.ob2_inspector.read_image(filename)
            boxes_ob2 = self.ob2_inspector.inspect(
                img_ob2, save=True, saveID=f"{fileID}_{idx}"
            )

            for boxes in boxes_ob2:
                boxes = self.merge_boxes(boxes)
                for box in boxes:
                    center_x, center_y = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
                    if abs(center_x - x) < 20 and abs(center_y - y) < 20:

                        return True

        ob1_boxes = self.convert_Z_to_oblique1(boxes_z, z_index)
        for p in ob1_boxes:
            idx = p[0]
            x = p[1]
            y = p[2]
            filename = "oblique1_" + fileID + f"_{idx:04}.png"

            if not os.path.exists(os.path.join(self.ob1_inspector.input_dir, filename)):
                continue

            img_ob1 = self.ob1_inspector.read_image(filename)
            boxes_ob1 = self.ob1_inspector.inspect(
                img_ob1, save=True, saveID=f"{fileID}_{idx}"
            )

            for boxes in boxes_ob1:
                boxes = self.merge_boxes(boxes)
                for box in boxes:
                    center_x, center_y = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
                    if abs(center_x - x) < 20 and abs(center_y - y) < 20:

                        return True

        return False

    def inspect_one_cell(self, input_dir=None):
        if input_dir is None:
            input_dir = self.input_dir
        input_path = glob.escape(self.z_inspector.input_dir)
        input_files = glob.glob(os.path.join(input_path, "*.png"))
        for filename in input_files:
            fileID = f"{filename.split('_')[-3]}_{filename.split('_')[-2]}"

            is_detected = self.inspect(filename, fileID)

            if is_detected:
                return True
        return False

    def inspect_cells(self):
        cnt = 0
        input_dirs = [
            folder for folder in os.listdir(self.input_dir) if folder != ".gitkeep"
        ]
        results = {input_dir: None for input_dir in input_dirs}
        for input_dir in tqdm(input_dirs):
            cell_path = os.path.join(self.input_dir, input_dir)
            cellID = f"{input_dir.split('_')[0]}_{input_dir.split('_')[1]}"
            # Update all three inspectors' input directories
            self.z_inspector.input_dir = os.path.join(cell_path, "[Z軸]")
            self.ob1_inspector.input_dir = os.path.join(cell_path, "[oblique1]")
            self.ob2_inspector.input_dir = os.path.join(cell_path, "[oblique2]")
            is_detected = self.inspect_one_cell(self.input_dir)
            results[input_dir] = is_detected
            if is_detected:
                cnt += 1

                logger.info(f"detected: {cellID}")
                logger.info("*" * 20)
            else:

                logger.info(f"not detected: {cellID}")
                logger.info("*" * 20)

        logger.info(f"detect count: {cnt}/{len(input_dirs)}")
        logger.info("*" * 20)
        with open(os.path.join(self.output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=4)
            logger.info("Results saved to results.json")
            logger.info("*" * 20)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Dependent inspection with YAML configuration"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration YAML file",
    )
    args = parser.parse_args()

    # Start timing
    start_time = time.time()

    # Load configuration
    config = load_config(args.config)

    # Extract parameters from config
    input_dir = config["input_dir"]
    weights_dir = config["weights_dir"]
    output_dir = config["output_dir"]
    weights_list = config["weights_list"]

    # Log configuration
    logger.info(f"Using configuration from: {args.config}")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Weights directory: {weights_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create and run dependent inspection
    inference = InferenceMain(input_dir, weights_dir, weights_list, output_dir)
    results = inference.inspect_cells()

    # Log timing information
    elapsed_time = time.time() - start_time
    logger.info(f"Elapsed time: {elapsed_time/60:.2f} minutes")
    logger.info("*" * 20)
    logger.info("Dependent inspection complete!")
