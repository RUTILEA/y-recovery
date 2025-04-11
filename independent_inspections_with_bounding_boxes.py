import os
import glob
import json
import time
import yaml
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

from detectron2.data import MetadataCatalog
import numpy as np
from tqdm import tqdm

from utilities.inference_Zaxis import InspectorZaxis
from utilities.inference_ob1 import InspectorOblique1
from utilities.inference_ob2 import InspectorOblique2
from utilities.config import logger, load_config


class IndependentInspection:
    def __init__(self, input_dir, weight_dir, weights_list, output_dir, max_workers=3):
        self.input_dir = input_dir
        self.weight_dir = weight_dir
        self.output_dir = output_dir
        self.weights_list = weights_list
        self.max_workers = max_workers
        self._set_predictors(weights_list)
        # Create metadata for visualization
        self.metadata = MetadataCatalog.get("__unused")
        self.metadata.thing_classes = ["defect"]  # Set your class names

    def _set_predictors(self, weights_list):
        # Z-axis inspector
        z_input_dir = os.path.join(self.input_dir, "[Z軸]")
        z_weights_dir = os.path.join(self.weight_dir, "Zaxis")
        z_output_dir = os.path.join(self.output_dir, "Zaxis")
        os.makedirs(z_output_dir, exist_ok=True)
        self.z_inspector = InspectorZaxis(
            z_input_dir, z_weights_dir, weights_list["Zaxis"], z_output_dir
        )

        # Oblique1 inspector
        ob1_input_dir = os.path.join(self.input_dir, "[oblique1]")
        ob1_weights_dir = os.path.join(self.weight_dir, "oblique1")
        ob1_output_dir = os.path.join(self.output_dir, "oblique1")
        os.makedirs(ob1_output_dir, exist_ok=True)
        self.ob1_inspector = InspectorOblique1(
            ob1_input_dir, ob1_weights_dir, weights_list["oblique1"], ob1_output_dir
        )

        # Oblique2 inspector
        ob2_input_dir = os.path.join(self.input_dir, "[oblique2]")
        ob2_weights_dir = os.path.join(self.weight_dir, "oblique2")
        ob2_output_dir = os.path.join(self.output_dir, "oblique2")
        os.makedirs(ob2_output_dir, exist_ok=True)
        self.ob2_inspector = InspectorOblique2(
            ob2_input_dir, ob2_weights_dir, weights_list["oblique2"], ob2_output_dir
        )

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

    def inspect_zaxis(self, cell_path):
        """Process all Z-axis images in a cell independently"""
        input_dir = os.path.join(cell_path, "[Z軸]")
        self.z_inspector.input_dir = input_dir

        results = {
            "total_images": 0,
            "defect_detected": 0,
            "threshold_values": [weight[1] for weight in self.weights_list["Zaxis"]],
            "defect_info": [],
        }

        if not os.path.exists(input_dir):
            logger.warning(f"Z-axis directory not found: {input_dir}")
            return results

        input_files = glob.glob(os.path.join(glob.escape(input_dir), "*.png"))
        results["total_images"] = len(input_files)

        for filename in input_files:
            base_filename = os.path.basename(filename)
            fileID = f"{base_filename.split('_')[-3]}_{base_filename.split('_')[-2]}"
            slice_number = int(base_filename[-8:-4])

            img = self.z_inspector.read_image(filename)
            boxes = self.z_inspector.inspect(
                img, slice_number, save=True, saveID=f"{fileID}_{slice_number}"
            )

            boxes = self.merge_boxes(boxes)
            if len(boxes) > 0:
                results["defect_detected"] += 1
                results["defect_info"].append(
                    {"filename": base_filename, "num_defects": len(boxes)}
                )

        return results

    def inspect_oblique1(self, cell_path):
        """Process all Oblique1 images in a cell independently"""
        input_dir = os.path.join(cell_path, "[oblique1]")
        self.ob1_inspector.input_dir = input_dir

        results = {
            "total_images": 0,
            "defect_detected": 0,
            "threshold_values": [weight[1] for weight in self.weights_list["oblique1"]],
            "defect_info": [],
        }

        if not os.path.exists(input_dir):
            logger.warning(f"Oblique1 directory not found: {input_dir}")
            return results

        input_files = glob.glob(os.path.join(glob.escape(input_dir), "*.png"))
        results["total_images"] = len(input_files)

        for filename in input_files:
            base_filename = os.path.basename(filename)
            fileID = f"{base_filename.split('_')[-3]}_{base_filename.split('_')[-2]}"
            idx = int(base_filename[-8:-4])

            img = self.ob1_inspector.read_image(filename)
            boxes_list = self.ob1_inspector.inspect(
                img, save=True, saveID=f"{fileID}_{idx}"
            )

            defect_found = False
            total_defects = 0

            for boxes in boxes_list:
                boxes = self.merge_boxes(boxes)
                total_defects += len(boxes)
                if len(boxes) > 0:
                    defect_found = True

            if defect_found:
                results["defect_detected"] += 1
                results["defect_info"].append(
                    {"filename": base_filename, "num_defects": total_defects}
                )

        return results

    def inspect_oblique2(self, cell_path):
        """Process all Oblique2 images in a cell independently"""
        input_dir = os.path.join(cell_path, "[oblique2]")
        self.ob2_inspector.input_dir = input_dir

        results = {
            "total_images": 0,
            "defect_detected": 0,
            "threshold_values": [weight[1] for weight in self.weights_list["oblique2"]],
            "defect_info": [],
        }

        if not os.path.exists(input_dir):
            logger.warning(f"Oblique2 directory not found: {input_dir}")
            return results

        input_files = glob.glob(os.path.join(glob.escape(input_dir), "*.png"))
        results["total_images"] = len(input_files)

        for filename in input_files:
            base_filename = os.path.basename(filename)
            fileID = f"{base_filename.split('_')[-3]}_{base_filename.split('_')[-2]}"
            idx = int(base_filename[-8:-4])

            img = self.ob2_inspector.read_image(filename)
            boxes_list = self.ob2_inspector.inspect(
                img, save=True, saveID=f"{fileID}_{idx}"
            )

            defect_found = False
            total_defects = 0

            for boxes in boxes_list:
                boxes = self.merge_boxes(boxes)
                total_defects += len(boxes)
                if len(boxes) > 0:
                    defect_found = True

            if defect_found:
                results["defect_detected"] += 1
                results["defect_info"].append(
                    {"filename": base_filename, "num_defects": total_defects}
                )

        return results

    def inspect_cell(self, cell_path):
        """Inspect a single cell with all three inspectors in parallel"""
        cell_name = os.path.basename(cell_path)
        logger.info(f"Processing cell: {cell_name}")

        results = {}

        # Run all three inspections in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all three inspection tasks
            z_future = executor.submit(self.inspect_zaxis, cell_path)
            ob1_future = executor.submit(self.inspect_oblique1, cell_path)
            ob2_future = executor.submit(self.inspect_oblique2, cell_path)

            # Get results from all three inspections
            results["Z-axis"] = z_future.result()
            results["Oblique1"] = ob1_future.result()
            results["Oblique2"] = ob2_future.result()

        return results

    def inspect_all_cells(self):
        """Process all cells in the input directory"""
        # Get all cell directories
        cell_dirs = [
            os.path.join(self.input_dir, folder)
            for folder in os.listdir(self.input_dir)
            if folder != ".gitkeep"
            and os.path.isdir(os.path.join(self.input_dir, folder))
        ]

        # Create results dictionary
        all_results = {}
        summary = {
            "Z-axis": {
                "total_cells": 0,
                "cells_with_defects": 0,
                "total_images": 0,
                "images_with_defects": 0,
            },
            "Oblique1": {
                "total_cells": 0,
                "cells_with_defects": 0,
                "total_images": 0,
                "images_with_defects": 0,
            },
            "Oblique2": {
                "total_cells": 0,
                "cells_with_defects": 0,
                "total_images": 0,
                "images_with_defects": 0,
            },
        }

        # Process each cell
        for cell_path in tqdm(cell_dirs, desc="Processing cells"):
            cell_name = os.path.basename(cell_path)
            cell_results = self.inspect_cell(cell_path)
            all_results[cell_name] = cell_results

            # Update summary statistics
            for axis in ["Z-axis", "Oblique1", "Oblique2"]:
                axis_result = cell_results[axis]
                summary[axis]["total_cells"] += 1
                summary[axis]["total_images"] += axis_result["total_images"]
                summary[axis]["images_with_defects"] += axis_result["defect_detected"]
                if axis_result["defect_detected"] > 0:
                    summary[axis]["cells_with_defects"] += 1

        # Save results and summary to JSON files
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, "all_results.json"), "w") as f:
            json.dump(all_results, f, indent=4)

        with open(os.path.join(self.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=4)

        # Log summary information
        logger.info("Inspection complete!")
        for axis in ["Z-axis", "Oblique1", "Oblique2"]:
            logger.info(f"{axis} Summary:")
            logger.info(
                f"  Cells with defects: {summary[axis]['cells_with_defects']}/{summary[axis]['total_cells']}"
            )
            logger.info(
                f"  Images with defects: {summary[axis]['images_with_defects']}/{summary[axis]['total_images']}"
            )
            logger.info(
                f"  Detection rate: {summary[axis]['images_with_defects']/max(1, summary[axis]['total_images'])*100:.2f}%"
            )
            logger.info("=" * 50)

        return summary


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Independent inspection with YAML configuration"
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
    max_workers = config.get("max_workers", 3)  # Default to 3 if not specified

    # Log configuration
    logger.info(f"Using configuration from: {args.config}")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Weights directory: {weights_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Max workers: {max_workers}")

    # Create and run independent inspection
    inspector = IndependentInspection(
        input_dir, weights_dir, weights_list, output_dir, max_workers
    )
    summary = inspector.inspect_all_cells()

    # Log timing information
    elapsed_time = time.time() - start_time
    logger.info(f"Total elapsed time: {elapsed_time/60:.2f} minutes")
    logger.info("Independent inspection complete!")
