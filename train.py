# データセットの登録
import os
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog

dataset_name = "train"
val_dataset_name = "valid"
dataset_dir = "/workspace/data/dataset/oblique2_crop3"

print(os.path.join(dataset_dir, "valid", "annotations.json"))
register_coco_instances(dataset_name, {},
                        os.path.join(dataset_dir, "train", "annotations.json"),
                        os.path.join(dataset_dir, "train"))

register_coco_instances(val_dataset_name, {},
                        os.path.join(dataset_dir, "valid", "annotations.json"),
                        os.path.join(dataset_dir, "valid"))

# データセットのメタデータを取得する
my_dataset_train_metadata = MetadataCatalog.get(dataset_name)
my_dataset_val_metadata = MetadataCatalog.get(val_dataset_name)


# 学習
import detectron2
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
import os
setup_logger()

# 設定をロードし、モデルを選択する
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = (dataset_name,)
cfg.DATASETS.TEST = (val_dataset_name,)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")  # 初期重み
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 5000    # 訓練回数
cfg.SOLVER.CHECKPOINT_PERIOD = 500
cfg.TEST.EVAL_PERIOD = 100
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # ROIヘッドのバッチサイズ
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # クラス数（背景を除く）
cfg.OUTPUT_DIR = dataset_dir + "/outputs"
cfg.MODEL.MASK_ON = False

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
