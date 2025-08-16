# dataset_registration.py
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

def register_datasets():
    register_coco_instances("hardhat_train", {}, "Hard-Hat-Workers-14/train/_annotations.coco.json", "Hard-Hat-Workers-14/train")
    register_coco_instances("hardhat_val", {}, "Hard-Hat-Workers-14/valid/_annotations.coco.json", "Hard-Hat-Workers-14/valid")

    MetadataCatalog.get("hardhat_train").thing_classes = ["Helmet", "No Helmet"]
    MetadataCatalog.get("hardhat_val").thing_classes = ["Helmet", "No Helmet"]
