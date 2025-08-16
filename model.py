import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os
import gdown

# Path to the weights file
WEIGHTS_PATH = "best.pth"

def download_weights(file_id, output_path):
    if not os.path.exists(output_path):
        print("Downloading model weights...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
        print("Download complete.")
    else:
        print("Model weights already exist.")

def load_model(weights_path=WEIGHTS_PATH, config_path=None):
    # Google Drive file ID for your weights
    google_drive_file_id = "1jE-h5dHHYuyxBYBgseiQbg5es4nnRb5s"
    
    # Ensure weights exist
    download_weights(google_drive_file_id, weights_path)
    
    cfg = get_cfg()
    if config_path:
        cfg.merge_from_file(config_path)
    else:
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # ["Helmet", "No Helmet"]
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    
    # Ensure CPU works if CUDA is not available
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    predictor = DefaultPredictor(cfg)
    return predictor

def predict(image, predictor):
    return predictor(image)
