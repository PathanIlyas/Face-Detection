# scripts/download_models.py
import os
import urllib.request
from pathlib import Path

# Define URLs
YOLO_ONNX_URL = "https://raw.githubusercontent.com/nabang1010/YOLO_Object_Tracking_TensorRT/main/models/onnx/yolov8n.onnx"
DEEPSORT_ONNX_URL = "https://raw.githubusercontent.com/nabang1010/YOLO_Object_Tracking_TensorRT/main/models/onnx/deepsort.onnx"

# Define Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DETECTION_DIR = PROJECT_ROOT / "models" / "detection"
REID_DIR = PROJECT_ROOT / "models" / "reid"

YOLO_ONNX_PATH = DETECTION_DIR / "yolov8n.onnx"
DEEPSORT_ONNX_PATH = REID_DIR / "deepsort_reid.onnx"

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    try:
        # Custom opener to handle User-Agent if needed
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        
        urllib.request.urlretrieve(url, dest_path)
        print(f"Successfully downloaded to {dest_path}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        raise

def main():
    # Create directories if they do not exist
    DETECTION_DIR.mkdir(parents=True, exist_ok=True)
    REID_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download YOLOv8 ONNX
    if not YOLO_ONNX_PATH.exists():
        download_file(YOLO_ONNX_URL, YOLO_ONNX_PATH)
    else:
        print(f"YOLOv8 ONNX file already exists at {YOLO_ONNX_PATH}")
        
    # Download DeepSORT ReID ONNX
    if not DEEPSORT_ONNX_PATH.exists():
        download_file(DEEPSORT_ONNX_URL, DEEPSORT_ONNX_PATH)
    else:
        print(f"DeepSORT ReID ONNX file already exists at {DEEPSORT_ONNX_PATH}")

if __name__ == "__main__":
    main()
