# src/detector/yolo_detector.py
#
# YOLOv8 detector using a TensorRT engine whose output is the raw
# [1, 84, 8400] tensor (no built-in NMS plugin).
# Post-processing (decode + NMS) is done in Python with torchvision.

import torch
import torchvision
import numpy as np
from typing import Tuple

from ..trt_utils.trt_engine import TRTEngine
from ..utils import image_processing
from .. import config


class YOLODetector:
    """
    YOLOv8 Detector using a TensorRT engine.

    Engine output format (Ultralytics raw export, no NMS plugin):
        output0: [1, 84, 8400]
            - Dim 1: 4 box params (cx, cy, w, h) + 80 class scores
            - Dim 2: 8400 anchor proposals (grid cells)
    """

    def __init__(
        self,
        engine_path: str = str(config.YOLO_ENGINE_PATH),
        input_shape: Tuple[int, int] = config.YOLO_INPUT_SHAPE,   # (H, W)
        conf_threshold: float = config.YOLO_CONF_THRESHOLD,
        nms_threshold: float = config.YOLO_NMS_THRESHOLD,
        device: torch.device = None,
    ):
        self.input_shape = input_shape
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.device = device or torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        self.trt_engine = TRTEngine(engine_path, device=self.device)
        self.input_name = self.trt_engine.get_input_details()[0].name

        out_names = [i.name for i in self.trt_engine.get_output_details()]
        print(f"YOLODetector initialised | engine: {engine_path}")
        print(f"  Input  : '{self.input_name}' shape={input_shape}")
        print(f"  Outputs: {out_names}")

    # ------------------------------------------------------------------
    def detect(
        self, frame_bgr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect objects in a BGR frame.

        Returns
        -------
        bboxes_xyxy : (N, 4) float32  – original-image coordinates
        scores      : (N,)   float32
        class_ids   : (N,)   int32
        kept_indices: (N,)   int      – indices within filtered candidates
        """
        original_shape = frame_bgr.shape[:2]  # H, W

        # ── Pre-process ───────────────────────────────────────────────
        img_tensor, ratios, (pad_w, pad_h) = image_processing.preprocess_yolo_input(
            frame_bgr, target_shape=self.input_shape
        )
        input_t = torch.from_numpy(img_tensor).to(self.device)

        # ── Inference ─────────────────────────────────────────────────
        outputs = self.trt_engine.infer({self.input_name: input_t})

        # output0: [1, 84, 8400]  (cx, cy, w, h, cls0…cls79)
        raw = list(outputs.values())[0]          # [1, 84, 8400]
        raw = raw.squeeze(0).T                   # [8400, 84]

        # ── Decode ────────────────────────────────────────────────────
        boxes_cxcywh = raw[:, :4]                # [8400, 4]
        cls_scores   = raw[:, 4:]                # [8400, 80]

        scores, class_ids = cls_scores.max(dim=1)

        # Confidence filter
        mask = scores >= self.conf_threshold
        if mask.sum() == 0:
            empty = np.empty((0, 4), dtype=np.float32)
            return empty, np.empty(0, np.float32), np.empty(0, np.int32), np.empty(0, int)

        boxes_cxcywh = boxes_cxcywh[mask]
        scores       = scores[mask]
        class_ids    = class_ids[mask]

        # cx,cy,w,h  ->  x1,y1,x2,y2  (letterboxed space)
        x1 = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
        y1 = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
        x2 = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
        y2 = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)  # [N, 4]

        # ── NMS (per-class) ───────────────────────────────────────────
        keep = torchvision.ops.nms(boxes_xyxy, scores, self.nms_threshold)
        boxes_xyxy = boxes_xyxy[keep].cpu().numpy()
        scores     = scores[keep].cpu().numpy()
        class_ids  = class_ids[keep].cpu().numpy().astype(np.int32)

        if boxes_xyxy.shape[0] == 0:
            empty = np.empty((0, 4), dtype=np.float32)
            return empty, np.empty(0, np.float32), np.empty(0, np.int32), np.empty(0, int)

        # ── Scale back to original image space ────────────────────────
        bboxes_orig = image_processing.scale_bboxes(
            boxes_xyxy,
            original_shape=original_shape,
            letterbox_shape=self.input_shape,
            ratio=ratios,
            padding=(pad_w, pad_h),
        )

        return bboxes_orig, scores, class_ids, np.arange(len(scores))


# ── Quick smoke-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import traceback

    print("--- Testing YOLODetector ---")
    if not config.YOLO_ENGINE_PATH.exists():
        print(f"Skipped: engine not found at {config.YOLO_ENGINE_PATH}")
    else:
        try:
            detector = YOLODetector()
            dummy = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            bboxes, scores, class_ids, _ = detector.detect(dummy)
            print(f"Detections: {len(bboxes)}")
            for i in range(min(5, len(bboxes))):
                print(
                    f"  [{i}] {config.CLASSES[class_ids[i]]:12s} "
                    f"score={scores[i]:.2f}  box={bboxes[i]}"
                )
            print("--- PASSED ---")
        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()