# scripts/export_trt_engines.py
#
# Converts ONNX models to TensorRT 11 engine files using the Python API.
# Compatible with TensorRT 11.x — no trtexec binary required.

import tensorrt as trt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DETECTION_DIR = PROJECT_ROOT / "models" / "detection"
REID_DIR      = PROJECT_ROOT / "models" / "reid"

YOLO_ONNX_PATH    = DETECTION_DIR / "yolov8n.onnx"
YOLO_ENGINE_PATH  = DETECTION_DIR / "yolov8n.engine"

DEEPSORT_ONNX_PATH   = REID_DIR / "deepsort_reid.onnx"
DEEPSORT_ENGINE_PATH = REID_DIR / "deepsort_reid.engine"


def build_engine(onnx_path: Path, engine_path: Path,
                 dynamic_shapes: dict = None) -> bool:
    """
    Build a TensorRT 11 engine from an ONNX file.
    
    TRT 11 changes vs TRT 8/9:
      - EXPLICIT_BATCH is always on; pass flags=0 to create_network().
      - BuilderFlag.FP16 removed; precision is now set via STRONGLY_TYPED
        networks or per-layer.  We use TF32 (the default) which gives a good
        speed/accuracy trade-off on Ampere GPUs (RTX 3050).
    """
    print("=" * 60)
    print(f"  Building engine: {onnx_path.name}")
    print(f"  Output:          {engine_path}")

    if engine_path.exists():
        size_mb = engine_path.stat().st_size / 1_000_000
        print(f"  Engine already exists ({size_mb:.1f} MB) -- skipping rebuild.")
        return True

    logger  = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config  = builder.create_builder_config()

    # TRT 11: create_network() with flags=0 — EXPLICIT_BATCH is implicit
    network = builder.create_network(0)
    parser  = trt.OnnxParser(network, logger)

    if not onnx_path.exists():
        print(f"  [ERROR] ONNX not found: {onnx_path}")
        return False

    with open(onnx_path, "rb") as f:
        raw = f.read()

    if not parser.parse(raw):
        print("  [ERROR] ONNX parse errors:")
        for i in range(parser.num_errors):
            print(f"    {parser.get_error(i)}")
        return False

    # Print detected input names / shapes
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"  Input [{i}]: name='{inp.name}'  shape={list(inp.shape)}")

    # Dynamic shape optimisation profile
    if dynamic_shapes:
        profile = builder.create_optimization_profile()
        for name, (min_s, opt_s, max_s) in dynamic_shapes.items():
            profile.set_shape(name, min_s, opt_s, max_s)
            print(f"  Profile '{name}': min={min_s} opt={opt_s} max={max_s}")
        config.add_optimization_profile(profile)
    else:
        # Auto-generate a profile for any dynamic (-1) dims
        for i in range(network.num_inputs):
            inp   = network.get_input(i)
            shape = list(inp.shape)
            if any(d == -1 for d in shape):
                fixed = tuple(max(1, d) for d in shape)
                profile = builder.create_optimization_profile()
                profile.set_shape(inp.name, fixed, fixed, fixed)
                config.add_optimization_profile(profile)
                print(f"  Auto profile '{inp.name}': {fixed}")

    print("  Building … (may take 5-15 min on first run)")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("  [ERROR] build_serialized_network returned None")
        return False

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)

    size_mb = engine_path.stat().st_size / 1_000_000
    print(f"  Saved => {engine_path}  ({size_mb:.1f} MB)")
    return True


def main():
    # YOLOv8n — static-batch input
    yolo_ok = build_engine(
        YOLO_ONNX_PATH, YOLO_ENGINE_PATH,
        dynamic_shapes={"images": ((1, 3, 640, 640),
                                   (1, 3, 640, 640),
                                   (1, 3, 640, 640))},
    )

    # DeepSORT ReID — dynamic batch (1–8)
    reid_ok = build_engine(
        DEEPSORT_ONNX_PATH, DEEPSORT_ENGINE_PATH,
        dynamic_shapes={"input": ((1, 3, 128, 64),
                                  (1, 3, 128, 64),
                                  (8, 3, 128, 64))},
    )

    print()
    print("=" * 60)
    print(f"  YOLOv8n engine : {'OK ✓' if yolo_ok else 'FAILED ✗'}")
    print(f"  DeepSORT ReID  : {'OK ✓' if reid_ok else 'FAILED ✗'}")
    if yolo_ok and reid_ok:
        print()
        print("All engines built. Run the tracker with:")
        print("  venv\\Scripts\\python.exe -m src.aicamera_tracker --show_display")


if __name__ == "__main__":
    main()
