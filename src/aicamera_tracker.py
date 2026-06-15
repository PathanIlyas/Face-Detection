# src/aicamera_tracker.py

import cv2
import torch
import argparse
import time
from pathlib import Path
import numpy as np

# Project-specific imports
from . import config # For configurations like model paths, classes, colors
from .utils import image_processing, visualization
from .detector.yolo_detector import YOLODetector
from .tracker.deepsort_tracker import DeepSORT

# For consistent Track IDs if multiple runs are in the same session and Track class is re-imported.
# This should ideally be handled if TrackerCore itself is re-instantiated per run.
from .tracker.core.track import Track 

def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the tracker."""
    parser = argparse.ArgumentParser(description="AICamera: Real-time Object Detection & Tracking")
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to input video file. If None, tries to use webcam."
    )
    parser.add_argument(
        "--webcam_id", type=int, default=0,
        help="Webcam ID to use if --input is not specified."
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs",
        help="Directory to save the output video."
    )
    parser.add_argument(
        "--output_filename", type=str, default=None,
        help="Name of the output video file. If None, generated from input name or timestamp."
    )
    parser.add_argument(
        "--show_display", action="store_true",
        help="Show the processed video frames in a window."
    )
    parser.add_argument(
        "--no_save", action="store_true",
        help="Do not save the output video."
    )
    parser.add_argument(
        "--yolo_engine", type=str, default=str(config.YOLO_ENGINE_PATH),
        help="Path to the YOLO TensorRT engine file."
    )
    parser.add_argument(
        "--reid_engine", type=str, default=str(config.REID_ENGINE_PATH),
        help="Path to the ReID (DeepSORT) TensorRT engine file."
    )
    parser.add_argument(
        "--conf_thresh", type=float, default=config.YOLO_CONF_THRESHOLD,
        help="Confidence threshold for YOLO detections."
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Device to use for inference (e.g., 'cuda:0', 'cpu'). TensorRT typically requires CUDA."
    )
    parser.add_argument(
        "--tripwire", type=str, default=None,
        help="Tripwire line coordinates as x1,y1,x2,y2. If None, defaults to middle horizontal line."
    )
    
    args = parser.parse_args()
    if args.device == "cpu" and (Path(args.yolo_engine).exists() or Path(args.reid_engine).exists()):
        print("Warning: TensorRT engines are specified, but device is 'cpu'. TRTEngine will not run on CPU.")
    return args

def main():
    """Main function to run the object detection and tracking pipeline."""
    args = parse_arguments()
    Track.reset_id_counter() # Reset track IDs for a fresh run

    # --- Setup Device ---
    if args.device.lower() == 'cpu':
        device = torch.device('cpu')
        print("Running on CPU. Note: TensorRT specific features will not be used effectively.")
    elif torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        print(f"Warning: CUDA device '{args.device}' not available. Falling back to CPU.")
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # --- Initialize Detector ---
    print("Initializing YOLOv8 Detector...")
    try:
        yolo_detector = YOLODetector(
            engine_path=args.yolo_engine,
            conf_threshold=args.conf_thresh,
            # nms_threshold is often handled by the engine, but can be passed if needed
            device=device # YOLODetector's TRTEngine will use this
        )
    except Exception as e:
        print(f"Error initializing YOLO Detector: {e}")
        print("Please ensure YOLO engine path is correct and TensorRT is set up.")
        return

    # --- Initialize Tracker ---
    print("Initializing DeepSORT Tracker...")
    try:
        # DeepSORT tracker uses its ReIDModel which initializes its own TRTEngine
        deepsort_tracker = DeepSORT(
            reid_model_path=args.reid_engine
            # Other DeepSORT parameters are taken from config.py by default
        )
    except Exception as e:
        print(f"Error initializing DeepSORT Tracker: {e}")
        print("Please ensure ReID engine path is correct and TensorRT is set up.")
        return

    # --- Setup Video Input ---
    if args.input:
        if not Path(args.input).exists():
            print(f"Error: Input video file not found: {args.input}")
            return
        video_source_name = Path(args.input).stem
        cap = cv2.VideoCapture(args.input)
        source_type = "video"
    else:
        print(f"No input video specified, attempting to use webcam ID: {args.webcam_id}")
        cap = cv2.VideoCapture(args.webcam_id)
        video_source_name = f"webcam_{args.webcam_id}"
        source_type = "webcam"

    if not cap.isOpened():
        print(f"Error: Could not open video source ({video_source_name}).")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps == 0: # Webcam might return 0
        source_fps = config.DEFAULT_OUTPUT_FPS 
    print(f"Opened {source_type}: {video_source_name} ({frame_width}x{frame_height} @ {source_fps:.2f} FPS)")

    # --- Initialize Tripwire & Counts ---
    tripwire_coords = None
    if args.tripwire:
        try:
            coords = [int(c.strip()) for c in args.tripwire.split(",")]
            if len(coords) == 4:
                tripwire_coords = ((coords[0], coords[1]), (coords[2], coords[3]))
                print(f"Using custom tripwire line: {tripwire_coords}")
            else:
                print("Warning: --tripwire must contain exactly 4 comma-separated integers (x1,y1,x2,y2). Using default.")
        except Exception as e:
            print(f"Warning: Failed to parse --tripwire ({e}). Using default.")

    if tripwire_coords is None:
        # Default to a horizontal line across the middle of the frame
        mid_y = frame_height // 2
        tripwire_coords = ((0, mid_y), (frame_width, mid_y))
        print(f"Using default middle horizontal tripwire: {tripwire_coords}")

    track_sides = {}
    in_count = 0
    out_count = 0
    from collections import defaultdict
    class_counts = defaultdict(lambda: {"in": 0, "out": 0})

    # --- Setup Video Output (if saving) ---
    video_writer = None
    if not args.no_save:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if args.output_filename:
            output_video_name = args.output_filename
            if not output_video_name.lower().endswith(('.mp4', '.avi')):
                output_video_name += ".mp4" # Default to mp4
        else:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_video_name = f"{video_source_name}_tracked_{timestamp}.mp4"
            
        output_video_path = output_dir / output_video_name
        
        # Use MP4V for .mp4, or XVID for .avi for broader compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') if output_video_name.lower().endswith('.mp4') else cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(str(output_video_path), fourcc, source_fps, (frame_width, frame_height))
        if video_writer.isOpened():
            print(f"Output video will be saved to: {output_video_path}")
        else:
            print(f"Error: Could not open video writer for {output_video_path}. Video will not be saved.")
            video_writer = None # Ensure it's None if opening failed

    # --- Main Processing Loop ---
    frame_idx = 0
    total_time_spent = 0
    display_fps = 0.0

    try:
        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret:
                print("End of video stream or error reading frame.")
                break

            start_time_frame = time.time()
            
            # 1. Detection
            try:
                # yolo_detector.detect returns: bboxes_xyxy, scores, class_ids, filtered_indices
                det_bboxes, det_scores, det_class_ids, _ = yolo_detector.detect(frame_bgr)
            except Exception as e:
                print(f"Error during detection on frame {frame_idx}: {e}")
                # Optionally, skip tracking for this frame or break
                if args.show_display: cv2.imshow("AICamera Tracking", frame_bgr) # Show original frame
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                continue


            # 2. Tracking
            try:
                # deepsort_tracker.update expects: yolo_bboxes_xyxy, yolo_confidences, yolo_class_ids, original_frame_bgr
                # It returns: List of (x1, y1, x2, y2, track_id, class_name, track_confidence)
                tracked_objects = deepsort_tracker.update(
                    det_bboxes, det_scores, det_class_ids, frame_bgr.copy() # Pass a copy if frame_bgr is modified by vis
                )
            except Exception as e:
                print(f"Error during tracking on frame {frame_idx}: {e}")
                tracked_objects = [] # Continue with no tracks for this frame

            # 2b. Line Crossing Check
            for obj in tracked_objects:
                x1, y1, x2, y2, track_id, class_name = obj[:6]
                
                # Calculate current centroid
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                
                # Calculate current side relative to the tripwire
                # Line defined by A = tripwire_coords[0], B = tripwire_coords[1]
                # Cross product: (B_x - A_x) * (P_y - A_y) - (B_y - A_y) * (P_x - A_x)
                A, B = tripwire_coords
                val = (B[0] - A[0]) * (cy - A[1]) - (B[1] - A[1]) * (cx - A[0])
                
                if val > 0:
                    curr_side = 1
                elif val < 0:
                    curr_side = -1
                else:
                    curr_side = 0
                
                # Check for side transition
                if track_id in track_sides:
                    prev_side = track_sides[track_id]
                    if prev_side != 0 and curr_side != 0 and prev_side != curr_side:
                        # Crossed!
                        if prev_side == -1 and curr_side == 1:
                            in_count += 1
                            class_counts[class_name]["in"] += 1
                            print(f"[TRIPWIRE] Track {track_id} ({class_name}) crossed: IN (Total IN: {in_count})")
                        elif prev_side == 1 and curr_side == -1:
                            out_count += 1
                            class_counts[class_name]["out"] += 1
                            print(f"[TRIPWIRE] Track {track_id} ({class_name}) crossed: OUT (Total OUT: {out_count})")
                
                # Update current side
                if curr_side != 0:
                    track_sides[track_id] = curr_side
            
            # Clean up track_sides for deleted track IDs to prevent memory growth
            core_track_ids = {t.track_id for t in deepsort_tracker.tracker_core.tracks}
            track_sides = {tid: side for tid, side in track_sides.items() if tid in core_track_ids}

            end_time_frame = time.time()
            frame_processing_time = end_time_frame - start_time_frame
            total_time_spent += frame_processing_time
            if frame_idx > 0 and total_time_spent > 0: # Avoid division by zero, smooth FPS
                 display_fps = (frame_idx + 1) / total_time_spent
            elif frame_processing_time > 0:
                 display_fps = 1.0 / frame_processing_time


            # 3. Visualization
            vis_frame = frame_bgr.copy() # Draw on a copy

            # Draw raw detections (optional, for debugging)
            # vis_frame = visualization.draw_detections(vis_frame, det_bboxes, det_scores, det_class_ids, config.CLASSES)

            # Draw tracks
            vis_frame = visualization.draw_tracks(vis_frame, tracked_objects)
            
            # Draw tripwire and counts dashboard
            vis_frame = visualization.draw_tripwire_and_counts(
                vis_frame, tripwire_coords, in_count, out_count, class_counts
            )
            
            # Draw FPS and other info
            info_lines = [
                f"AICamera: YOLOv8 + DeepSORT",
                f"Input: {video_source_name}",
                f"FPS: {display_fps:.2f}"
            ]
            vis_frame = visualization.draw_info_panel(vis_frame, info_lines)


            # 4. Display and Save
            if args.show_display:
                cv2.imshow("AICamera Tracking", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exiting...")
                    break
            
            if video_writer and video_writer.isOpened():
                video_writer.write(vis_frame)

            frame_idx += 1
            if frame_idx % 100 == 0: # Print progress every 100 frames
                print(f"Processed {frame_idx} frames. Current FPS: {display_fps:.2f}")

    except KeyboardInterrupt:
        print("Processing interrupted by user.")
    finally:
        # --- Cleanup ---
        if cap:
            cap.release()
            print("Video source released.")
        if video_writer and video_writer.isOpened():
            video_writer.release()
            print("Output video writer released.")
        if args.show_display:
            cv2.destroyAllWindows()
            print("Display windows closed.")
        
        avg_fps = (frame_idx / total_time_spent) if total_time_spent > 0 else 0
        print(f"\n--- Processing Summary ---")
        print(f"Total frames processed: {frame_idx}")
        print(f"Total time: {total_time_spent:.2f} seconds")
        print(f"Average FPS: {avg_fps:.2f}")
        print("AICamera finished.")

if __name__ == "__main__":
    # This structure ensures that if this script is run directly,
    # the main() function is called.
    # For `python -m src.aicamera_tracker`, Python handles the module execution.
    main()