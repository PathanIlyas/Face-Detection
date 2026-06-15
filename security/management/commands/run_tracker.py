import os
import cv2
import time
import logging
import torch
from pathlib import Path
from django.core.management.base import BaseCommand
from django.utils import timezone
from django.conf import settings

# Imports from existing CV codebase
from src.detector.yolo_detector import YOLODetector
from src.tracker.deepsort_tracker import DeepSORT
from src import config
from src.utils import visualization

# Model and Alert imports
from security.models import Camera, DetectionLog
from security.utils.alerts import send_alert_email_async

# Setup loggers matching Django settings config
logger = logging.getLogger('security_app')
det_logger = logging.getLogger('detection')
err_logger = logging.getLogger('error')

class Command(BaseCommand):
    help = "Runs the AI Camera surveillance system, analyzing frames and logging person detections to PostgreSQL."

    def add_arguments(self, parser):
        parser.add_argument(
            "--camera_name", type=str, default=None,
            help="Name of the camera. If None, loaded from env or defaults to 'Webcam 0'"
        )
        parser.add_argument(
            "--input", type=str, default=None,
            help="Path to input video file or webcam index. If None, loaded from env."
        )
        parser.add_argument(
            "--show_display", action="store_true",
            help="Show the processed video stream in a GUI window."
        )
        parser.add_argument(
            "--yolo_engine", type=str, default=str(config.YOLO_ENGINE_PATH),
            help="Path to the YOLO TensorRT engine file."
        )
        parser.add_argument(
            "--reid_engine", type=str, default=str(config.REID_ENGINE_PATH),
            help="Path to the ReID (DeepSORT) TensorRT engine file."
        )

    def handle(self, *args, **options):
        # 1. Configuration parsing
        camera_name = options["camera_name"] or os.getenv("CAMERA_NAME", "Webcam 0")
        camera_source = options["input"] or os.getenv("CAMERA_SOURCE", "0")
        
        # Check if source is integer (for webcam index)
        try:
            camera_source = int(camera_source)
        except ValueError:
            pass # Keep it as string path (for video files)
            
        show_display = options["show_display"]
        yolo_engine = options["yolo_engine"]
        reid_engine = options["reid_engine"]
        
        logger.info(f"Starting Security Tracker. Camera: {camera_name}, Source: {camera_source}")
        
        # 2. Database registration with retry
        def get_or_create_camera():
            cam, _ = Camera.objects.get_or_create(
                name=camera_name,
                defaults={'stream_url': str(camera_source), 'status': 'offline'}
            )
            cam.status = 'online'
            cam.last_seen_at = timezone.now()
            cam.save()
            return cam

        try:
            camera = self.retry_db_action(get_or_create_camera)
            logger.info(f"Camera '{camera_name}' registered successfully.")
        except Exception as e:
            err_logger.error(f"Failed to connect to database or register camera. Aborting tracker. Error: {e}")
            return

        # 3. Setup device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using inference device: {device}")

        # 4. Load AI Models
        logger.info("Loading YOLOv8 Detector Engine...")
        try:
            yolo_detector = YOLODetector(
                engine_path=yolo_engine,
                conf_threshold=float(os.getenv("YOLO_CONF_THRESHOLD", 0.3)),
                device=device
            )
        except Exception as e:
            err_logger.error(f"Error loading YOLOv8 engine: {e}")
            self.set_camera_offline(camera)
            return

        logger.info("Loading DeepSORT ReID Engine...")
        try:
            deepsort_tracker = DeepSORT(reid_model_path=reid_engine)
        except Exception as e:
            err_logger.error(f"Error loading DeepSORT ReID engine: {e}")
            self.set_camera_offline(camera)
            return

        # 5. Surveillance loop state variables
        seen_track_ids = set()
        last_db_ping = time.time()
        cap = None
        
        try:
            while True:
                # Reconnection handler: ensure camera opens successfully
                if cap is None or not cap.isOpened():
                    logger.info(f"Connecting to video source: {camera_source}")
                    cap = cv2.VideoCapture(camera_source)
                    if not cap.isOpened():
                        err_logger.error(f"Could not open source {camera_source}. Retrying in 5 seconds...")
                        self.set_camera_offline(camera)
                        time.sleep(5)
                        continue
                    
                    self.set_camera_online(camera)
                    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    logger.info(f"Camera connected successfully: {frame_width}x{frame_height}")

                ret, frame_bgr = cap.read()
                if not ret:
                    err_logger.error("Error reading frame from camera. Disconnected. Reconnecting in 5 seconds...")
                    self.set_camera_offline(camera)
                    cap.release()
                    time.sleep(5)
                    continue

                # Keep-alive heartbeat ping to PostgreSQL
                now_time = time.time()
                if now_time - last_db_ping > 15:
                    self.set_camera_online(camera)
                    last_db_ping = now_time

                # 1. Bounding Box Object Detection
                try:
                    det_bboxes, det_scores, det_class_ids, _ = yolo_detector.detect(frame_bgr)
                except Exception as e:
                    err_logger.error(f"YOLO detector run error: {e}")
                    continue

                # 2. Tracking ID Association
                try:
                    tracked_objects = deepsort_tracker.update(
                        det_bboxes, det_scores, det_class_ids, frame_bgr.copy()
                    )
                except Exception as e:
                    err_logger.error(f"DeepSORT tracker run error: {e}")
                    tracked_objects = []

                # 3. Log Security Alerts
                for obj in tracked_objects:
                    x1, y1, x2, y2, track_id, class_name, confidence = obj[:7]
                    
                    # Log and capture only for class 'person'
                    if class_name == 'person':
                        if track_id not in seen_track_ids:
                            seen_track_ids.add(track_id)
                            
                            # Console log & local file logs
                            logger.info(f"[SECURITY ALERT] Person detected! Track ID: {track_id}, Conf: {confidence:.2f}")
                            det_logger.info(f"Track={track_id}, Camera={camera.name}, Confidence={confidence:.2f}, Time={timezone.now()}")
                            
                            # Date folder naming layout
                            timestamp_str = timezone.now().strftime("%H%M%S")
                            date_path = timezone.now().strftime("%Y/%m/%d")
                            filename = f"track_{track_id}_{timestamp_str}.jpg"
                            relative_screenshot_dir = os.path.join('detections', date_path)
                            absolute_screenshot_dir = os.path.join(settings.MEDIA_ROOT, relative_screenshot_dir)
                            
                            absolute_screenshot_path = os.path.join(absolute_screenshot_dir, filename)
                            relative_screenshot_path = os.path.join(settings.MEDIA_URL, 'detections', date_path, filename).replace('\\', '/')
                            
                            # Save image to file system with retry loops
                            save_ok = self.retry_screenshot_save(frame_bgr, absolute_screenshot_path)
                            
                            if save_ok:
                                # Write to DB with retry loops
                                def create_log():
                                    return DetectionLog.objects.create(
                                        track_id=track_id,
                                        confidence=confidence,
                                        camera=camera,
                                        screenshot_path=relative_screenshot_path,
                                        detected_at=timezone.now()
                                    )
                                try:
                                    self.retry_db_action(create_log)
                                    logger.info(f"DB entry saved for Track ID: {track_id}")
                                except Exception as e:
                                    err_logger.error(f"Failed to write DetectionLog for Track {track_id} to database: {e}")
                                
                                # Send alert email in background thread
                                send_alert_email_async(
                                    track_id=track_id,
                                    camera_name=camera.name,
                                    screenshot_path=absolute_screenshot_path,
                                    confidence=confidence
                                )
                            else:
                                err_logger.error(f"Skipping database write for Track {track_id} due to screenshot write failures.")

                # 4. GUI Window Display (if flagged)
                if show_display:
                    vis_frame = frame_bgr.copy()
                    vis_frame = visualization.draw_tracks(vis_frame, tracked_objects)
                    
                    person_tracks = [t for t in tracked_objects if t[5] == 'person']
                    info_lines = [
                        f"STATUS: SECURE MONITORING ACTIVE",
                        f"CAMERA: {camera.name}",
                        f"ACTIVE PERSONS: {len(person_tracks)}",
                    ]
                    vis_frame = visualization.draw_info_panel(vis_frame, info_lines)
                    
                    cv2.imshow("Surveillance Feed Client", vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("Received exit command. Shutting down tracker...")
                        break

            if cap:
                cap.release()
            cv2.destroyAllWindows()
            self.set_camera_offline(camera)
            
        except KeyboardInterrupt:
            logger.info("Tracker stopped manually.")
            if cap:
                cap.release()
            cv2.destroyAllWindows()
            self.set_camera_offline(camera)
        except Exception as e:
            err_logger.error(f"Fatal tracker exception occurred: {e}")
            if cap:
                cap.release()
            cv2.destroyAllWindows()
            self.set_camera_offline(camera)

    def retry_db_action(self, action_fn, max_retries=3, delay=1.5):
        """Retries a database interaction if OperationalErrors or query failures occur."""
        for attempt in range(1, max_retries + 1):
            try:
                from django.db import transaction
                with transaction.atomic():
                    return action_fn()
            except Exception as e:
                err_logger.error(f"PostgreSQL Operational Error (attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    time.sleep(delay)
                else:
                    raise e

    def retry_screenshot_save(self, frame, path, max_retries=3, delay=0.5):
        """Saves screenshot image, retrying in case of temporary filesystem locked files or folder creation limits."""
        for attempt in range(1, max_retries + 1):
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                success = cv2.imwrite(path, frame)
                if success:
                    return True
                raise IOError(f"cv2.imwrite failed to save image at {path}")
            except Exception as e:
                err_logger.error(f"Filesystem screenshot write error (attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    time.sleep(delay)
        return False

    def set_camera_offline(self, camera):
        """Sets the Camera database entry status to Offline."""
        def action():
            camera.status = 'offline'
            camera.save()
        try:
            self.retry_db_action(action)
            logger.info(f"Camera '{camera.name}' status marked OFFLINE.")
        except Exception as e:
            err_logger.error(f"Failed to set camera '{camera.name}' status to offline: {e}")

    def set_camera_online(self, camera):
        """Sets the Camera database entry status to Online and updates heartbeat timestamps."""
        def action():
            camera.status = 'online'
            camera.last_seen_at = timezone.now()
            camera.save()
        try:
            self.retry_db_action(action)
        except Exception as e:
            err_logger.error(f"Failed to set camera '{camera.name}' status to online: {e}")
