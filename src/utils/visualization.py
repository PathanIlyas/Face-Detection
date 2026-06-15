# src/utils/visualization.py

import cv2
import numpy as np
from . import image_processing # To import config if needed for colors/fonts
from .. import config # Import config directly from src
from typing import List

def draw_detections(
    frame: np.ndarray,
    bboxes_xyxy: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    class_names: tuple
) -> np.ndarray:
    """
    Draws raw detection bounding boxes, scores, and class names on the frame.
    Mainly for debugging purposes.

    Args:
        frame (np.ndarray): The image frame to draw on.
        bboxes_xyxy (np.ndarray): Bounding boxes in (x1, y1, x2, y2) format.
        scores (np.ndarray): Detection confidence scores.
        class_ids (np.ndarray): Class IDs for each detection.
        class_names (tuple): Tuple of all possible class names.

    Returns:
        np.ndarray: Frame with detections drawn.
    """
    for i in range(len(bboxes_xyxy)):
        x1, y1, x2, y2 = map(int, bboxes_xyxy[i])
        score = scores[i]
        class_id = int(class_ids[i])
        
        if class_id < 0 or class_id >= len(class_names):
            label_name = "Unknown"
            color = (128, 128, 128) # Gray for unknown
        else:
            label_name = class_names[class_id]
            color = config.get_class_color(label_name)

        label = f"{label_name}: {score:.2f}"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Calculate text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            label, config.FONT, config.FONT_SCALE_ID, config.FONT_THICKNESS
        )
        # Put text background
        cv2.rectangle(
            frame,
            (x1, y1 - text_height - baseline),
            (x1 + text_width, y1),
            color,
            -1, # Filled
        )
        # Put text
        cv2.putText(
            frame,
            label,
            (x1, y1 - baseline // 2), # Adjust for better vertical alignment
            config.FONT,
            config.FONT_SCALE_ID,
            (255, 255, 255), # White text
            config.FONT_THICKNESS,
            cv2.LINE_AA,
        )
    return frame


def draw_tracks(
    frame: np.ndarray,
    tracked_objects: list # List of tuples: (x1, y1, x2, y2, track_id, class_name, Optional[score])
) -> np.ndarray:
    """
    Draws tracked bounding boxes with track IDs and class names on the frame.

    Args:
        frame (np.ndarray): The image frame to draw on.
        tracked_objects (list): A list of tuples, where each tuple contains
                                (x1, y1, x2, y2, track_id, class_name, Optional[score]).

    Returns:
        np.ndarray: Frame with tracks drawn.
    """
    for obj_data in tracked_objects:
        x1, y1, x2, y2 = map(int, obj_data[:4])
        track_id = obj_data[4]
        class_name = obj_data[5]
        
        color = config.get_track_color(class_name) # Use class-specific color for the track

        label = f"ID:{track_id} {class_name}"
        if len(obj_data) > 6: # If score is provided
            score = obj_data[6]
            label += f" {score:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Calculate text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            label, config.FONT, config.FONT_SCALE_ID, config.FONT_THICKNESS
        )
        # Put text background
        cv2.rectangle(
            frame,
            (x1, y1 - text_height - baseline - 2), # Add a small margin
            (x1 + text_width, y1),
            color,
            -1, # Filled
        )
        # Put text
        cv2.putText(
            frame,
            label,
            (x1, y1 - baseline // 2 - 1), # Adjust for better vertical alignment
            config.FONT,
            config.FONT_SCALE_ID,
            (255, 255, 255), # White text
            config.FONT_THICKNESS,
            cv2.LINE_AA,
        )
    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """
    Draws the current FPS on the top-left corner of the frame.

    Args:
        frame (np.ndarray): The image frame to draw on.
        fps (float): The calculated frames per second.

    Returns:
        np.ndarray: Frame with FPS drawn.
    """
    fps_text = f"FPS: {fps:.2f}"
    
    # Calculate text size for background rectangle
    (text_width, text_height), baseline = cv2.getTextSize(
        fps_text, config.FONT, config.FONT_SCALE_INFO, config.FONT_THICKNESS
    )
    
    # Position for the text (top-left corner)
    text_x = 10
    text_y = text_height + 10 + baseline // 2

    # Put text background (optional, for better visibility)
    cv2.rectangle(
        frame,
        (text_x - 5, text_y - text_height - baseline - 5),
        (text_x + text_width + 5, text_y + 5),
        (50, 50, 50), # Dark gray background
        -1,
    )

    cv2.putText(
        frame,
        fps_text,
        (text_x, text_y - baseline // 2),
        config.FONT,
        config.FONT_SCALE_INFO,
        (255, 255, 255), # White text
        config.FONT_THICKNESS,
        cv2.LINE_AA,
    )
    return frame

def draw_info_panel(frame: np.ndarray, info_lines: List[str]) -> np.ndarray:
    """
    Draws multiple lines of informational text on the frame, typically at the top.
    Args:
        frame (np.ndarray): The image frame to draw on.
        info_lines (List[str]): A list of strings, each to be drawn on a new line.
    Returns:
        np.ndarray: Frame with info panel drawn.
    """
    start_x = 10
    start_y = 30 # Initial y position
    line_height_offset = 0

    max_text_width = 0

    # First pass to determine max width for background
    for line_index, text_line in enumerate(info_lines):
        (text_width, text_height), baseline = cv2.getTextSize(
            text_line, config.FONT, config.FONT_SCALE_INFO, config.FONT_THICKNESS
        )
        if text_width > max_text_width:
            max_text_width = text_width
        if line_index == 0: # Use first line's height for consistent spacing
            line_height_offset = text_height + baseline + 10 # 10px spacing

    # Draw background for the panel (optional)
    if info_lines:
        panel_height = len(info_lines) * line_height_offset
        cv2.rectangle(
            frame,
            (start_x - 5, start_y - line_height_offset + 15), # Adjust to cover first line properly
            (start_x + max_text_width + 5, start_y + panel_height - line_height_offset + 15),
            (50, 50, 50, 180), # Dark semi-transparent background (if alpha is supported by drawing context)
            -1,
        )


    current_y = start_y
    for text_line in info_lines:
        (text_width, text_height), baseline = cv2.getTextSize(
            text_line, config.FONT, config.FONT_SCALE_INFO, config.FONT_THICKNESS
        )
        
        # text_y_position = current_y + text_height # Baseline is below text
        text_y_position = current_y + baseline + (text_height // 2) # Center text a bit better

        cv2.putText(
            frame,
            text_line,
            (start_x, text_y_position),
            config.FONT,
            config.FONT_SCALE_INFO,
            (255, 255, 255), # White text
            config.FONT_THICKNESS,
            cv2.LINE_AA,
        )
        current_y += line_height_offset # Move to next line position

    return frame

def draw_semi_transparent_rect(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: tuple, alpha: float = 0.6) -> None:
    """Draws a semi-transparent rectangle on the frame."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

def draw_tripwire_and_counts(
    frame: np.ndarray,
    line_coords: tuple, # ((lx1, ly1), (lx2, ly2))
    in_count: int,
    out_count: int,
    class_counts: dict # e.g. {"person": {"in": 2, "out": 1}}
) -> np.ndarray:
    """Draws a glowing tripwire line and a premium glassmorphic dashboard panel showing line crossing statistics."""
    (lx1, ly1), (lx2, ly2) = line_coords
    
    # 1. Draw glowing tripwire line (Yellow glow + White core)
    overlay = frame.copy()
    cv2.line(overlay, (lx1, ly1), (lx2, ly2), (0, 255, 255), 6, cv2.LINE_AA) # Neon yellow outer glow
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    cv2.line(frame, (lx1, ly1), (lx2, ly2), (255, 255, 255), 2, cv2.LINE_AA) # White inner core
    
    # Draw arrow indicators for IN / OUT (perpendicular to line direction)
    dx = lx2 - lx1
    dy = ly2 - ly1
    length = np.sqrt(dx**2 + dy**2)
    if length > 0:
        nx = -dy / length
        ny = dx / length
        
        # Draw arrows at the center of the tripwire
        cx = int((lx1 + lx2) / 2)
        cy = int((ly1 + ly2) / 2)
        
        # IN arrow (positive normal direction)
        arrow_len = 25
        in_arrow_end = (int(cx + nx * arrow_len), int(cy + ny * arrow_len))
        cv2.arrowedLine(frame, (cx, cy), in_arrow_end, (0, 255, 0), 2, tipLength=0.3)
        cv2.putText(frame, "IN", (in_arrow_end[0] + 5, in_arrow_end[1]), config.FONT, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        
        # OUT arrow (negative normal direction)
        out_arrow_end = (int(cx - nx * arrow_len), int(cy - ny * arrow_len))
        cv2.arrowedLine(frame, (cx, cy), out_arrow_end, (0, 0, 255), 2, tipLength=0.3)
        cv2.putText(frame, "OUT", (out_arrow_end[0] + 5, out_arrow_end[1]), config.FONT, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        
    # 2. Draw counts panel
    h, w = frame.shape[:2]
    panel_w = 260
    
    # Count how many classes have non-zero crossings to size the panel dynamically
    active_classes = {cls: cnts for cls, cnts in class_counts.items() if cnts["in"] > 0 or cnts["out"] > 0}
    panel_h = 95 + len(active_classes) * 30
    
    margin = 20
    px1 = w - panel_w - margin
    py1 = margin
    px2 = w - margin
    py2 = py1 + panel_h
    
    # Semi-transparent dark background (glassmorphism look)
    draw_semi_transparent_rect(frame, px1, py1, px2, py2, (30, 30, 30), alpha=0.75)
    
    # Accent top border (Cyan-Yellow style)
    cv2.rectangle(frame, (px1, py1), (px2, py1 + 4), (0, 215, 255), -1)
    # Thin gray border around panel
    cv2.rectangle(frame, (px1, py1), (px2, py2), (100, 100, 100), 1)
    
    # Draw Title
    cv2.putText(frame, "LINE CROSSING COUNTER", (px1 + 15, py1 + 25), config.FONT, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(frame, (px1 + 15, py1 + 35), (px2 - 15, py1 + 35), (80, 80, 80), 1)
    
    # Draw Total Counts
    cv2.putText(frame, f"IN: {in_count}", (px1 + 20, py1 + 65), config.FONT, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"OUT: {out_count}", (px1 + 140, py1 + 65), config.FONT, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Draw breakdown by class
    curr_y = py1 + 95
    for cls_name, counts in active_classes.items():
        # Draw line for class separator
        cv2.line(frame, (px1 + 15, curr_y - 15), (px2 - 15, curr_y - 15), (60, 60, 60), 1)
        # Draw class details
        cv2.putText(frame, f"{cls_name.capitalize()}:", (px1 + 20, curr_y), config.FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"In: {counts['in']}", (px1 + 110, curr_y), config.FONT, 0.45, (0, 200, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Out: {counts['out']}", (px1 + 180, curr_y), config.FONT, 0.45, (0, 0, 200), 1, cv2.LINE_AA)
        curr_y += 30
            
    return frame