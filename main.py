"""Two-camera gaze-guided object detection interface for the SIH prototype."""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO
import tkinter as tk
from PIL import Image, ImageTk

from gaze_tracker import load_predictor, process_gaze_frame, release_resources

PROJECT_DIR = Path(__file__).resolve().parent
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


def process_frame(model, frame, direction):
    """Detect the highest-confidence object in the gaze-selected scene region."""
    if direction not in {"Left", "Center", "Right"}:
        return "No gaze detected; no target selected.", frame

    width = frame.shape[1]
    offset = width // 2 if direction == "Right" else 0
    if direction == "Left":
        roi = frame[:, :width // 2]
    elif direction == "Right":
        roi = frame[:, offset:]
    else:
        roi = frame

    best_box = None
    highest_confidence = 0.0
    for result in model(roi, verbose=False):
        for box in result.boxes:
            confidence = float(box.conf[0])
            if confidence > highest_confidence:
                highest_confidence = confidence
                best_box = box

    if best_box is None:
        return "No object detected in the selected region.", frame

    x1, y1, x2, y2 = best_box.xyxy[0].cpu().tolist()
    x1, x2 = int(x1 + offset), int(x2 + offset)
    y1, y2 = int(y1), int(y2)
    label_name = model.names[int(best_box.cls[0])]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, label_name.upper(), (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    text = (f"Label: {label_name} ({highest_confidence:.2f})\n"
            f"Coordinates: ({x1}, {y1}) to ({x2}, {y2})")
    return text, frame


def display_frame(label, frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_tk = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
    label.configure(image=image_tk)
    label.image = image_tk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaze-camera", type=int, default=0,
                        help="Camera index for the user's face (default: 0)")
    parser.add_argument("--scene-camera", type=int, default=1,
                        help="Camera index for the surrounding objects (default: 1)")
    parser.add_argument("--model", type=Path, default=PROJECT_DIR / "yolov5su.pt",
                        help="Path to an Ultralytics object detection checkpoint")
    args = parser.parse_args()
    if args.gaze_camera == args.scene_camera:
        parser.error("Use different camera indices for gaze and scene views.")

    load_predictor()
    if not args.model.is_file():
        parser.error(f"Model not found: {args.model}. See README.md for setup.")
    model = YOLO(str(args.model))
    cap_gaze = cv2.VideoCapture(args.gaze_camera)
    cap_scene = cv2.VideoCapture(args.scene_camera)
    root = None
    try:
        if not cap_gaze.isOpened() or not cap_scene.isOpened():
            raise RuntimeError("Cannot open both cameras. Check connections, permissions, and camera indices.")
        root = tk.Tk()
        root.title("Prosthetic Arm CV Interface — SIH 2025")
        tk.Label(root, text="Gaze-Guided Object Selection", bg="#D35400", fg="white",
                 font=("Arial", 26)).grid(row=0, column=0, columnspan=2, sticky="ew", pady=10)
        tk.Label(root, text="User-facing camera · Gaze tracking").grid(row=1, column=0)
        tk.Label(root, text="Scene-facing camera · Object detection").grid(row=1, column=1)
        gaze_view, scene_view = tk.Label(root), tk.Label(root)
        gaze_view.grid(row=2, column=0, padx=10)
        scene_view.grid(row=2, column=1, padx=10)
        gaze_status, scene_status = tk.Label(root), tk.Label(root)
        gaze_status.grid(row=3, column=0, pady=10)
        scene_status.grid(row=3, column=1, pady=10)

        def update_frame():
            ret_gaze, frame_gaze = cap_gaze.read()
            ret_scene, frame_scene = cap_scene.read()
            direction = "None"
            if ret_gaze:
                frame_gaze = cv2.resize(frame_gaze, (FRAME_WIDTH, FRAME_HEIGHT))
                frame_gaze, gaze_info = process_gaze_frame(frame_gaze)
                direction = gaze_info["direction"]
                display_frame(gaze_view, frame_gaze)
                gaze_status.configure(text=(f"Gaze: {direction}\n"
                                           f"Left eye center: {gaze_info['left_eye_center']}\n"
                                           f"Right eye center: {gaze_info['right_eye_center']}"))
            else:
                gaze_view.configure(image="")
                gaze_status.configure(text="Gaze camera unavailable; retrying…")

            if ret_scene:
                frame_scene = cv2.resize(frame_scene, (FRAME_WIDTH, FRAME_HEIGHT))
                coords_text, frame_scene = process_frame(model, frame_scene, direction)
                display_frame(scene_view, frame_scene)
                scene_status.configure(text=coords_text)
            else:
                scene_view.configure(image="")
                scene_status.configure(text="Scene camera unavailable; retrying…")
            root.after(10, update_frame)

        root.after(0, update_frame)
        root.mainloop()
    finally:
        cap_gaze.release()
        cap_scene.release()
        release_resources()
        if root is not None:
            try:
                root.destroy()
            except tk.TclError:
                pass


if __name__ == "__main__":
    main()
