from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


@dataclass
class DetectionConfig:
    model_name: str = "yolo11n.pt"
    confidence: float = 0.40
    iou: float = 0.45
    person_class_id: int = 0


@dataclass
class FrameResult:
    frame: np.ndarray
    count: int
    avg_confidence: float
    density_level: str
    records: list[dict]


class CrowdDetector:
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.model = YOLO(config.model_name)

    @staticmethod
    def density_from_count(count: int, area: int) -> str:
        # These are practical screen-level heuristics, not a real-world
        # crowd safety standard. Adjust thresholds for the camera scene.
        if count == 0:
            return "Empty"
        if area <= 0:
            return "Unknown"
        normalized = count / max(area / 100_000, 1)
        if normalized < 4:
            return "Low"
        if normalized < 9:
            return "Medium"
        if normalized < 16:
            return "High"
        return "Very High"

    @staticmethod
    def _draw_density_overlay(frame: np.ndarray, count: int) -> None:
        h, w = frame.shape[:2]
        grid_y, grid_x = 3, 4
        # Count detections by center in each cell.
        # The actual heatmap is rendered from boxes in process_frame.
        for i in range(1, grid_x):
            x = int(w * i / grid_x)
            cv2.line(frame, (x, 0), (x, h), (180, 180, 180), 1)
        for i in range(1, grid_y):
            y = int(h * i / grid_y)
            cv2.line(frame, (0, y), (w, y), (180, 180, 180), 1)

    def process_frame(
        self,
        frame: np.ndarray,
        draw: bool = True,
        labels: bool = True,
        density: bool = True,
    ) -> FrameResult:
        if frame is None or frame.size == 0:
            raise ValueError("Invalid image/frame supplied.")

        h, w = frame.shape[:2]
        annotated = frame.copy()

        result = self.model.predict(
            source=frame,
            conf=self.config.confidence,
            iou=self.config.iou,
            classes=[self.config.person_class_id],
            verbose=False,
        )[0]

        records = []
        confidences = []
        centers = []

        if result.boxes is not None:
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().item())
                x1, y1, x2, y2 = xyxy.tolist()

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                records.append({
                    "class": "person",
                    "confidence": round(conf, 4),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "center_x": cx,
                    "center_y": cy,
                })
                confidences.append(conf)
                centers.append((cx, cy))

                if draw:
                    cv2.rectangle(
                        annotated, (x1, y1), (x2, y2),
                        (0, 220, 80), 2
                    )
                    if labels:
                        text = f"Person {conf:.0%}"
                        cv2.putText(
                            annotated, text, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 220, 80), 2, cv2.LINE_AA
                        )

        count = len(records)
        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        density_level = self.density_from_count(count, h * w)

        if density:
            # Subtle scene grid plus center points for visual density analysis.
            self._draw_density_overlay(annotated, count)
            for cx, cy in centers:
                cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)

        # Dashboard overlay
        cv2.rectangle(annotated, (10, 10), (285, 82), (20, 20, 20), -1)
        cv2.putText(
            annotated, f"PEOPLE: {count}", (22, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2
        )
        cv2.putText(
            annotated, f"DENSITY: {density_level}", (22, 68),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
        )

        return FrameResult(
            frame=annotated,
            count=count,
            avg_confidence=avg_conf,
            density_level=density_level,
            records=records,
        )

    def process_video(
        self,
        input_path: str,
        output_path: str,
        csv_path: str,
        draw: bool = True,
        labels: bool = True,
        density: bool = True,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        preview_callback: Optional[Callable[[np.ndarray], None]] = None,
    ) -> dict:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # mp4v is broadly available for OpenCV-generated MP4 files.
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Could not create output video writer.")

        rows = []
        counts = []
        frame_index = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                result = self.process_frame(
                    frame, draw=draw, labels=labels, density=density
                )
                writer.write(result.frame)
                counts.append(result.count)

                rows.append({
                    "frame": frame_index,
                    "timestamp_sec": round(frame_index / fps, 3),
                    "people_count": result.count,
                    "average_confidence": round(result.avg_confidence, 4),
                    "density_level": result.density_level,
                })

                frame_index += 1

                if preview_callback and (frame_index == 1 or frame_index % 15 == 0):
                    preview_callback(result.frame)

                if progress_callback and total:
                    progress = frame_index / total * 100
                    progress_callback(
                        progress,
                        f"Processed {frame_index}/{total} frames"
                    )
        finally:
            cap.release()
            writer.release()

        df = pd.DataFrame(rows)
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)

        return {
            "frames_processed": frame_index,
            "max_count": int(max(counts)) if counts else 0,
            "min_count": int(min(counts)) if counts else 0,
            "avg_count": float(np.mean(counts)) if counts else 0.0,
            "peak_density": self._peak_density(df),
            "output_video": output_path,
            "csv": csv_path,
        }

    @staticmethod
    def _peak_density(df: pd.DataFrame) -> str:
        order = {"Empty": 0, "Low": 1, "Medium": 2, "High": 3, "Very High": 4}
        if df.empty:
            return "Empty"
        peak = max(df["density_level"], key=lambda x: order.get(x, 0))
        return peak
