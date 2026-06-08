import sys
import csv
from pathlib import Path
import math
import cv2
import numpy as np

from PyQt5.QtCore import Qt, QPoint, QRect, QSize, pyqtSignal
from PyQt5.QtGui import (
    QImage, QPixmap, QPainter, QPen, QColor, QFont,
    QLinearGradient, QPalette, QBrush
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QFileDialog, QVBoxLayout, QHBoxLayout, QScrollArea,
    QTextEdit, QMessageBox, QFrame, QDoubleSpinBox,
    QGroupBox, QFormLayout, QLineEdit, QSizePolicy,
    QSplitter, QSpacerItem
)


# ─────────────────────────── STYLESHEET ────────────────────────────
DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #0f1117;
    color: #e2e8f0;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 13px;
}

QScrollArea {
    background-color: #0f1117;
    border: 1px solid #1e2535;
    border-radius: 6px;
}

QScrollBar:vertical {
    background: #1a1f2e;
    width: 10px;
    border-radius: 5px;
}
QScrollBar::handle:vertical {
    background: #2d3a52;
    border-radius: 5px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover {
    background: #3d5a80;
}
QScrollBar:horizontal {
    background: #1a1f2e;
    height: 10px;
    border-radius: 5px;
}
QScrollBar::handle:horizontal {
    background: #2d3a52;
    border-radius: 5px;
    min-width: 20px;
}
QScrollBar::handle:horizontal:hover {
    background: #3d5a80;
}
QScrollBar::add-line, QScrollBar::sub-line {
    width: 0; height: 0;
}

QPushButton {
    background-color: #1a2035;
    color: #93c5fd;
    border: 1px solid #2d3a52;
    border-radius: 6px;
    padding: 8px 16px;
    font-family: 'Consolas', monospace;
    font-size: 12px;
    font-weight: bold;
    letter-spacing: 0.5px;
}
QPushButton:hover {
    background-color: #1e3050;
    border-color: #3b82f6;
    color: #bfdbfe;
}
QPushButton:pressed {
    background-color: #1d4ed8;
    border-color: #60a5fa;
    color: #ffffff;
}
QPushButton:disabled {
    background-color: #141824;
    color: #3d4a60;
    border-color: #1e2535;
}

QPushButton#danger {
    background-color: #1f1520;
    color: #f87171;
    border-color: #3d1f2a;
}
QPushButton#danger:hover {
    background-color: #2d1b1b;
    border-color: #ef4444;
}

QPushButton#success {
    background-color: #0f2520;
    color: #34d399;
    border-color: #1e4a40;
}
QPushButton#success:hover {
    background-color: #123020;
    border-color: #10b981;
}

QPushButton#accent {
    background-color: #1a1040;
    color: #a78bfa;
    border-color: #2d1f5e;
}
QPushButton#accent:hover {
    background-color: #221550;
    border-color: #7c3aed;
}

QTextEdit {
    background-color: #0d1220;
    color: #7dd3fc;
    border: 1px solid #1e2d45;
    border-radius: 6px;
    padding: 10px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12px;
    selection-background-color: #1e3a5f;
}

QGroupBox {
    background-color: #121826;
    border: 1px solid #1e2d45;
    border-radius: 8px;
    margin-top: 12px;
    padding: 12px 8px 8px 8px;
    font-weight: bold;
    color: #64748b;
    font-size: 11px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    color: #475569;
    background: transparent;
}

QDoubleSpinBox, QLineEdit {
    background-color: #0d1220;
    color: #e2e8f0;
    border: 1px solid #1e2d45;
    border-radius: 5px;
    padding: 5px 8px;
    font-family: 'Consolas', monospace;
    font-size: 12px;
    selection-background-color: #1e3a5f;
}
QDoubleSpinBox:focus, QLineEdit:focus {
    border-color: #3b82f6;
    background-color: #0f1830;
}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    background: #1a2035;
    border: none;
    width: 18px;
}
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
    background: #2d3a52;
}
QDoubleSpinBox::up-arrow {
    image: none;
    width: 0; height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-bottom: 5px solid #64748b;
}
QDoubleSpinBox::down-arrow {
    image: none;
}

QLabel#header {
    color: #38bdf8;
    font-size: 15px;
    font-weight: bold;
    letter-spacing: 2px;
    padding: 4px 0;
}
QLabel#subheader {
    color: #475569;
    font-size: 10px;
    letter-spacing: 3px;
}
QLabel#result_label {
    color: #22d3ee;
    font-size: 13px;
    font-weight: bold;
    padding: 6px;
    background: #0a1a2a;
    border: 1px solid #1e3a55;
    border-radius: 5px;
}
QLabel#section_title {
    color: #475569;
    font-size: 10px;
    letter-spacing: 2px;
    font-weight: bold;
}
QLabel#value_display {
    color: #34d399;
    font-size: 14px;
    font-family: 'Consolas', monospace;
    font-weight: bold;
    background: #071a10;
    border: 1px solid #1a4a30;
    border-radius: 5px;
    padding: 6px 10px;
}

QFrame#divider {
    color: #1e2d45;
    max-height: 1px;
    background-color: #1e2d45;
}
QFrame#sidebar {
    background-color: #0d1220;
    border-right: 1px solid #1e2d45;
}
QFrame#canvas_container {
    background-color: #0a0f1a;
    border: 1px solid #1a2540;
    border-radius: 6px;
}
QFrame#status_bar {
    background-color: #080e1a;
    border-top: 1px solid #1e2d45;
    padding: 4px 12px;
}
"""


# ──────────────────────────── CANVAS ───────────────────────────────
class ImageCanvas(QLabel):
    def __init__(self):
        super().__init__()
        self.original_img = None
        self.display_img = None
        self.qpixmap = None
        self.zoom_factor = 1.0
        self.min_zoom = 0.10
        self.max_zoom = 5.0
        self.zoom_step = 1.15
        self.roi_rect = None
        self.temp_rect = None
        self.points = []

        # Direct user calibration: 1 pixel = N millimeters.
        # Example: if user enters 20, then 10 px = 200 mm.
        self.mm_per_px = 1.0

        self.mode = "idle"
        self.start_point = None

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setStyleSheet("background-color: #070b14;")
        self.setCursor(Qt.CrossCursor)

    def load_image(self, image_path):
        self.original_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if self.original_img is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        self.display_img = self._make_display(self.original_img)
        self.qpixmap = self._cv_to_pixmap(self.display_img)

        # Important: do not use QLabel.setPixmap for huge images.
        # We will draw manually inside paintEvent.
        self.clear()

        self.zoom_factor = 1.0
        self._apply_zoom()

        self.roi_rect = None
        self.temp_rect = None
        self.points = []
        self.mode = "idle"
        self.update()

    def _make_display(self, img):
        if img.dtype == np.uint8:
            disp = img.copy()
        else:
            disp = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if len(disp.shape) == 2:
            disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
        return disp

    def _cv_to_pixmap(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        qimg = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg.copy())

    def _apply_zoom(self):
        """
        Memory-safe zoom.
        Do NOT create a new scaled QPixmap.
        Only resize the canvas; paintEvent will draw the original pixmap with scaling.
        """
        if self.qpixmap is None:
            return

        new_w = max(1, int(self.qpixmap.width() * self.zoom_factor))
        new_h = max(1, int(self.qpixmap.height() * self.zoom_factor))

        self.setFixedSize(new_w, new_h)
        self.update()


    def _view_to_image_point(self, pos):
        """Convert mouse position on zoomed display to original image coordinates."""
        x = int(pos.x() / self.zoom_factor)
        y = int(pos.y() / self.zoom_factor)
        return QPoint(x, y)


    def _get_parent_scroll_area(self):
        """Find the QScrollArea that contains this canvas."""
        parent = self.parent()
        while parent is not None:
            if isinstance(parent, QScrollArea):
                return parent
            parent = parent.parent()
        return None


    def wheelEvent(self, event):
        """
        Ctrl + Mouse Wheel zoom.
        Normal mouse wheel still scrolls the image.
        """
        if self.original_img is None:
            event.ignore()
            return

        if event.modifiers() & Qt.ControlModifier:
            old_zoom = self.zoom_factor

            if event.angleDelta().y() > 0:
                self.zoom_factor *= self.zoom_step
            else:
                self.zoom_factor /= self.zoom_step

            self.zoom_factor = max(self.min_zoom, min(self.max_zoom, self.zoom_factor))

            if abs(self.zoom_factor - old_zoom) < 1e-9:
                event.accept()
                return

            scroll_area = self._get_parent_scroll_area()

            # Keep zoom centered around mouse position
            image_x = event.pos().x() / old_zoom
            image_y = event.pos().y() / old_zoom

            if scroll_area is not None:
                viewport_pos = self.mapTo(scroll_area.viewport(), event.pos())

            self._apply_zoom()

            if scroll_area is not None:
                hbar = scroll_area.horizontalScrollBar()
                vbar = scroll_area.verticalScrollBar()

                new_canvas_x = image_x * self.zoom_factor
                new_canvas_y = image_y * self.zoom_factor

                hbar.setValue(int(new_canvas_x - viewport_pos.x()))
                vbar.setValue(int(new_canvas_y - viewport_pos.y()))

            event.accept()
        else:
            super().wheelEvent(event)

    def set_mm_per_px(self, value: float):
        """Update calibration used for all displayed measurements."""
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 1.0
        self.mm_per_px = max(value, 1e-12)
        self.update()

    def px_to_mm(self, px_value: float) -> float:
        return float(px_value) * self.mm_per_px

    def set_draw_roi_mode(self):
        self.mode = "draw_roi"
        self.points = []
        self.temp_rect = None
        self.setCursor(Qt.CrossCursor)
        self.update()

    def set_pick_points_mode(self):
        if self.roi_rect is None:
            QMessageBox.warning(self, "ROI Missing", "Please draw ROI first.")
            return
        self.mode = "pick_points"
        self.points = []
        self.setCursor(Qt.PointingHandCursor)
        self.update()

    def reset_all(self):
        self.roi_rect = None
        self.temp_rect = None
        self.points = []
        self.mode = "idle"
        self.setCursor(Qt.ArrowCursor)
        self.update()

    def is_inside_image(self, x, y):
        if self.original_img is None:
            return False
        h, w = self.original_img.shape[:2]
        return 0 <= x < w and 0 <= y < h

    def get_pixel_value(self, x, y):
        val = self.original_img[y, x]
        return val.tolist() if isinstance(val, np.ndarray) else int(val)

    def mousePressEvent(self, event):
        if self.original_img is None:
            return
        img_pos = self._view_to_image_point(event.pos())
        x, y = img_pos.x(), img_pos.y()
        if not self.is_inside_image(x, y):
            return

        if event.button() == Qt.LeftButton:
            if self.mode == "draw_roi":
                self.start_point = img_pos
                self.temp_rect = QRect(self.start_point, self.start_point)
            elif self.mode == "pick_points":
                if self.roi_rect is None:
                    return
                if not self.roi_rect.contains(QPoint(x, y)):
                    QMessageBox.warning(self, "Outside ROI",
                        "Please click only inside the selected ROI.")
                    return
                if len(self.points) < 4:
                    self.points.append(QPoint(x, y))
                    self.update()
                if len(self.points) == 4:
                    self.mode = "idle"
                    self.setCursor(Qt.ArrowCursor)

    def mouseMoveEvent(self, event):
        if self.mode == "draw_roi" and self.start_point is not None:
            img_pos = self._view_to_image_point(event.pos())
            self.temp_rect = QRect(self.start_point, img_pos).normalized()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.mode == "draw_roi" and self.start_point is not None:
            img_pos = self._view_to_image_point(event.pos())
            rect = QRect(self.start_point, img_pos).normalized()
            if rect.width() > 5 and rect.height() > 5:
                self.roi_rect = rect
                self.points = []
            self.temp_rect = None
            self.start_point = None
            self.mode = "idle"
            self.setCursor(Qt.ArrowCursor)
            self.update()

    def get_measurement_result(self):
        if self.roi_rect is None:
            return "[ NO ROI SELECTED ]"
        result = []
        rx, ry = self.roi_rect.x(), self.roi_rect.y()
        rw, rh = self.roi_rect.width(), self.roi_rect.height()

        result.append("════════ CALIBRATION ════════")
        result.append(f"  1 px = {self.mm_per_px:.6f} mm")
        result.append("")

        result.append("══════════ ROI ══════════")
        result.append(f"  Origin  : ({rx}, {ry}) px")
        result.append(f"  Size px : {rw} × {rh} px")
        result.append(f"  Size mm : {self.px_to_mm(rw):.4f} × {self.px_to_mm(rh):.4f} mm")

        if self.points:
            result.append("")
            result.append("══════ POINT VALUES ═════")
            for i, p in enumerate(self.points, 1):
                roi_x, roi_y = p.x() - rx, p.y() - ry
                pv = self.get_pixel_value(p.x(), p.y())
                result.append(f"  P{i}: ({p.x()},{p.y()}) roi({roi_x},{roi_y}) val={pv}")

        if len(self.points) >= 2:
            p1, p2 = self.points[0], self.points[1]
            w_px = math.hypot(p2.x() - p1.x(), p2.y() - p1.y())
            w_mm = self.px_to_mm(w_px)
            result.append("")
            result.append("══════════ WIDTH ════════")
            result.append(f"  P1 → P2 : {w_px:.2f} px")
            result.append(f"  P1 → P2 : {w_mm:.4f} mm")

        if len(self.points) == 4:
            p3, p4 = self.points[2], self.points[3]
            h_px = math.hypot(p4.x() - p3.x(), p4.y() - p3.y())
            h_mm = self.px_to_mm(h_px)
            result.append("")
            result.append("══════════ HEIGHT ═══════")
            result.append(f"  P3 → P4 : {h_px:.2f} px")
            result.append(f"  P3 → P4 : {h_mm:.4f} mm")

        return "\n".join(result)

    def save_results(self, out_dir="roi_4point_results"):
        if self.roi_rect is None:
            QMessageBox.warning(self, "No ROI", "Please draw ROI first.")
            return
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True)

        csv_path = out_dir / "roi_4point_measurement.csv"
        overlay_path = out_dir / "roi_4point_overlay.png"

        rx, ry = self.roi_rect.x(), self.roi_rect.y()
        rw, rh = self.roi_rect.width(), self.roi_rect.height()

        rows = []
        for i, p in enumerate(self.points, 1):
            rows.append({
                "point_id": i,
                "full_x": p.x(), "full_y": p.y(),
                "roi_x": p.x() - rx, "roi_y": p.y() - ry,
                "pixel_value": self.get_pixel_value(p.x(), p.y())
            })

        measurement = {
            "mm_per_px": self.mm_per_px,
            "roi_x": rx, "roi_y": ry,
            "roi_width_px": rw, "roi_height_px": rh,
            "roi_width_mm": self.px_to_mm(rw),
            "roi_height_mm": self.px_to_mm(rh),
            "line_width_px": "", "line_width_mm": "",
            "line_height_px": "", "line_height_mm": "",
            "axis_width_px": "", "axis_height_px": "",
            "axis_width_mm": "", "axis_height_mm": "",
            "rotated_width_px": "", "rotated_height_px": "",
            "rotated_width_mm": "", "rotated_height_mm": ""
        }

        if len(self.points) >= 2:
            p1, p2 = self.points[0], self.points[1]
            line_w_px = math.hypot(p2.x() - p1.x(), p2.y() - p1.y())
            measurement["line_width_px"] = line_w_px
            measurement["line_width_mm"] = self.px_to_mm(line_w_px)

        if len(self.points) == 4:
            p3, p4 = self.points[2], self.points[3]
            line_h_px = math.hypot(p4.x() - p3.x(), p4.y() - p3.y())
            measurement["line_height_px"] = line_h_px
            measurement["line_height_mm"] = self.px_to_mm(line_h_px)

            pts = np.array([[p.x(), p.y()] for p in self.points], dtype=np.float32)
            axis_w_px = float(np.max(pts[:, 0]) - np.min(pts[:, 0]))
            axis_h_px = float(np.max(pts[:, 1]) - np.min(pts[:, 1]))
            measurement["axis_width_px"] = axis_w_px
            measurement["axis_height_px"] = axis_h_px
            measurement["axis_width_mm"] = self.px_to_mm(axis_w_px)
            measurement["axis_height_mm"] = self.px_to_mm(axis_h_px)
            rr = cv2.minAreaRect(pts)
            measurement["rotated_width_px"] = float(rr[1][0])
            measurement["rotated_height_px"] = float(rr[1][1])
            measurement["rotated_width_mm"] = self.px_to_mm(float(rr[1][0]))
            measurement["rotated_height_mm"] = self.px_to_mm(float(rr[1][1]))

        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ROI and 4 Point Measurement"])
            for k, v in measurement.items():
                w.writerow([k, v])
            w.writerow([])
            w.writerow(["point_id", "full_x", "full_y", "roi_x", "roi_y", "pixel_value"])
            for row in rows:
                w.writerow([row["point_id"], row["full_x"], row["full_y"],
                             row["roi_x"], row["roi_y"], row["pixel_value"]])

        overlay = self.display_img.copy()
        cv2.rectangle(overlay, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)

        for i, p in enumerate(self.points, 1):
            cv2.circle(overlay, (p.x(), p.y()), 5, (0, 0, 255), -1)
            cv2.putText(overlay, f"P{i}", (p.x() + 8, p.y() - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if len(self.points) >= 2:
            p1, p2 = self.points[0], self.points[1]
            w_px = math.hypot(p2.x() - p1.x(), p2.y() - p1.y())
            mx, my = int((p1.x() + p2.x()) / 2), int((p1.y() + p2.y()) / 2)
            cv2.line(overlay, (p1.x(), p1.y()), (p2.x(), p2.y()), (0, 255, 0), 2)
            cv2.putText(overlay, f"W: {w_px:.1f}px / {self.px_to_mm(w_px):.2f}mm", (mx + 8, my - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        if len(self.points) == 4:
            p3, p4 = self.points[2], self.points[3]
            h_px = math.hypot(p4.x() - p3.x(), p4.y() - p3.y())
            mx, my = int((p3.x() + p4.x()) / 2), int((p3.y() + p4.y()) / 2)
            cv2.line(overlay, (p3.x(), p3.y()), (p4.x(), p4.y()), (255, 255, 0), 2)
            cv2.putText(overlay, f"H: {h_px:.1f}px / {self.px_to_mm(h_px):.2f}mm", (mx + 8, my - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

            pts_i = np.array([[p.x(), p.y()] for p in self.points], dtype=np.int32)
            cv2.polylines(overlay, [pts_i], True, (0, 255, 0), 2)
            box = cv2.boxPoints(cv2.minAreaRect(pts_i.astype(np.float32)))
            cv2.polylines(overlay, [box.astype(np.int32)], True, (255, 0, 0), 2)

        cv2.imwrite(str(overlay_path), overlay)
        QMessageBox.information(self, "Saved",
            f"CSV:\n{csv_path}\n\nOverlay:\n{overlay_path}")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#070b14"))

        if self.qpixmap is None:
            return

        painter.setRenderHint(QPainter.Antialiasing)

        # For large images, avoid expensive smooth transform while zooming.
        painter.setRenderHint(QPainter.SmoothPixmapTransform, False)

        # Convert visible widget area to original image coordinate area
        visible = event.rect()
        z = max(self.zoom_factor, 1e-9)

        src = QRect(
            int(visible.x() / z),
            int(visible.y() / z),
            int(visible.width() / z) + 4,
            int(visible.height() / z) + 4
        ).intersected(self.qpixmap.rect())

        # Draw only visible image region, not the full 42000px image every repaint
        painter.scale(self.zoom_factor, self.zoom_factor)
        painter.drawPixmap(src, self.qpixmap, src)

        painter.setFont(QFont("Consolas", 10, QFont.Bold))

        # ROI rectangle
        if self.roi_rect is not None:
            painter.setPen(QPen(QColor(255, 220, 0), 2))
            painter.drawRect(self.roi_rect)
            rx, ry = self.roi_rect.x(), self.roi_rect.y()
            rw, rh = self.roi_rect.width(), self.roi_rect.height()

            # Corner marks
            corner_len = 12
            for cx, cy, dx, dy in [
                (rx, ry, 1, 1), (rx + rw, ry, -1, 1),
                (rx, ry + rh, 1, -1), (rx + rw, ry + rh, -1, -1)
            ]:
                painter.setPen(QPen(QColor(255, 220, 0), 3))
                painter.drawLine(cx, cy, cx + dx * corner_len, cy)
                painter.drawLine(cx, cy, cx, cy + dy * corner_len)

            # Label
            painter.setPen(QPen(QColor(255, 220, 0), 1))
            label = f" {rw}×{rh} px "
            painter.fillRect(rx, max(0, ry - 20), len(label) * 7 + 4, 18,
                             QColor(0, 0, 0, 160))
            painter.drawText(rx + 2, max(14, ry - 5), label)

        # Temp drawing rect
        if self.temp_rect is not None:
            painter.setPen(QPen(QColor(0, 200, 255), 1, Qt.DashLine))
            painter.drawRect(self.temp_rect)

        # Width line (P1–P2)
        if len(self.points) >= 2:
            p1, p2 = self.points[0], self.points[1]
            painter.setPen(QPen(QColor(74, 222, 128), 2))
            painter.drawLine(p1, p2)
            w_px = math.hypot(p2.x() - p1.x(), p2.y() - p1.y())
            mx, my = (p1.x() + p2.x()) // 2, (p1.y() + p2.y()) // 2
            w_mm = self.px_to_mm(w_px)
            label = f" W:{w_px:.1f}px / {w_mm:.2f}mm "
            painter.fillRect(mx, my - 16, len(label) * 7, 16, QColor(0, 0, 0, 160))
            painter.setPen(QColor(74, 222, 128))
            painter.drawText(mx, my - 3, label)

        # Height line (P3–P4)
        if len(self.points) == 4:
            p3, p4 = self.points[2], self.points[3]
            painter.setPen(QPen(QColor(56, 189, 248), 2))
            painter.drawLine(p3, p4)
            h_px = math.hypot(p4.x() - p3.x(), p4.y() - p3.y())
            mx, my = (p3.x() + p4.x()) // 2, (p3.y() + p4.y()) // 2
            h_mm = self.px_to_mm(h_px)
            label = f" H:{h_px:.1f}px / {h_mm:.2f}mm "
            painter.fillRect(mx, my - 16, len(label) * 7, 16, QColor(0, 0, 0, 160))
            painter.setPen(QColor(56, 189, 248))
            painter.drawText(mx, my - 3, label)

        # Points
        colors = [
            QColor(74, 222, 128),   # P1 green
            QColor(74, 222, 128),   # P2 green
            QColor(56, 189, 248),   # P3 cyan
            QColor(56, 189, 248),   # P4 cyan
        ]
        for i, p in enumerate(self.points):
            c = colors[i] if i < len(colors) else QColor(255, 100, 100)
            # Outer ring
            painter.setPen(QPen(c, 2))
            painter.setBrush(QColor(0, 0, 0, 0))
            painter.drawEllipse(p, 8, 8)
            # Inner dot
            painter.setBrush(c)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(p, 3, 3)
            # Label
            painter.setPen(c)
            painter.setFont(QFont("Consolas", 9, QFont.Bold))
            painter.fillRect(p.x() + 10, p.y() - 18, 22, 16, QColor(0, 0, 0, 180))
            painter.drawText(p.x() + 11, p.y() - 5, f"P{i+1}")


# ─────────────────────────── PX/MM PANEL ───────────────────────────
class PxMmPanel(QFrame):
    """Sidebar widget for direct conversion using user input: 1 px = N mm."""

    calibrationChanged = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.setObjectName("sidebar")
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        # ── Header ──
        title = QLabel("PX → MM")
        title.setObjectName("header")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        sub = QLabel("USER SCALE CALIBRATION")
        sub.setObjectName("subheader")
        sub.setAlignment(Qt.AlignCenter)
        layout.addWidget(sub)

        div = QFrame()
        div.setObjectName("divider")
        div.setFrameShape(QFrame.HLine)
        layout.addWidget(div)

        # ── Direct scale group ──
        scale_group = QGroupBox("CALIBRATION")
        scale_layout = QFormLayout(scale_group)
        scale_layout.setLabelAlignment(Qt.AlignRight)
        scale_layout.setSpacing(8)

        lbl_direct = QLabel("1 px =")
        lbl_direct.setStyleSheet("color: #64748b; font-size: 11px;")
        self.mm_per_px_input = QDoubleSpinBox()
        self.mm_per_px_input.setRange(0.000001, 1e9)
        self.mm_per_px_input.setValue(1.0)
        self.mm_per_px_input.setDecimals(6)
        self.mm_per_px_input.setSuffix("  mm")
        self.mm_per_px_input.setToolTip("Enter how many millimeters one pixel represents. Example: 20 means 1 px = 20 mm.")
        self.mm_per_px_input.valueChanged.connect(self._update_all)
        scale_layout.addRow(lbl_direct, self.mm_per_px_input)

        self.scale_hint = QLabel("Example: 1 px = 20 mm → 10 px = 200 mm")
        self.scale_hint.setWordWrap(True)
        self.scale_hint.setStyleSheet("color: #64748b; font-size: 10px; padding-top: 4px;")
        scale_layout.addRow("", self.scale_hint)

        layout.addWidget(scale_group)

        # ── PX → MM ──
        px_group = QGroupBox("PIXELS  →  MM")
        px_layout = QFormLayout(px_group)
        px_layout.setSpacing(8)

        lbl_px = QLabel("Pixels:")
        lbl_px.setStyleSheet("color: #64748b; font-size: 11px;")
        self.px_input = QDoubleSpinBox()
        self.px_input.setRange(0, 1e9)
        self.px_input.setDecimals(3)
        self.px_input.setSuffix("  px")
        self.px_input.valueChanged.connect(self._px_changed)
        px_layout.addRow(lbl_px, self.px_input)

        lbl_mm_res = QLabel("= mm:")
        lbl_mm_res.setStyleSheet("color: #64748b; font-size: 11px;")
        self.mm_result = QLabel("0.000 mm")
        self.mm_result.setObjectName("value_display")
        self.mm_result.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        px_layout.addRow(lbl_mm_res, self.mm_result)

        layout.addWidget(px_group)

        # ── MM → PX ──
        mm_group = QGroupBox("MM  →  PIXELS")
        mm_layout = QFormLayout(mm_group)
        mm_layout.setSpacing(8)

        lbl_mm = QLabel("Millimeters:")
        lbl_mm.setStyleSheet("color: #64748b; font-size: 11px;")
        self.mm_input = QDoubleSpinBox()
        self.mm_input.setRange(0, 1e12)
        self.mm_input.setDecimals(4)
        self.mm_input.setSuffix("  mm")
        self.mm_input.valueChanged.connect(self._mm_changed)
        mm_layout.addRow(lbl_mm, self.mm_input)

        lbl_px_res = QLabel("= px:")
        lbl_px_res.setStyleSheet("color: #64748b; font-size: 11px;")
        self.px_result = QLabel("0.000 px")
        self.px_result.setObjectName("value_display")
        self.px_result.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        mm_layout.addRow(lbl_px_res, self.px_result)

        layout.addWidget(mm_group)

        # ── Quick reference table ──
        ref_group = QGroupBox("QUICK REFERENCE")
        ref_layout = QVBoxLayout(ref_group)
        ref_layout.setSpacing(4)

        self.ref_labels = []
        for px_val in [1, 10, 50, 100, 500, 1000]:
            row = QHBoxLayout()
            lbl_px_v = QLabel(f"{px_val} px")
            lbl_px_v.setStyleSheet("color: #4b6080; font-size: 11px; font-family: Consolas;")
            lbl_px_v.setFixedWidth(55)
            lbl_eq = QLabel("=")
            lbl_eq.setStyleSheet("color: #2d3a52; font-size: 11px;")
            lbl_eq.setFixedWidth(14)
            lbl_mm_v = QLabel("0.000 mm")
            lbl_mm_v.setStyleSheet(
                "color: #34d399; font-size: 11px; font-family: Consolas; font-weight: bold;")
            row.addWidget(lbl_px_v)
            row.addWidget(lbl_eq)
            row.addWidget(lbl_mm_v)
            row.addStretch()
            ref_layout.addLayout(row)
            self.ref_labels.append((px_val, lbl_mm_v))

        layout.addWidget(ref_group)
        layout.addStretch()
        self._update_all()

    def mm_per_px(self) -> float:
        return max(float(self.mm_per_px_input.value()), 1e-12)

    def px_to_mm(self, px_value: float) -> float:
        return float(px_value) * self.mm_per_px()

    def mm_to_px(self, mm_value: float) -> float:
        return float(mm_value) / self.mm_per_px()

    def _px_changed(self, val):
        mm = self.px_to_mm(val)
        self.mm_result.setText(f"{mm:.4f} mm")

    def _mm_changed(self, val):
        px = self.mm_to_px(val)
        self.px_result.setText(f"{px:.3f} px")

    def _update_all(self):
        self._px_changed(self.px_input.value())
        self._mm_changed(self.mm_input.value())
        for px_val, lbl in self.ref_labels:
            lbl.setText(f"{self.px_to_mm(px_val):.4f} mm")
        self.calibrationChanged.emit(self.mm_per_px())

    def inject_px(self, px_value: float):
        """Call from main window to auto-fill a measured px value."""
        self.px_input.setValue(px_value)



# ──────────────────────────── MAIN WINDOW ──────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ROI · 4-Point Measurement  //  direct px→mm")
        self.resize(1440, 860)
        self.setMinimumSize(900, 600)
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Top bar ──
        top_bar = QFrame()
        top_bar.setStyleSheet(
            "background: #070c18; border-bottom: 1px solid #1a2540;")
        top_bar.setFixedHeight(52)
        top_bar_layout = QHBoxLayout(top_bar)
        top_bar_layout.setContentsMargins(18, 0, 18, 0)
        top_bar_layout.setSpacing(8)

        app_title = QLabel("◈  VISION MEASURE")
        app_title.setStyleSheet(
            "color: #38bdf8; font-size: 16px; font-weight: bold; "
            "letter-spacing: 3px; font-family: Consolas;")
        top_bar_layout.addWidget(app_title)

        app_sub = QLabel("ROI + 4-POINT PIXEL TOOL")
        app_sub.setStyleSheet(
            "color: #2d3a52; font-size: 10px; letter-spacing: 2px; "
            "font-family: Consolas; padding-top: 4px;")
        top_bar_layout.addWidget(app_sub)
        top_bar_layout.addStretch()

        self.status_label = QLabel("● READY")
        self.status_label.setStyleSheet(
            "color: #22c55e; font-size: 11px; font-family: Consolas; letter-spacing: 1px;")
        top_bar_layout.addWidget(self.status_label)
        root.addWidget(top_bar)

        # ── Toolbar ──
        toolbar = QFrame()
        toolbar.setStyleSheet(
            "background: #0a0f1a; border-bottom: 1px solid #141e30;")
        toolbar.setFixedHeight(52)
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(12, 0, 12, 0)
        tb_layout.setSpacing(6)

        self.btn_load = QPushButton("⊕  LOAD IMAGE")
        self.btn_draw_roi = QPushButton("⬜  DRAW ROI")
        self.btn_pick_points = QPushButton("⊙  PICK 4 PTS")
        self.btn_measure = QPushButton("◎  CALCULATE")
        self.btn_save = QPushButton("⊞  SAVE")
        self.btn_reset = QPushButton("↺  RESET")

        self.btn_draw_roi.setObjectName("accent")
        self.btn_measure.setObjectName("success")
        self.btn_save.setObjectName("success")
        self.btn_reset.setObjectName("danger")

        for btn in [self.btn_load, self.btn_draw_roi, self.btn_pick_points,
                    self.btn_measure, self.btn_save, self.btn_reset]:
            btn.setFixedHeight(34)
            tb_layout.addWidget(btn)

        tb_layout.addStretch()

        self.mode_indicator = QLabel("MODE: IDLE")
        self.mode_indicator.setStyleSheet(
            "color: #3d4a60; font-size: 10px; font-family: Consolas; letter-spacing: 2px;")
        tb_layout.addWidget(self.mode_indicator)

        root.addWidget(toolbar)

        # ── Main area (3-pane) ──
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setHandleWidth(2)
        main_splitter.setStyleSheet(
            "QSplitter::handle { background: #1e2d45; }")

        # Left: px→mm panel inside its own scroll area
        self.px_mm_panel = PxMmPanel()
        self.px_mm_panel.calibrationChanged.connect(self._on_calibration_changed)

        self.sidebar_scroll = QScrollArea()
        self.sidebar_scroll.setWidget(self.px_mm_panel)
        self.sidebar_scroll.setWidgetResizable(True)
        self.sidebar_scroll.setFixedWidth(300)
        self.sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.sidebar_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        main_splitter.addWidget(self.sidebar_scroll)

        # Center: canvas
        canvas_container = QFrame()
        canvas_container.setObjectName("canvas_container")
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)

        self.canvas = ImageCanvas()
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.canvas)
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setAlignment(Qt.AlignCenter)
        canvas_layout.addWidget(self.scroll_area)

        main_splitter.addWidget(canvas_container)

        # Right: results
        right_panel = QFrame()
        right_panel.setFixedWidth(300)
        right_panel.setStyleSheet(
            "background: #0a0f1a; border-left: 1px solid #141e30;")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.setSpacing(8)

        res_title = QLabel("RESULTS")
        res_title.setObjectName("section_title")
        res_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(res_title)

        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)
        right_layout.addWidget(self.result_box)

        # Send-to-converter button
        self.btn_send_width = QPushButton("↗  SEND WIDTH → CONVERTER")
        self.btn_send_width.setObjectName("accent")
        self.btn_send_width.setToolTip(
            "Send measured width in px to the px↔mm converter")
        self.btn_send_width.clicked.connect(self._send_width_to_converter)
        right_layout.addWidget(self.btn_send_width)

        self.btn_send_height = QPushButton("↗  SEND HEIGHT → CONVERTER")
        self.btn_send_height.setObjectName("accent")
        self.btn_send_height.setToolTip(
            "Send measured height in px to the px↔mm converter")
        self.btn_send_height.clicked.connect(self._send_height_to_converter)
        right_layout.addWidget(self.btn_send_height)

        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([280, 860, 300])

        root.addWidget(main_splitter, 1)

        # ── Bottom status bar ──
        status_bar = QFrame()
        status_bar.setObjectName("status_bar")
        status_bar.setFixedHeight(26)
        sb_layout = QHBoxLayout(status_bar)
        sb_layout.setContentsMargins(14, 0, 14, 0)
        sb_layout.setSpacing(20)

        self.lbl_img_info = QLabel("No image loaded")
        self.lbl_img_info.setStyleSheet(
            "color: #2d3a52; font-size: 10px; font-family: Consolas;")
        sb_layout.addWidget(self.lbl_img_info)

        self.lbl_point_count = QLabel("Points: 0 / 4")
        self.lbl_point_count.setStyleSheet(
            "color: #2d3a52; font-size: 10px; font-family: Consolas;")
        sb_layout.addWidget(self.lbl_point_count)

        sb_layout.addStretch()

        version_lbl = QLabel("v2.0 · VISION MEASURE")
        version_lbl.setStyleSheet(
            "color: #1a2535; font-size: 10px; font-family: Consolas;")
        sb_layout.addWidget(version_lbl)

        root.addWidget(status_bar)

        # ── Connect buttons ──
        self.btn_load.clicked.connect(self.load_image)
        self.btn_draw_roi.clicked.connect(self.draw_roi)
        self.btn_pick_points.clicked.connect(self.pick_points)
        self.btn_measure.clicked.connect(self.calculate)
        self.btn_save.clicked.connect(self.save)
        self.btn_reset.clicked.connect(self.reset)

        self._set_mode_indicator("IDLE")

    # ── Helpers ──
    def _on_calibration_changed(self, mm_per_px: float):
        self.canvas.set_mm_per_px(mm_per_px)
        # Refresh the right result panel only after user has selected an ROI.
        if self.canvas.roi_rect is not None:
            self.result_box.setText(self.canvas.get_measurement_result())

    def _set_mode_indicator(self, text):
        colors = {
            "IDLE": "#3d4a60",
            "DRAW ROI": "#fbbf24",
            "PICK POINTS": "#34d399",
            "DONE": "#38bdf8",
        }
        c = colors.get(text, "#64748b")
        self.mode_indicator.setStyleSheet(
            f"color: {c}; font-size: 10px; font-family: Consolas; letter-spacing: 2px;")
        self.mode_indicator.setText(f"MODE: {text}")

    def _set_status(self, text, color="#22c55e"):
        self.status_label.setStyleSheet(
            f"color: {color}; font-size: 11px; font-family: Consolas; letter-spacing: 1px;")
        self.status_label.setText(f"● {text}")

    def _get_measured_width(self):
        pts = self.canvas.points
        if len(pts) >= 2:
            return math.hypot(pts[1].x() - pts[0].x(), pts[1].y() - pts[0].y())
        return None

    def _get_measured_height(self):
        pts = self.canvas.points
        if len(pts) == 4:
            return math.hypot(pts[3].x() - pts[2].x(), pts[3].y() - pts[2].y())
        return None

    def _send_width_to_converter(self):
        w = self._get_measured_width()
        if w is not None:
            self.px_mm_panel.inject_px(w)
            self._set_status(f"Width {w:.2f}px → converter", "#a78bfa")
        else:
            QMessageBox.information(self, "No Width",
                "Pick at least 2 points and calculate first.")

    def _send_height_to_converter(self):
        h = self._get_measured_height()
        if h is not None:
            self.px_mm_panel.inject_px(h)
            self._set_status(f"Height {h:.2f}px → converter", "#a78bfa")
        else:
            QMessageBox.information(self, "No Height",
                "Pick all 4 points and calculate first.")

    # ── Slots ──
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if not path:
            return
        try:
            self.canvas.load_image(path)
            self.canvas.set_mm_per_px(self.px_mm_panel.mm_per_px())
            h, w = self.canvas.original_img.shape[:2]
            dtype = self.canvas.original_img.dtype
            self.lbl_img_info.setText(
                f"{Path(path).name}  ·  {w}×{h}px  ·  {dtype}")
            self.result_box.setText(
                f"Image loaded ✓\n\n"
                f"  Width  : {w} px\n"
                f"  Height : {h} px\n"
                f"  Dtype  : {dtype}\n\n"
                f"Next → Draw ROI")
            self.lbl_point_count.setText("Points: 0 / 4")
            self._set_status("IMAGE LOADED")
            self._set_mode_indicator("IDLE")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))
            self._set_status("ERROR", "#ef4444")

    def draw_roi(self):
        if self.canvas.original_img is None:
            QMessageBox.warning(self, "No Image", "Load an image first.")
            return
        self.canvas.set_draw_roi_mode()
        self.result_box.setText(
            "[ DRAW ROI MODE ]\n\n"
            "Drag a rectangle on the image\nto define the region of interest.")
        self._set_mode_indicator("DRAW ROI")
        self._set_status("DRAW ROI MODE", "#fbbf24")

    def pick_points(self):
        if self.canvas.original_img is None:
            QMessageBox.warning(self, "No Image", "Load an image first.")
            return
        self.canvas.set_pick_points_mode()
        self.result_box.setText(
            "[ PICK POINTS MODE ]\n\n"
            "Click 4 points inside the ROI.\n\n"
            "  P1 + P2  →  WIDTH line\n"
            "  P3 + P4  →  HEIGHT line\n\n"
            "Then click CALCULATE.")
        self._set_mode_indicator("PICK POINTS")
        self._set_status("PICK 4 POINTS", "#34d399")

    def calculate(self):
        self.canvas.set_mm_per_px(self.px_mm_panel.mm_per_px())
        result = self.canvas.get_measurement_result()
        self.result_box.setText(result)
        self.lbl_point_count.setText(f"Points: {len(self.canvas.points)} / 4")
        self._set_mode_indicator("DONE")
        self._set_status("CALCULATED", "#38bdf8")

    def save(self):
        self.canvas.set_mm_per_px(self.px_mm_panel.mm_per_px())
        self.canvas.save_results()

    def reset(self):
        self.canvas.reset_all()
        self.result_box.setText("[ RESET ]\n\nReady for a new measurement.")
        self.lbl_point_count.setText("Points: 0 / 4")
        self._set_mode_indicator("IDLE")
        self._set_status("READY")


# ──────────────────────────── ENTRY ────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLE)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())