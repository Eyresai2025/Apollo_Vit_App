
import os, json, random, shutil
from pathlib import Path
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QLineEdit, QTextEdit, QCheckBox,
                             QDoubleSpinBox, QSpinBox, QGroupBox, QFileDialog,
                             QProgressBar, QMessageBox, QWidget, QSplitter, 
                             QListWidget, QTabWidget, QFormLayout, QComboBox,
                             QStackedWidget, QFrame)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon, QPixmap
import threading

# -----------------------
# Augmentation Worker Thread
# -----------------------
class AugmentationWorker(QThread):
    progress = pyqtSignal(int)
    message = pyqtSignal(str)
    finished_signal = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, config):
        super().__init__()
        self.config = config

    def run(self):
        try:
            self.perform_augmentation()
        except Exception as e:
            self.error.emit(str(e))

    def perform_augmentation(self):
        random.seed(self.config['seed'])

        input_dir = Path(self.config['input_dir'])
        root = Path(self.config['output_dir'])
        self.make_dataset_dirs(root)

        pairs = self.collect_pairs(input_dir)
        if not pairs:
            self.error.emit("No (image, json) pairs found.")
            return

        class_names = self.collect_labels_from_jsons(pairs)
        if not class_names:
            self.error.emit("No labels found in JSONs.")
            return
        
        class_map = self.class_map_from_names(class_names)
        self.message.emit(f"Detected {len(class_names)} classes: {class_names}")

        # Split by original image stems
        stems = [p[0].stem for p in pairs]
        random.shuffle(stems)
        cutoff = int(len(stems) * self.config['train_ratio'])
        train_stems = set(stems[:cutoff])
        valid_stems = set(stems[cutoff:])

        total_files = len(pairs)
        processed = 0

        for img_path, json_path in pairs:
            stem = img_path.stem
            split = "train" if stem in train_stems else "valid"

            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                self.message.emit(f"Skipping unreadable image: {img_path}")
                continue

            _, shapes, W_json, H_json, _ = self.load_labelme(json_path)
            H_img, W_img = img.shape[:2]
            W, H = (W_img, H_img) if (W_img, H_img) != (W_json, H_json) else (W_json, H_json)

            # Create output directories
            img_dir = root / split / "images"
            lbl_dir = root / split / "labels"

            # Save original
            out_img = img_dir / f"{stem}.jpg"
            out_txt = lbl_dir / f"{stem}.txt"
            self.save_image(out_img, img)
            self.write_yolo_txt(out_txt, shapes, W, H, class_map)

            # Apply augmentations
            if self.config['flip_horizontal'] or self.config['flip_vertical']:
                if self.config['flip_horizontal']:
                    img_h = cv2.flip(img, 1)
                    shapes_h = self.flipped_shapes(shapes, W, H, horizontal=True, vertical=False)
                    save_path = img_dir / f"{stem}_flipH.jpg"
                    self.save_image(save_path, img_h)
                    self.write_yolo_txt(lbl_dir / f"{stem}_flipH.txt", shapes_h, W, H, class_map)
                
                if self.config['flip_vertical']:
                    img_v = cv2.flip(img, 0)
                    shapes_v = self.flipped_shapes(shapes, W, H, horizontal=False, vertical=True)
                    save_path = img_dir / f"{stem}_flipV.jpg"
                    self.save_image(save_path, img_v)
                    self.write_yolo_txt(lbl_dir / f"{stem}_flipV.txt", shapes_v, W, H, class_map)

            if self.config['brightness_pct'] != 0:
                img_bp = self.adjust_brightness(img, +self.config['brightness_pct'])
                img_bm = self.adjust_brightness(img, -self.config['brightness_pct'])
                self.save_image(img_dir / f"{stem}_bplus{int(abs(self.config['brightness_pct']))}.jpg", img_bp)
                self.save_image(img_dir / f"{stem}_bminus{int(abs(self.config['brightness_pct']))}.jpg", img_bm)
                self.write_yolo_txt(lbl_dir / f"{stem}_bplus{int(abs(self.config['brightness_pct']))}.txt", shapes, W, H, class_map)
                self.write_yolo_txt(lbl_dir / f"{stem}_bminus{int(abs(self.config['brightness_pct']))}.txt", shapes, W, H, class_map)

            if self.config['saturation_pct'] != 0:
                img_sp = self.adjust_saturation(img, +self.config['saturation_pct'])
                img_sm = self.adjust_saturation(img, -self.config['saturation_pct'])
                self.save_image(img_dir / f"{stem}_splus{int(abs(self.config['saturation_pct']))}.jpg", img_sp)
                self.save_image(img_dir / f"{stem}_sminus{int(abs(self.config['saturation_pct']))}.jpg", img_sm)
                self.write_yolo_txt(lbl_dir / f"{stem}_splus{int(abs(self.config['saturation_pct']))}.txt", shapes, W, H, class_map)
                self.write_yolo_txt(lbl_dir / f"{stem}_sminus{int(abs(self.config['saturation_pct']))}.txt", shapes, W, H, class_map)

            processed += 1
            progress = int((processed / total_files) * 100)
            self.progress.emit(progress)
            self.message.emit(f"Processed {stem}")

        # Write YAML file
        self.write_yaml(root, class_names)
        self.finished_signal.emit(f"Augmentation completed! Processed {total_files} image pairs.\nClasses: {class_names}")

    def make_dataset_dirs(self, root: Path):
        for split in ["train", "valid"]:
            self.ensure_dir(root / split / "images")
            self.ensure_dir(root / split / "labels")

    def ensure_dir(self, p: Path):
        p.mkdir(parents=True, exist_ok=True)

    def is_image_file(self, p: Path):
        IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        return p.suffix.lower() in IMG_EXTS

    def collect_pairs(self, input_dir: Path):
        images = [p for p in input_dir.rglob("*") if self.is_image_file(p)]
        pairs = []
        for img in images:
            json_path = img.with_suffix(".json")
            if json_path.exists():
                pairs.append((img, json_path))
        return pairs

    def load_labelme(self, json_path: Path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            W = int(data["imageWidth"])
            H = int(data["imageHeight"])
            shapes = data.get("shapes", [])
            stem = Path(data.get("imagePath", json_path.stem)).stem
            
            self.message.emit(f"Loaded JSON: {json_path.name}")
            self.message.emit(f"Image dimensions: {W}x{H}")
            self.message.emit(f"Number of shapes: {len(shapes)}")
            
            for i, shape in enumerate(shapes):
                self.message.emit(f"Shape {i}: label='{shape.get('label')}', type='{shape.get('shape_type')}', points={len(shape.get('points', []))}")
                
            return data, shapes, W, H, stem
        except Exception as e:
            self.message.emit(f"Error loading {json_path}: {str(e)}")
            raise

    def collect_labels_from_jsons(self, pairs):
        labels, seen = [], set()
        for _, jpath in pairs:
            try:
                _, shapes, _, _, _ = self.load_labelme(jpath)
            except Exception:
                continue
            for sh in shapes:
                lbl = str(sh.get("label", "unknown"))
                if lbl not in seen:
                    seen.add(lbl)
                    labels.append(lbl)
        return sorted(labels)

    def class_map_from_names(self, names):
        return {name: idx for idx, name in enumerate(names)}

    def yolo_seg_txt_lines(self, shapes, W, H, class_map):
        lines = []
        
        for sh in shapes:
            label = str(sh.get("label", "unknown"))
            
            if label not in class_map:
                continue
                
            pts = sh.get("points", [])
            shape_type = sh.get("shape_type", "polygon")
            
            # Handle rectangles (convert to polygons)
            if shape_type == "rectangle" and len(pts) == 2:
                x1, y1 = float(pts[0][0]), float(pts[0][1])
                x2, y2 = float(pts[1][0]), float(pts[1][1])
                
                # Create polygon from rectangle corners
                polygon_points = [
                    [x1, y1],  # top-left
                    [x2, y1],  # top-right  
                    [x2, y2],  # bottom-right
                    [x1, y2]   # bottom-left
                ]
                pts = polygon_points
                
            elif shape_type != "polygon":
                continue
                
            if len(pts) < 3:
                continue

            cls_id = class_map[label]
            flat_norm = []
            for x, y in pts:
                nx = float(x) / float(W)
                ny = float(y) / float(H)
                nx = min(1.0, max(0.0, nx))
                ny = min(1.0, max(0.0, ny))
                flat_norm.extend([nx, ny])

            formatted_points = []
            for v in flat_norm:
                formatted_val = f"{v:.{self.config['float_precision']}f}"
                if '.' in formatted_val:
                    formatted_val = formatted_val.rstrip('0').rstrip('.')
                formatted_points.append(formatted_val)
            
            line = str(cls_id) + " " + " ".join(formatted_points)
            lines.append(line)
        
        return lines

    def write_yolo_txt(self, out_txt: Path, shapes, W, H, class_map):
        out_txt.parent.mkdir(parents=True, exist_ok=True)
        lines = self.yolo_seg_txt_lines(shapes, W, H, class_map)
        with open(out_txt, "w", encoding="utf-8") as f:
            if lines:
                f.write("\n".join(lines) + "\n")

    def save_image(self, out_img: Path, img):
        out_img.parent.mkdir(parents=True, exist_ok=True)
        if out_img.suffix.lower() in {".jpg", ".jpeg"}:
            cv2.imwrite(str(out_img), img, [int(cv2.IMWRITE_JPEG_QUALITY), self.config['jpeg_quality']])
        else:
            cv2.imwrite(str(out_img), img)

    def flip_points_horizontal(self, points, W):
        return [[(W - 1 - float(x)), float(y)] for (x, y) in points]

    def flip_points_vertical(self, points, H):
        return [[float(x), (H - 1 - float(y))] for (x, y) in points]

    def flipped_shapes(self, shapes, W, H, horizontal=False, vertical=False):
        new_shapes = []
        for sh in shapes:
            pts = sh.get("points", [])
            new_pts = [list(p) for p in pts]
            if horizontal:
                new_pts = self.flip_points_horizontal(new_pts, W)
            if vertical:
                new_pts = self.flip_points_vertical(new_pts, H)
            nsh = dict(sh)
            nsh["points"] = new_pts
            new_shapes.append(nsh)
        return new_shapes

    def adjust_brightness(self, img, percent):
        beta = float(percent) * 255.0 / 100.0
        out = cv2.convertScaleAbs(img, alpha=1.0, beta=beta)
        return out

    def adjust_saturation(self, img, percent):
        factor = 1.0 + float(percent) / 100.0
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = np.clip(s.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        hsv2 = cv2.merge([h, s, v])
        out = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)
        return out

    def write_yaml(self, root: Path, class_names):
        yaml_text = f"""train: train/images
val: valid/images

nc: {len(class_names)}
names: [{", ".join("'" + n.replace("'", "''") + "'" for n in class_names)}]
"""
        (root / "data.yaml").write_text(yaml_text, encoding="utf-8")

# -----------------------
# Sidebar Widget
# -----------------------
class SidebarWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 20, 10, 20)
        layout.setSpacing(20)

        # Company Logo Area
        logo_frame = QFrame()
        logo_frame.setStyleSheet("QFrame { background-color: #1a1a1a; border-radius: 10px; padding: 15px; }")
        logo_layout = QVBoxLayout(logo_frame)
        
        # Company Name (as placeholder for logo)
        company_label = QLabel("AI Vision Pro")
        company_label.setFont(QFont("Arial", 16, QFont.Bold))
        company_label.setStyleSheet("color: #ffffff; background-color: transparent;")
        company_label.setAlignment(Qt.AlignCenter)
        logo_layout.addWidget(company_label)
        
        # Tagline
        tagline_label = QLabel("Augmentation Suite")
        tagline_label.setFont(QFont("Arial", 10))
        tagline_label.setStyleSheet("color: #b0b0b0; background-color: transparent;")
        tagline_label.setAlignment(Qt.AlignCenter)
        logo_layout.addWidget(tagline_label)
        
        layout.addWidget(logo_frame)

        # Navigation Buttons
        nav_frame = QFrame()
        nav_frame.setStyleSheet("QFrame { background-color: transparent; }")
        nav_layout = QVBoxLayout(nav_frame)
        nav_layout.setSpacing(10)

        # Dataset Setup Button
        self.dataset_btn = QPushButton("📁 Dataset Setup")
        self.dataset_btn.setFont(QFont("Arial", 11))
        self.dataset_btn.setStyleSheet("""
            QPushButton {
                background-color: #2c3e50;
                color: white;
                padding: 12px;
                border: none;
                border-radius: 8px;
                text-align: left;
                padding-left: 15px;
            }
            QPushButton:hover {
                background-color: #34495e;
            }
            QPushButton:pressed {
                background-color: #1abc9c;
            }
        """)
        nav_layout.addWidget(self.dataset_btn)

        # Augmentation Settings Button
        self.augmentation_btn = QPushButton("⚙️ Augmentation Settings")
        self.augmentation_btn.setFont(QFont("Arial", 11))
        self.augmentation_btn.setStyleSheet("""
            QPushButton {
                background-color: #2c3e50;
                color: white;
                padding: 12px;
                border: none;
                border-radius: 8px;
                text-align: left;
                padding-left: 15px;
            }
            QPushButton:hover {
                background-color: #34495e;
            }
            QPushButton:pressed {
                background-color: #1abc9c;
            }
        """)
        nav_layout.addWidget(self.augmentation_btn)

        layout.addWidget(nav_frame)
        layout.addStretch()

# -----------------------
# Wizard Pages
# -----------------------
class DatasetSetupPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 20, 40, 20)
        main_layout.setSpacing(25)
        
        # Title
        title = QLabel("Dataset Configuration")
        title.setFont(QFont("Arial", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #ffffff; padding: 10px;")
        main_layout.addWidget(title)
        
        # Description
        desc = QLabel("Configure your input and output directories for the augmentation process")
        desc.setFont(QFont("Arial", 11))
        desc.setAlignment(Qt.AlignCenter)
        desc.setStyleSheet("color: #b0b0b0; padding: 5px;")
        main_layout.addWidget(desc)
        
        # Centered Form Container
        form_container = QFrame()
        form_container.setStyleSheet("""
            QFrame {
                background-color: #2c2c2c;
                border: 1px solid #404040;
                border-radius: 12px;
                padding: 30px;
            }
        """)
        form_container.setMaximumWidth(600)
        
        form_layout = QFormLayout(form_container)
        form_layout.setVerticalSpacing(15)
        form_layout.setHorizontalSpacing(20)
        form_layout.setLabelAlignment(Qt.AlignRight)
        
        # Input Directory
        self.input_dir_edit = QLineEdit()
        self.input_dir_edit.setPlaceholderText("Select directory containing images and JSON files...")
        self.input_dir_edit.setFont(QFont("Arial", 10))
        self.input_dir_edit.setStyleSheet("""
            QLineEdit {
                background-color: #1a1a1a;
                color: white;
                border: 1px solid #404040;
                border-radius: 6px;
                padding: 8px 12px;
                min-width: 300px;
            }
            QLineEdit:focus {
                border-color: #1abc9c;
            }
        """)
        input_btn = QPushButton("Browse")
        input_btn.setFont(QFont("Arial", 9))
        input_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 15px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        input_btn.clicked.connect(self.browse_input_dir)
        
        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input_dir_edit)
        input_layout.addWidget(input_btn)
        form_layout.addRow(QLabel("Input Directory:"), input_layout)
        
        # Output Directory
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Select output directory for augmented dataset...")
        self.output_dir_edit.setFont(QFont("Arial", 10))
        self.output_dir_edit.setStyleSheet("""
            QLineEdit {
                background-color: #1a1a1a;
                color: white;
                border: 1px solid #404040;
                border-radius: 6px;
                padding: 8px 12px;
                min-width: 300px;
            }
            QLineEdit:focus {
                border-color: #1abc9c;
            }
        """)
        output_btn = QPushButton("Browse")
        output_btn.setFont(QFont("Arial", 9))
        output_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 15px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        output_btn.clicked.connect(self.browse_output_dir)
        
        output_layout = QHBoxLayout()
        output_layout.addWidget(self.output_dir_edit)
        output_layout.addWidget(output_btn)
        form_layout.addRow(QLabel("Output Directory:"), output_layout)
        
        # Dataset parameters
        param_font = QFont("Arial", 10)
        label_font = QFont("Arial", 10, QFont.Bold)
        
        self.train_ratio_spin = QDoubleSpinBox()
        self.train_ratio_spin.setFont(param_font)
        self.train_ratio_spin.setRange(0.1, 0.9)
        self.train_ratio_spin.setValue(0.8)
        self.train_ratio_spin.setSingleStep(0.1)
        self.train_ratio_spin.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #1a1a1a;
                color: white;
                border: 1px solid #404040;
                border-radius: 6px;
                padding: 8px;
                min-width: 100px;
            }
        """)
        train_label = QLabel("Train/Validation Ratio:")
        train_label.setFont(label_font)
        train_label.setStyleSheet("color: #ffffff;")
        form_layout.addRow(train_label, self.train_ratio_spin)
        
        self.seed_spin = QSpinBox()
        self.seed_spin.setFont(param_font)
        self.seed_spin.setRange(0, 9999)
        self.seed_spin.setValue(42)
        self.seed_spin.setStyleSheet("""
            QSpinBox {
                background-color: #1a1a1a;
                color: white;
                border: 1px solid #404040;
                border-radius: 6px;
                padding: 8px;
                min-width: 100px;
            }
        """)
        seed_label = QLabel("Random Seed:")
        seed_label.setFont(label_font)
        seed_label.setStyleSheet("color: #ffffff;")
        form_layout.addRow(seed_label, self.seed_spin)
        
        self.quality_spin = QSpinBox()
        self.quality_spin.setFont(param_font)
        self.quality_spin.setRange(1, 100)
        self.quality_spin.setValue(95)
        self.quality_spin.setStyleSheet("""
            QSpinBox {
                background-color: #1a1a1a;
                color: white;
                border: 1px solid #404040;
                border-radius: 6px;
                padding: 8px;
                min-width: 100px;
            }
        """)
        quality_label = QLabel("JPEG Quality:")
        quality_label.setFont(label_font)
        quality_label.setStyleSheet("color: #ffffff;")
        form_layout.addRow(quality_label, self.quality_spin)
        
        self.float_precision_combo = QComboBox()
        self.float_precision_combo.setFont(param_font)
        self.float_precision_combo.addItems(["4", "6", "8", "10", "12", "16"])
        self.float_precision_combo.setCurrentText("16")
        self.float_precision_combo.setStyleSheet("""
            QComboBox {
                background-color: #1a1a1a;
                color: white;
                border: 1px solid #404040;
                border-radius: 6px;
                padding: 8px;
                min-width: 100px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid white;
            }
        """)
        precision_label = QLabel("Float Precision:")
        precision_label.setFont(label_font)
        precision_label.setStyleSheet("color: #ffffff;")
        form_layout.addRow(precision_label, self.float_precision_combo)
        
        # Center the form in the main layout
        main_layout.addStretch()
        main_layout.addWidget(form_container, alignment=Qt.AlignCenter)
        main_layout.addStretch()
        
    def browse_input_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if directory:
            self.input_dir_edit.setText(directory)
            
    def browse_output_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_dir_edit.setText(directory)
            
    def get_config(self):
        return {
            'input_dir': self.input_dir_edit.text(),
            'output_dir': self.output_dir_edit.text(),
            'train_ratio': self.train_ratio_spin.value(),
            'seed': self.seed_spin.value(),
            'jpeg_quality': self.quality_spin.value(),
            'float_precision': int(self.float_precision_combo.currentText())
        }

class AugmentationSettingsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 20, 40, 20)
        main_layout.setSpacing(25)
        
        # Title
        title = QLabel("Augmentation Settings")
        title.setFont(QFont("Arial", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #ffffff; padding: 10px;")
        main_layout.addWidget(title)
        
        # Description
        desc = QLabel("Configure data augmentation techniques to enhance your dataset")
        desc.setFont(QFont("Arial", 11))
        desc.setAlignment(Qt.AlignCenter)
        desc.setStyleSheet("color: #b0b0b0; padding: 5px;")
        main_layout.addWidget(desc)
        
        # Centered Settings Container
        settings_container = QFrame()
        settings_container.setStyleSheet("""
            QFrame {
                background-color: #2c2c2c;
                border: 1px solid #404040;
                border-radius: 12px;
                padding: 30px;
            }
        """)
        settings_container.setMaximumWidth(600)
        
        settings_layout = QVBoxLayout(settings_container)
        settings_layout.setSpacing(20)
        
        # Font settings
        group_font = QFont("Arial", 12, QFont.Bold)
        checkbox_font = QFont("Arial", 10)
        spinbox_font = QFont("Arial", 10)
        
        # Flip Augmentations
        flip_group = QGroupBox("🔄 Flip Augmentations")
        flip_group.setFont(group_font)
        flip_group.setStyleSheet("""
            QGroupBox {
                color: #ffffff;
                background-color: #1a1a1a;
                border: 1px solid #404040;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        flip_layout = QVBoxLayout(flip_group)
        
        self.flip_horizontal_cb = QCheckBox("Horizontal Flip")
        self.flip_horizontal_cb.setFont(checkbox_font)
        self.flip_horizontal_cb.setStyleSheet("color: #ffffff;")
        self.flip_vertical_cb = QCheckBox("Vertical Flip")
        self.flip_vertical_cb.setFont(checkbox_font)
        self.flip_vertical_cb.setStyleSheet("color: #ffffff;")
        
        flip_layout.addWidget(self.flip_horizontal_cb)
        flip_layout.addWidget(self.flip_vertical_cb)
        settings_layout.addWidget(flip_group)
        
        # Brightness Augmentations
        brightness_group = QGroupBox("💡 Brightness Augmentations")
        brightness_group.setFont(group_font)
        brightness_group.setStyleSheet("""
            QGroupBox {
                color: #ffffff;
                background-color: #1a1a1a;
                border: 1px solid #404040;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        brightness_layout = QFormLayout(brightness_group)
        
        self.brightness_cb = QCheckBox("Enable Brightness Augmentation")
        self.brightness_cb.setFont(checkbox_font)
        self.brightness_cb.setStyleSheet("color: #ffffff;")
        self.brightness_spin = QDoubleSpinBox()
        self.brightness_spin.setFont(spinbox_font)
        self.brightness_spin.setRange(0, 100)
        self.brightness_spin.setValue(20)
        self.brightness_spin.setSuffix("%")
        self.brightness_spin.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #2c2c2c;
                color: white;
                border: 1px solid #404040;
                border-radius: 6px;
                padding: 6px;
            }
        """)
        
        brightness_layout.addRow(self.brightness_cb)
        brightness_layout.addRow(QLabel("Brightness Variation:"), self.brightness_spin)
        settings_layout.addWidget(brightness_group)
        
        # Saturation Augmentations
        saturation_group = QGroupBox("🎨 Saturation Augmentations")
        saturation_group.setFont(group_font)
        saturation_group.setStyleSheet("""
            QGroupBox {
                color: #ffffff;
                background-color: #1a1a1a;
                border: 1px solid #404040;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        saturation_layout = QFormLayout(saturation_group)
        
        self.saturation_cb = QCheckBox("Enable Saturation Augmentation")
        self.saturation_cb.setFont(checkbox_font)
        self.saturation_cb.setStyleSheet("color: #ffffff;")
        self.saturation_spin = QDoubleSpinBox()
        self.saturation_spin.setFont(spinbox_font)
        self.saturation_spin.setRange(0, 100)
        self.saturation_spin.setValue(20)
        self.saturation_spin.setSuffix("%")
        self.saturation_spin.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #2c2c2c;
                color: white;
                border: 1px solid #404040;
                border-radius: 6px;
                padding: 6px;
            }
        """)
        
        saturation_layout.addRow(self.saturation_cb)
        saturation_layout.addRow(QLabel("Saturation Variation:"), self.saturation_spin)
        settings_layout.addWidget(saturation_group)
        
        # Center the settings in the main layout
        main_layout.addStretch()
        main_layout.addWidget(settings_container, alignment=Qt.AlignCenter)
        main_layout.addStretch()
        
    def get_config(self):
        return {
            'flip_horizontal': self.flip_horizontal_cb.isChecked(),
            'flip_vertical': self.flip_vertical_cb.isChecked(),
            'brightness_pct': self.brightness_spin.value() if self.brightness_cb.isChecked() else 0,
            'saturation_pct': self.saturation_spin.value() if self.saturation_cb.isChecked() else 0
        }

class ProgressPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 20, 40, 20)
        main_layout.setSpacing(25)
        
        # Title
        title = QLabel("Augmentation Progress")
        title.setFont(QFont("Arial", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #ffffff; padding: 10px;")
        main_layout.addWidget(title)
        
        # Description
        desc = QLabel("Monitor the augmentation process in real-time")
        desc.setFont(QFont("Arial", 11))
        desc.setAlignment(Qt.AlignCenter)
        desc.setStyleSheet("color: #b0b0b0; padding: 5px;")
        main_layout.addWidget(desc)
        
        # Centered Progress Container
        progress_container = QFrame()
        progress_container.setStyleSheet("""
            QFrame {
                background-color: #2c2c2c;
                border: 1px solid #404040;
                border-radius: 12px;
                padding: 30px;
            }
        """)
        progress_container.setMaximumWidth(700)
        
        progress_layout = QVBoxLayout(progress_container)
        progress_layout.setSpacing(20)
        
        # Progress Bar
        progress_label = QLabel("Processing Progress:")
        progress_label.setFont(QFont("Arial", 12, QFont.Bold))
        progress_label.setStyleSheet("color: #ffffff;")
        progress_layout.addWidget(progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setFont(QFont("Arial", 10))
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #404040;
                border-radius: 8px;
                text-align: center;
                background-color: #1a1a1a;
                color: white;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #1abc9c;
                border-radius: 6px;
            }
        """)
        progress_layout.addWidget(self.progress_bar)
        
        # Log Area
        log_label = QLabel("Processing Log:")
        log_label.setFont(QFont("Arial", 12, QFont.Bold))
        log_label.setStyleSheet("color: #ffffff;")
        progress_layout.addWidget(log_label)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setStyleSheet("""
            QTextEdit {
                border: 2px solid #404040;
                border-radius: 8px;
                background-color: #1a1a1a;
                color: #00ff00;
                padding: 15px;
                min-height: 300px;
            }
        """)
        progress_layout.addWidget(self.log_text)
        
        # Center the progress container in the main layout
        main_layout.addStretch()
        main_layout.addWidget(progress_container, alignment=Qt.AlignCenter)
        main_layout.addStretch()
        
    def log_message(self, message):
        self.log_text.append(f"> {message}")
        
    def update_progress(self, value):
        self.progress_bar.setValue(value)

# -----------------------
# Main Wizard Window
# -----------------------
class AugmentationWizard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Vision Pro - Augmentation Suite")
        self.setGeometry(100, 100, 1200, 750)
        self.current_page = 0
        self.config = {}
        self.setup_ui()
        
    def setup_ui(self):
        # Set dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
                color: #ffffff;
            }
        """)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Sidebar
        self.sidebar = SidebarWidget()
        self.sidebar.setFixedWidth(250)
        self.sidebar.setStyleSheet("""
            QWidget {
                background-color: #2c2c2c;
                border-right: 1px solid #404040;
            }
        """)
        main_layout.addWidget(self.sidebar)
        
        # Main content area
        content_widget = QWidget()
        content_widget.setStyleSheet("QWidget { background-color: #1e1e1e; }")
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        
        # Create stacked widget for pages
        self.stacked_widget = QStackedWidget()
        content_layout.addWidget(self.stacked_widget)
        
        # Create pages
        self.dataset_page = DatasetSetupPage()
        self.augmentation_page = AugmentationSettingsPage()
        self.progress_page = ProgressPage()
        
        self.stacked_widget.addWidget(self.dataset_page)
        self.stacked_widget.addWidget(self.augmentation_page)
        self.stacked_widget.addWidget(self.progress_page)
        
        # Navigation buttons
        nav_frame = QFrame()
        nav_frame.setStyleSheet("""
            QFrame {
                background-color: #2c2c2c;
                border-top: 1px solid #404040;
                padding: 15px;
            }
        """)
        nav_layout = QHBoxLayout(nav_frame)
        
        self.back_btn = QPushButton("← Back")
        self.back_btn.setFont(QFont("Arial", 11))
        self.back_btn.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 6px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #7f8c8d;
            }
            QPushButton:disabled {
                background-color: #404040;
                color: #666666;
            }
        """)
        self.back_btn.clicked.connect(self.previous_page)
        
        self.next_btn = QPushButton("Next →")
        self.next_btn.setFont(QFont("Arial", 11))
        self.next_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 6px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        self.next_btn.clicked.connect(self.next_page)
        
        self.start_btn = QPushButton("Start Augmentation")
        self.start_btn.setFont(QFont("Arial", 11, QFont.Bold))
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 6px;
                min-width: 150px;
            }
            QPushButton:hover {
                background-color: #219a52;
            }
            QPushButton:disabled {
                background-color: #404040;
                color: #666666;
            }
        """)
        self.start_btn.clicked.connect(self.start_augmentation)
        self.start_btn.hide()
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setFont(QFont("Arial", 11))
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 6px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        self.cancel_btn.clicked.connect(self.close)
        
        nav_layout.addWidget(self.back_btn)
        nav_layout.addStretch()
        nav_layout.addWidget(self.cancel_btn)
        nav_layout.addWidget(self.start_btn)
        nav_layout.addWidget(self.next_btn)
        
        content_layout.addWidget(nav_frame)
        
        main_layout.addWidget(content_widget)
        
        # Connect sidebar buttons
        self.sidebar.dataset_btn.clicked.connect(lambda: self.set_page(0))
        self.sidebar.augmentation_btn.clicked.connect(lambda: self.set_page(1))
        
        # Initialize worker
        self.worker = None
        self.update_navigation()
        
    def set_page(self, page_index):
        self.current_page = page_index
        self.stacked_widget.setCurrentIndex(page_index)
        self.update_navigation()
        
    def update_navigation(self):
        self.back_btn.setVisible(self.current_page > 0)
        
        # Update sidebar button states
        self.sidebar.dataset_btn.setStyleSheet(self.get_sidebar_button_style(0))
        self.sidebar.augmentation_btn.setStyleSheet(self.get_sidebar_button_style(1))
        
        if self.current_page == 0:  # Dataset setup
            self.next_btn.setVisible(True)
            self.start_btn.setVisible(False)
            self.next_btn.setText("Next →")
        elif self.current_page == 1:  # Augmentation settings
            self.next_btn.setVisible(True)
            self.start_btn.setVisible(False)
            self.next_btn.setText("Next →")
        else:  # Progress page
            self.next_btn.setVisible(False)
            self.start_btn.setVisible(True)
            
    def get_sidebar_button_style(self, page_index):
        if page_index == self.current_page:
            return """
                QPushButton {
                    background-color: #1abc9c;
                    color: white;
                    padding: 12px;
                    border: none;
                    border-radius: 8px;
                    text-align: left;
                    padding-left: 15px;
                }
            """
        else:
            return """
                QPushButton {
                    background-color: #2c3e50;
                    color: white;
                    padding: 12px;
                    border: none;
                    border-radius: 8px;
                    text-align: left;
                    padding-left: 15px;
                }
                QPushButton:hover {
                    background-color: #34495e;
                }
            """
            
    def next_page(self):
        if self.current_page == 0:
            # Validate dataset setup
            if not self.validate_dataset_setup():
                return
            self.config.update(self.dataset_page.get_config())
            self.current_page = 1
        elif self.current_page == 1:
            # Get augmentation settings
            self.config.update(self.augmentation_page.get_config())
            self.current_page = 2
            
        self.stacked_widget.setCurrentIndex(self.current_page)
        self.update_navigation()
        
    def previous_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.stacked_widget.setCurrentIndex(self.current_page)
            self.update_navigation()
        
    def validate_dataset_setup(self):
        config = self.dataset_page.get_config()
        
        if not config['input_dir']:
            QMessageBox.warning(self, "Validation Error", "Please select input directory")
            return False
            
        if not config['output_dir']:
            QMessageBox.warning(self, "Validation Error", "Please select output directory")
            return False
            
        input_dir = Path(config['input_dir'])
        if not input_dir.exists():
            QMessageBox.warning(self, "Validation Error", "Input directory does not exist")
            return False
            
        return True
        
    def start_augmentation(self):
        # Disable UI
        self.start_btn.setEnabled(False)
        self.back_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        
        # Start worker thread
        self.worker = AugmentationWorker(self.config)
        self.worker.progress.connect(self.progress_page.update_progress)
        self.worker.message.connect(self.progress_page.log_message)
        self.worker.finished_signal.connect(self.augmentation_finished)
        self.worker.error.connect(self.augmentation_error)
        self.worker.start()
        
    def cancel_augmentation(self):
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
            self.progress_page.log_message("Augmentation cancelled by user")
        
        self.start_btn.setEnabled(True)
        self.back_btn.setEnabled(True)
        
    def augmentation_finished(self, message):
        self.progress_page.log_message(message)
        self.start_btn.setEnabled(True)
        self.back_btn.setEnabled(True)
        QMessageBox.information(self, "Success", "Augmentation completed successfully!")
        
    def augmentation_error(self, error_message):
        self.progress_page.log_message(f"ERROR: {error_message}")
        self.start_btn.setEnabled(True)
        self.back_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Augmentation failed:\n{error_message}")

# -----------------------
# Integration Function with Threading
# -----------------------
def launch_augmentation_tool(input_dir=None, output_dir=None):
    """
    Call this function from your main annotation tool when the augmentation button is clicked
    Launches augmentation tool in a separate thread
    """
    def run_tool():
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        # Set dark theme
        app.setStyle('Fusion')
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(30, 30, 30))
        palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(35, 35, 35))
        palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
        palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
        palette.setColor(QPalette.Text, QColor(255, 255, 255))
        palette.setColor(QPalette.Button, QColor(50, 50, 50))
        palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        app.setPalette(palette)
        
        window = AugmentationWizard()
        
        # Pre-fill directories if provided
        if input_dir and hasattr(window.dataset_page, 'input_dir_edit'):
            window.dataset_page.input_dir_edit.setText(input_dir)
        if output_dir and hasattr(window.dataset_page, 'output_dir_edit'):
            window.dataset_page.output_dir_edit.setText(output_dir)
        
        window.show()
        app.exec_()
    
    # Launch in separate thread
    thread = threading.Thread(target=run_tool, daemon=True)
    thread.start()
    
    return thread

# -----------------------
# Standalone Usage
# -----------------------
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    
    # Set dark theme for standalone usage
    app.setStyle('Fusion')
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(35, 35, 35))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
    palette.setColor(QPalette.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.Button, QColor(50, 50, 50))
    palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)
    
    window = AugmentationWizard()
    window.show()
    
    sys.exit(app.exec_())