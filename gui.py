# PyQt5 GUI for pseudo-colorization

import sys
import os

# Fix path - we're at root now
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch

# Check PyQt5
try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QSlider, QCheckBox, QFileDialog, QMessageBox, QProgressBar
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5.QtGui import QImage, QPixmap
except ImportError as e:
    print("ERROR: PyQt5 not installed!")
    print("Install with: pip install PyQt5")
    sys.exit(1)

from src.utils import load_image, rgb_to_lab, save_image
# Import from root-level inference.py
from inference import load_model, colorize_image


class InferenceThread(QThread):
    finished = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)

    def __init__(self, L, model, device, bias_strength, use_color_balance, temperature):
        super().__init__()
        self.L = L
        self.model = model
        self.device = device
        self.bias_strength = bias_strength
        self.use_color_balance = use_color_balance
        self.temperature = temperature

    def run(self):
        try:
            rgb_output = colorize_image(
                self.L, self.model, self.device,
                bias_strength=self.bias_strength,
                use_color_balance=self.use_color_balance,
                temperature=self.temperature
            )
            self.finished.emit(rgb_output)
        except Exception as e:
            self.error.emit(str(e))


class ColorizationGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pseudo-Colorization GUI")
        self.setGeometry(100, 100, 1200, 700)

        # Store project root for path resolution
        self.project_root = project_root

        self.current_image = None
        self.current_L = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.inference_thread = None
        self.current_output = None

        self.init_ui()

        # Try to load default model (use absolute path)
        default_model = os.path.join(self.project_root, "models", "colorization_final.pth")
        if os.path.exists(default_model):
            try:
                self.load_model(default_model)
            except:
                pass

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Top buttons
        top_layout = QHBoxLayout()
        self.btn_load_image = QPushButton("Load Image")
        self.btn_load_image.clicked.connect(self.load_image)
        self.btn_load_model = QPushButton("Load Model")
        self.btn_load_model.clicked.connect(self.load_model_dialog)
        self.btn_run = QPushButton("Run Colorization")
        self.btn_run.clicked.connect(self.run_colorization)
        self.btn_run.setEnabled(False)
        self.btn_save = QPushButton("Save Output")
        self.btn_save.clicked.connect(self.save_output)
        self.btn_save.setEnabled(False)

        top_layout.addWidget(self.btn_load_image)
        top_layout.addWidget(self.btn_load_model)
        top_layout.addWidget(self.btn_run)
        top_layout.addWidget(self.btn_save)
        top_layout.addStretch()
        main_layout.addLayout(top_layout)

        # Images side by side
        image_layout = QHBoxLayout()

        # Input image
        input_group = QVBoxLayout()
        input_label = QLabel("Input (Grayscale)")
        input_label.setAlignment(Qt.AlignCenter)
        self.label_input = QLabel()
        self.label_input.setMinimumSize(512, 256)
        self.label_input.setAlignment(Qt.AlignCenter)
        self.label_input.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        self.label_input.setText("No image loaded")
        input_group.addWidget(input_label)
        input_group.addWidget(self.label_input)
        image_layout.addLayout(input_group)

        # Output image
        output_group = QVBoxLayout()
        output_label = QLabel("Output (Colorized)")
        output_label.setAlignment(Qt.AlignCenter)
        self.label_output = QLabel()
        self.label_output.setMinimumSize(512, 256)
        self.label_output.setAlignment(Qt.AlignCenter)
        self.label_output.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        self.label_output.setText("No output yet")
        output_group.addWidget(output_label)
        output_group.addWidget(self.label_output)
        image_layout.addLayout(output_group)

        main_layout.addLayout(image_layout)

        # Controls
        controls_layout = QVBoxLayout()

        # Bias slider
        bias_layout = QHBoxLayout()
        bias_label = QLabel("Bias Map Strength:")
        self.slider_bias = QSlider(Qt.Horizontal)
        self.slider_bias.setMinimum(0)
        self.slider_bias.setMaximum(100)
        self.slider_bias.setValue(100)
        self.slider_bias.valueChanged.connect(self.on_controls_changed)
        self.label_bias_value = QLabel("1.00")
        self.label_bias_value.setMinimumWidth(50)

        bias_layout.addWidget(bias_label)
        bias_layout.addWidget(self.slider_bias)
        bias_layout.addWidget(self.label_bias_value)
        controls_layout.addLayout(bias_layout)

        # Color balance checkbox
        self.checkbox_color_balance = QCheckBox("Use Color Balance Correction")
        self.checkbox_color_balance.setChecked(True)
        self.checkbox_color_balance.stateChanged.connect(self.on_controls_changed)
        controls_layout.addWidget(self.checkbox_color_balance)

        # Temperature slider (original feature)
        temp_layout = QHBoxLayout()
        temp_label = QLabel("Color Temperature:")
        self.slider_temperature = QSlider(Qt.Horizontal)
        self.slider_temperature.setMinimum(-100)
        self.slider_temperature.setMaximum(100)
        self.slider_temperature.setValue(0)
        self.slider_temperature.valueChanged.connect(self.on_controls_changed)
        self.label_temp_value = QLabel("0.00 (neutral)")
        self.label_temp_value.setMinimumWidth(100)

        temp_layout.addWidget(temp_label)
        temp_layout.addWidget(self.slider_temperature)
        temp_layout.addWidget(self.label_temp_value)
        controls_layout.addLayout(temp_layout)

        # Status
        self.label_status = QLabel("Ready. Load an image and model to start.")
        controls_layout.addWidget(self.label_status)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        controls_layout.addWidget(self.progress_bar)

        main_layout.addLayout(controls_layout)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )

        if not file_path:
            return

        try:
            img = load_image(file_path, target_size=(256, 256))
            self.current_image = img
            L, _, _ = rgb_to_lab(img)
            self.current_L = L

            self.display_image(L, self.label_input, grayscale=True)

            if self.model is not None:
                self.btn_run.setEnabled(True)

            self.label_status.setText(f"Image loaded: {os.path.basename(file_path)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image:\n{str(e)}")

    def load_model_dialog(self):
        models_dir = os.path.join(self.project_root, "models")
        if not os.path.exists(models_dir):
            models_dir = self.project_root
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Model", models_dir, "Model Files (*.pth)"
        )
        if file_path:
            self.load_model(file_path)

    def load_model(self, model_path):
        try:
            self.model = load_model(model_path, self.device)
            self.label_status.setText(f"Model loaded: {os.path.basename(model_path)} (Device: {self.device})")
            if self.current_L is not None:
                self.btn_run.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model:\n{str(e)}")

    def on_controls_changed(self):
        bias_value = self.slider_bias.value() / 100.0
        self.label_bias_value.setText(f"{bias_value:.2f}")

        temp_value = self.slider_temperature.value() / 100.0
        temp_text = "neutral" if temp_value == 0 else ("warmer" if temp_value > 0 else "cooler")
        self.label_temp_value.setText(f"{temp_value:.2f} ({temp_text})")

        if self.current_output is not None and self.model is not None and self.current_L is not None:
            self.run_colorization()

    def run_colorization(self):
        if self.model is None:
            QMessageBox.warning(self, "Warning", "Please load a model first.")
            return

        if self.current_L is None:
            QMessageBox.warning(self, "Warning", "Please load an image first.")
            return

        self.btn_run.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.label_status.setText("Running colorization...")

        bias_strength = self.slider_bias.value() / 100.0
        use_color_balance = self.checkbox_color_balance.isChecked()
        temperature = self.slider_temperature.value() / 100.0

        self.inference_thread = InferenceThread(
            self.current_L, self.model, self.device,
            bias_strength, use_color_balance, temperature
        )
        self.inference_thread.finished.connect(self.on_inference_finished)
        self.inference_thread.error.connect(self.on_inference_error)
        self.inference_thread.start()

    def on_inference_finished(self, rgb_output):
        self.current_output = rgb_output
        self.display_image(rgb_output, self.label_output, grayscale=False)
        self.btn_run.setEnabled(True)
        self.btn_save.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.label_status.setText("Colorization complete!")

    def on_inference_error(self, error_msg):
        QMessageBox.critical(self, "Error", f"Inference failed:\n{error_msg}")
        self.btn_run.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.label_status.setText("Error during colorization.")

    def display_image(self, image, label, grayscale=False):
        # Display numpy image in QLabel with proper error handling
        try:
            if grayscale:
                if image.dtype != np.uint8:
                    img_display = (np.clip(image, 0, 100) / 100.0 * 255.0).astype(np.uint8)
                else:
                    img_display = image
                height, width = img_display.shape
                q_image = QImage(img_display.data, width, height, width, QImage.Format_Grayscale8)
            else:
                if image.dtype != np.uint8:
                    img_display = np.clip(image, 0, 255).astype(np.uint8)
                else:
                    img_display = image
                height, width, channels = img_display.shape
                bytes_per_line = channels * width
                q_image = QImage(img_display.data, width, height, bytes_per_line, QImage.Format_RGB888)

            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(scaled_pixmap)
        except Exception as e:
            label.setText(f"Display error: {str(e)}")

    def save_output(self):
        if self.current_output is None:
            QMessageBox.warning(self, "Warning", "No output to save.")
            return

        output_dir = os.path.join(self.project_root, "output")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", output_dir, "Image Files (*.png *.jpg *.jpeg)"
        )

        if file_path:
            try:
                save_image(self.current_output, file_path)
                self.label_status.setText(f"Image saved to {os.path.basename(file_path)}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save image:\n{str(e)}")


def main():
    try:
        app = QApplication(sys.argv)
        window = ColorizationGUI()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"ERROR starting GUI: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
        sys.exit(1)


if __name__ == "__main__":
    main()
