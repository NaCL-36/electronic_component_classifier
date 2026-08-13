from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

import joblib
import numpy as np
from PIL import Image, ImageOps, ImageTk


APP_NAME = "SMARTLITE"
BASE_DIR = Path(__file__).resolve().parent
KERAS_MODEL_PATH = BASE_DIR / "models" / "mlp_model.keras"
SKLEARN_MODEL_PATH = BASE_DIR / "models" / "mlp_model.joblib"
KERAS_METADATA_PATH = BASE_DIR / "models" / "mlp_model_metadata.npz"
IMAGE_SIZE = (64, 64)
RESAMPLE_FILTER = getattr(Image, "Resampling", Image).LANCZOS
FALLBACK_CLASSES = [
    "battery holder",
    "breadboard",
    "capacitor",
    "connector",
    "crystal oscillator",
    "diode",
    "fuse",
    "heat sink",
    "ic chip",
    "inductor coil",
    "jumper wire",
    "led",
    "microcontroller board",
    "potentiometer",
    "power supply module",
    "relay",
    "resistor",
    "switch",
    "transformer",
    "transistor",
]


def preprocess_image(image_path, image_size):
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image).convert("L")
    image = image.resize(image_size, RESAMPLE_FILTER)
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    return pixels.reshape(1, -1)


def load_model_bundle():
    if KERAS_METADATA_PATH.exists():
        metadata = np.load(KERAS_METADATA_PATH, allow_pickle=True)
        classes = metadata["classes"].tolist()
        image_size = tuple(int(v) for v in metadata["IMAGE_SIZE"])
        accuracy = float(metadata["validation_accuracy"])

        if KERAS_MODEL_PATH.exists():
            try:
                from tensorflow import keras

                return {
                    "model": keras.models.load_model(KERAS_MODEL_PATH),
                    "classes": classes,
                    "image_size": image_size,
                    "validation_accuracy": accuracy,
                    "source": KERAS_MODEL_PATH.name,
                    "type": "keras",
                }
            except Exception:
                pass

        if SKLEARN_MODEL_PATH.exists():
            try:
                return {
                    "model": joblib.load(SKLEARN_MODEL_PATH),
                    "classes": classes,
                    "image_size": image_size,
                    "validation_accuracy": accuracy,
                    "source": SKLEARN_MODEL_PATH.name,
                    "type": "sklearn",
                }
            except Exception:
                pass

    return None


class SmartLiteApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"{APP_NAME} - Electronic Components Classifier")
        self.geometry("1120x720")
        self.minsize(980, 620)

        self.model_bundle = load_model_bundle()
        self.selected_image = None
        self.preview_photo = None

        self.colors = {
            "bg": "#eef2f6",
            "panel": "#ffffff",
            "panel_alt": "#f8fafc",
            "ink": "#172033",
            "muted": "#667085",
            "soft": "#98a2b3",
            "accent": "#0e7c66",
            "accent_dark": "#075e4d",
            "accent_soft": "#dff6ef",
            "line": "#d7dde7",
            "warn": "#b54708",
            "warn_soft": "#fff4df",
            "error": "#b42318",
            "error_soft": "#fee4e2",
            "shadow": "#d5dce7",
        }

        self.configure(bg=self.colors["bg"])
        self._configure_style()
        self._build_layout()
        self._refresh_model_status()

    def _configure_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background=self.colors["bg"])
        style.configure("Panel.TFrame", background=self.colors["panel"])
        style.configure("AltPanel.TFrame", background=self.colors["panel_alt"])
        style.configure(
            "Title.TLabel",
            background=self.colors["bg"],
            foreground=self.colors["ink"],
            font=("Segoe UI", 26, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background=self.colors["bg"],
            foreground=self.colors["muted"],
            font=("Segoe UI", 11),
        )
        style.configure(
            "PanelTitle.TLabel",
            background=self.colors["panel"],
            foreground=self.colors["ink"],
            font=("Segoe UI", 15, "bold"),
        )
        style.configure(
            "Body.TLabel",
            background=self.colors["panel"],
            foreground=self.colors["muted"],
            font=("Segoe UI", 10),
        )
        style.configure(
            "Small.TLabel",
            background=self.colors["panel"],
            foreground=self.colors["soft"],
            font=("Segoe UI", 9),
        )
        style.configure(
            "Result.TLabel",
            background=self.colors["panel"],
            foreground=self.colors["ink"],
            font=("Segoe UI", 24, "bold"),
        )
        style.configure(
            "Accent.TButton",
            background=self.colors["accent"],
            foreground="#ffffff",
            borderwidth=0,
            focusthickness=0,
            padding=(20, 11),
            font=("Segoe UI", 10, "bold"),
        )
        style.map(
            "Accent.TButton",
            background=[("active", self.colors["accent_dark"]), ("disabled", "#9ca3af")],
            foreground=[("disabled", "#ffffff")],
        )
        style.configure(
            "Ghost.TButton",
            background=self.colors["panel_alt"],
            foreground=self.colors["ink"],
            borderwidth=1,
            relief="solid",
            padding=(18, 10),
            font=("Segoe UI", 10, "bold"),
        )
        style.map(
            "Ghost.TButton",
            background=[("active", "#edf2f7"), ("disabled", "#f2f4f7")],
        )
        style.configure("TProgressbar", troughcolor="#e8edf3", background=self.colors["accent"])

    def _build_layout(self):
        header = ttk.Frame(self, padding=(32, 24, 32, 10))
        header.pack(fill="x")

        header.columnconfigure(0, weight=1)
        header.columnconfigure(1, weight=0)

        title_area = ttk.Frame(header)
        title_area.grid(row=0, column=0, sticky="ew")
        ttk.Label(title_area, text=APP_NAME, style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            title_area,
            text="Electronic component image classifier powered by the saved MLP model.",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        self.model_chip = tk.Label(
            header,
            text="Checking model",
            bg=self.colors["warn_soft"],
            fg=self.colors["warn"],
            padx=14,
            pady=8,
            font=("Segoe UI", 9, "bold"),
        )
        self.model_chip.grid(row=0, column=1, sticky="ne", padx=(16, 0), pady=(4, 0))

        main = ttk.Frame(self, padding=(32, 14, 32, 32))
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=7)
        main.columnconfigure(1, weight=4)
        main.rowconfigure(0, weight=1)

        left = self._panel(main, padding=22)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        image_header = ttk.Frame(left, style="Panel.TFrame")
        image_header.grid(row=0, column=0, sticky="ew")
        image_header.columnconfigure(0, weight=1)
        ttk.Label(image_header, text="Image Preview", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            image_header,
            text="JPG, PNG, WEBP, BMP",
            style="Small.TLabel",
        ).grid(row=0, column=1, sticky="e")

        self.preview_label = tk.Label(
            left,
            text="Choose an image to start",
            bg=self.colors["panel_alt"],
            fg=self.colors["muted"],
            bd=0,
            relief="flat",
            highlightthickness=1,
            highlightbackground=self.colors["line"],
            font=("Segoe UI", 14, "bold"),
            compound="center",
        )
        self.preview_label.grid(row=1, column=0, sticky="nsew", pady=(18, 18))

        controls = ttk.Frame(left, style="Panel.TFrame")
        controls.grid(row=2, column=0, sticky="ew")
        controls.columnconfigure(0, weight=0)
        controls.columnconfigure(1, weight=0)
        controls.columnconfigure(2, weight=1)

        ttk.Button(controls, text="Open Image", style="Ghost.TButton", command=self.choose_image).grid(
            row=0, column=0, sticky="w"
        )
        self.predict_button = ttk.Button(
            controls,
            text="Classify",
            style="Accent.TButton",
            command=self.classify_image,
            state="disabled",
        )
        self.predict_button.grid(row=0, column=1, sticky="w", padx=(12, 0))
        self.file_label = ttk.Label(controls, text="No image selected", style="Body.TLabel")
        self.file_label.grid(row=0, column=2, sticky="e", padx=(12, 0))

        right = self._panel(main, padding=22)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(7, weight=1)

        ttk.Label(right, text="Prediction", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w")
        self.status_label = ttk.Label(right, text="", style="Body.TLabel", wraplength=330)
        self.status_label.grid(row=1, column=0, sticky="ew", pady=(10, 20))

        self.result_label = ttk.Label(right, text="Waiting for image", style="Result.TLabel", wraplength=330)
        self.result_label.grid(row=2, column=0, sticky="w")

        self.confidence_label = ttk.Label(right, text="", style="Body.TLabel")
        self.confidence_label.grid(row=3, column=0, sticky="w", pady=(8, 12))

        self.confidence_bar = ttk.Progressbar(right, value=0, maximum=100)
        self.confidence_bar.grid(row=4, column=0, sticky="ew", pady=(0, 18))

        self.probability_frame = ttk.Frame(right, style="Panel.TFrame")
        self.probability_frame.grid(row=5, column=0, sticky="ew")
        self.probability_frame.columnconfigure(1, weight=1)

        ttk.Separator(right).grid(row=6, column=0, sticky="ew", pady=24)

        ttk.Label(right, text="Supported Classes", style="PanelTitle.TLabel").grid(row=7, column=0, sticky="sw")
        class_text = ", ".join(self._classes())
        ttk.Label(right, text=class_text, style="Body.TLabel", wraplength=330).grid(
            row=8, column=0, sticky="ew", pady=(10, 0)
        )

    def _panel(self, parent, padding):
        return ttk.Frame(parent, style="Panel.TFrame", padding=padding)

    def _classes(self):
        if self.model_bundle:
            return list(self.model_bundle.get("classes", FALLBACK_CLASSES))
        return FALLBACK_CLASSES

    def _image_size(self):
        if self.model_bundle:
            size = self.model_bundle.get("image_size", IMAGE_SIZE)
            return tuple(int(v) for v in size)
        return IMAGE_SIZE

    def _refresh_model_status(self):
        if not self.model_bundle:
            self.model_chip.configure(
                text="Model missing",
                bg=self.colors["error_soft"],
                fg=self.colors["error"],
            )
            self.status_label.configure(
                text=(
                    "Model not found. Run `python notebooks\\MLP_Training.py` first, then reopen the app."
                ),
                foreground=self.colors["error"],
            )
            return

        accuracy = self.model_bundle.get("validation_accuracy")
        if accuracy is None:
            text = f"Loaded model: {self.model_bundle.get('source', KERAS_MODEL_PATH.name)}"
        else:
            text = (
                f"Loaded model: {self.model_bundle.get('source', KERAS_MODEL_PATH.name)}"
                f" | Validation accuracy: {accuracy:.2%}"
            )
        self.model_chip.configure(
            text="Model ready",
            bg=self.colors["accent_soft"],
            fg=self.colors["accent_dark"],
        )
        self.status_label.configure(text=text, foreground=self.colors["muted"])

    def choose_image(self):
        path = filedialog.askopenfilename(
            title="Choose component image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.webp *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        try:
            self._show_preview(Path(path))
        except Exception as exc:
            messagebox.showerror(APP_NAME, f"Could not open this image:\n{exc}")
            return

        self.selected_image = Path(path)
        self.file_label.configure(text=self.selected_image.name)
        self.result_label.configure(text="Ready to classify")
        self.confidence_label.configure(text="")
        self.confidence_bar.configure(value=0)
        self._clear_probabilities()

        if self.model_bundle:
            self.predict_button.configure(state="normal")

    def _show_preview(self, image_path):
        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image)
        image.thumbnail((650, 470), RESAMPLE_FILTER)
        self.preview_photo = ImageTk.PhotoImage(image)
        self.preview_label.configure(image=self.preview_photo, text="")

    def classify_image(self):
        if not self.model_bundle:
            messagebox.showerror(APP_NAME, "Model not found. Run `python notebooks\\MLP_Training.py` first.")
            return
        if not self.selected_image:
            messagebox.showinfo(APP_NAME, "Please choose an image first.")
            return

        try:
            features = preprocess_image(self.selected_image, self._image_size())
            model = self.model_bundle["model"]
            classes = self._classes()

            if self.model_bundle.get("type") == "keras":
                probabilities = model.predict(features, verbose=0)[0]
            else:
                probabilities = model.predict_proba(features)[0]
            top_indices = np.argsort(probabilities)[::-1][:3]
            best_index = int(top_indices[0])
            confidence = float(probabilities[best_index])

            self.result_label.configure(text=classes[best_index])
            self.confidence_label.configure(text=f"Confidence: {confidence:.2%}")
            self.confidence_bar.configure(value=confidence * 100)
            self._render_probabilities(classes, probabilities, top_indices)
        except Exception as exc:
            messagebox.showerror(APP_NAME, f"Could not classify this image:\n{exc}")

    def _clear_probabilities(self):
        for child in self.probability_frame.winfo_children():
            child.destroy()

    def _render_probabilities(self, classes, probabilities, top_indices):
        self._clear_probabilities()
        for row, index in enumerate(top_indices):
            name = classes[int(index)]
            score = float(probabilities[int(index)])

            ttk.Label(self.probability_frame, text=name, style="Body.TLabel").grid(
                row=row, column=0, sticky="w", pady=5
            )
            bar = ttk.Progressbar(self.probability_frame, value=score * 100, maximum=100)
            bar.grid(row=row, column=1, sticky="ew", padx=10, pady=5)
            ttk.Label(self.probability_frame, text=f"{score:.1%}", style="Body.TLabel").grid(
                row=row, column=2, sticky="e", pady=5
            )


if __name__ == "__main__":
    app = SmartLiteApp()
    app.mainloop()
