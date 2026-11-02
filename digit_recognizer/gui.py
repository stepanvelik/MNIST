"""Окно для рисования и распознавания цифр."""

import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw

from .image import prepare_digit_image
from .knn import KNNClassifier


class DigitDrawingApp:
    CANVAS_SIZE = 280
    BRUSH_RADIUS = 11

    def __init__(self, classifier: KNNClassifier) -> None:
        self.classifier = classifier
        self.root = tk.Tk()
        self.root.title("Распознавание цифры — MNIST")
        self.root.resizable(False, False)

        self.image = Image.new("L", (self.CANVAS_SIZE, self.CANVAS_SIZE), 255)
        self.image_draw = ImageDraw.Draw(self.image)
        self.last_position: tuple[int, int] | None = None
        self.smoothing = tk.BooleanVar(value=True)

        self._build_interface()
        self.clear()

    def _build_interface(self) -> None:
        container = ttk.Frame(self.root, padding=16)
        container.grid()

        ttk.Label(container, text="Нарисуйте одну цифру", font=("Segoe UI", 12, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(0, 10)
        )
        self.canvas = tk.Canvas(
            container,
            width=self.CANVAS_SIZE,
            height=self.CANVAS_SIZE,
            background="white",
            highlightthickness=1,
            highlightbackground="#777777",
            cursor="crosshair",
        )
        self.canvas.grid(row=1, column=0, columnspan=2)
        self.canvas.bind("<ButtonPress-1>", self.start_stroke)
        self.canvas.bind("<B1-Motion>", self.continue_stroke)
        self.canvas.bind("<ButtonRelease-1>", self.end_stroke)

        ttk.Checkbutton(container, text="Сглаживание", variable=self.smoothing).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(10, 4)
        )
        self.result_label = ttk.Label(container, font=("Segoe UI", 16, "bold"))
        self.result_label.grid(row=3, column=0, columnspan=2, pady=(8, 2))
        self.confidence_label = ttk.Label(container, font=("Segoe UI", 11))
        self.confidence_label.grid(row=4, column=0, columnspan=2, pady=(0, 12))

        recognize_button = ttk.Button(container, text="Распознать", command=self.recognize)
        recognize_button.grid(row=5, column=0, sticky="ew", padx=(0, 4))
        clear_button = ttk.Button(container, text="Очистить", command=self.clear)
        clear_button.grid(row=5, column=1, sticky="ew", padx=(4, 0))
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)

    def start_stroke(self, event: tk.Event) -> None:
        self.last_position = (event.x, event.y)
        self._draw_point(event.x, event.y)

    def continue_stroke(self, event: tk.Event) -> None:
        if self.last_position is None:
            self.last_position = (event.x, event.y)
        previous = self.last_position
        current = (event.x, event.y)
        self.canvas.create_line(
            *previous,
            *current,
            fill="black",
            width=self.BRUSH_RADIUS * 2,
            capstyle=tk.ROUND,
            smooth=self.smoothing.get(),
        )
        self.image_draw.line((*previous, *current), fill=0, width=self.BRUSH_RADIUS * 2)
        self._draw_point(event.x, event.y)
        self.last_position = current

    def end_stroke(self, _event: tk.Event) -> None:
        self.last_position = None

    def _draw_point(self, x: int, y: int) -> None:
        radius = self.BRUSH_RADIUS
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill="black", outline="black")
        self.image_draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=0)

    def recognize(self) -> None:
        try:
            image = prepare_digit_image(self.image, smoothing=self.smoothing.get())
            probabilities = self.classifier.predict_proba(image)[0]
        except ValueError as error:
            messagebox.showinfo("Нет рисунка", str(error), parent=self.root)
            return

        digit = int(np.argmax(probabilities))
        confidence = float(probabilities[digit]) * 100
        self.result_label.config(text=f"Это цифра {digit}")
        self.confidence_label.config(text=f"Уверенность: {confidence:.1f}%")

    def clear(self) -> None:
        self.canvas.delete("all")
        self.image_draw.rectangle((0, 0, self.CANVAS_SIZE, self.CANVAS_SIZE), fill=255)
        self.last_position = None
        self.result_label.config(text="Цифра: —")
        self.confidence_label.config(text="Уверенность: —")

    def run(self) -> None:
        self.root.mainloop()
