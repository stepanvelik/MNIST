"""Преобразование пользовательской картинки к формату MNIST 28×28."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, ImageStat


def load_digit_image(path: str | Path, smoothing: bool = True) -> np.ndarray:
    """Загружает картинку и возвращает нормализованное изображение формы (784,)."""
    with Image.open(path) as source:
        return prepare_digit_image(source, smoothing)


def prepare_digit_image(source: Image.Image, smoothing: bool = True) -> np.ndarray:
    """Преобразует PIL-изображение к формату MNIST 28×28."""
    if source.mode in ("RGBA", "LA") or "transparency" in source.info:
        rgba = source.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        source = Image.alpha_composite(background, rgba)
    image = source.convert("L")

    # В MNIST цифра светлая, а фон тёмный. Для обычного фото выполняем инверсию.
    if ImageStat.Stat(image).mean[0] > 127:
        image = ImageOps.invert(image)

    foreground = image.point(lambda pixel: 255 if pixel > 30 else 0)
    bounds = foreground.getbbox()
    if bounds is None:
        raise ValueError("Сначала нарисуйте цифру")
    image = image.crop(bounds)

    # Сохраняем пропорции: большая сторона цифры занимает 20 пикселей, как в MNIST.
    scale = 20 / max(image.size)
    size = tuple(max(1, round(side * scale)) for side in image.size)
    resampling = Image.Resampling.LANCZOS if smoothing else Image.Resampling.NEAREST
    image = image.resize(size, resampling)

    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(image, ((28 - image.width) // 2, (28 - image.height) // 2))
    return np.asarray(canvas, dtype=np.float32).reshape(-1) / 255.0
