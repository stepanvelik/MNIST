"""Загрузка и чтение оригинальных IDX-файлов MNIST."""

import gzip
from pathlib import Path
import struct
from urllib.request import urlopen

import numpy as np

BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist"
FILES = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
    "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
}


def _download(filename: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urlopen(f"{BASE_URL}/{filename}", timeout=60) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as file:
        data = file.read()

    if len(data) < 4:
        raise ValueError(f"Повреждённый IDX-файл: {path}")
    magic = struct.unpack_from(">I", data)[0]
    data_type = magic >> 8 & 0xFF
    dimensions = magic & 0xFF
    if data_type != 0x08 or dimensions not in (1, 3):
        raise ValueError(f"Неожиданный формат IDX-файла: {path}")

    shape = struct.unpack_from(f">{dimensions}I", data, offset=4)
    header_size = 4 + 4 * dimensions
    array = np.frombuffer(data, dtype=np.uint8, offset=header_size)
    if array.size != int(np.prod(shape)):
        raise ValueError(f"Неполный IDX-файл: {path}")
    return array.reshape(shape)


def load_mnist(
    split: str,
    data_dir: str | Path = "data",
    limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает изображения (N, 784) в диапазоне 0–1 и метки (N,)."""
    if split not in FILES:
        raise ValueError("split должен быть 'train' или 'test'")
    if limit is not None and limit < 1:
        raise ValueError("limit должен быть положительным числом")

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    filenames = FILES[split]
    paths = [data_dir / filename for filename in filenames]
    for filename, path in zip(filenames, paths):
        if not path.exists():
            print(f"Загрузка {filename}...")
            _download(filename, path)

    images = _read_idx(paths[0])
    labels = _read_idx(paths[1])
    if len(images) != len(labels):
        raise ValueError("Количество изображений и меток не совпадает")

    if limit is not None:
        images = images[:limit]
        labels = labels[:limit]
    images = images.reshape(len(images), -1).astype(np.float32) / 255.0
    return images, labels
