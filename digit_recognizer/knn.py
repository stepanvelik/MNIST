"""Классификатор k-ближайших соседей для изображений цифр."""

from collections.abc import Iterator

import numpy as np


class KNNClassifier:
    """Ищет k самых похожих изображений и выбирает цифру голосованием."""

    def __init__(self, k: int = 3, batch_size: int = 128) -> None:
        if k < 1:
            raise ValueError("k должно быть положительным числом")
        if batch_size < 1:
            raise ValueError("batch_size должен быть положительным числом")
        self.k = k
        self.batch_size = batch_size
        self._images: np.ndarray | None = None
        self._labels: np.ndarray | None = None
        self._squared_norms: np.ndarray | None = None

    def fit(self, images: np.ndarray, labels: np.ndarray) -> "KNNClassifier":
        images = np.asarray(images, dtype=np.float32)
        labels = np.asarray(labels)

        if images.ndim != 2:
            raise ValueError("Изображения должны иметь форму (количество, 784)")
        if labels.ndim != 1 or len(labels) != len(images):
            raise ValueError("Каждому изображению должна соответствовать одна метка")
        if len(images) == 0:
            raise ValueError("Обучающая выборка не должна быть пустой")
        if self.k > len(images):
            raise ValueError("k не может быть больше размера обучающей выборки")

        self._images = images
        self._labels = labels
        self._squared_norms = np.sum(images * images, axis=1)
        return self

    def predict(self, images: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(images), axis=1)

    def predict_proba(self, images: np.ndarray) -> np.ndarray:
        """Возвращает долю голосов соседей для каждой цифры от 0 до 9."""
        if self._images is None or self._labels is None or self._squared_norms is None:
            raise RuntimeError("Сначала вызовите fit()")

        images = np.asarray(images, dtype=np.float32)
        if images.ndim == 1:
            images = images.reshape(1, -1)
        if images.ndim != 2 or images.shape[1] != self._images.shape[1]:
            raise ValueError(f"Ожидаются изображения формы (количество, {self._images.shape[1]})")
        if len(images) == 0:
            return np.empty((0, 10), dtype=np.float64)

        return np.concatenate(list(self._probability_batches(images)))

    def _probability_batches(self, images: np.ndarray) -> Iterator[np.ndarray]:
        assert self._images is not None
        assert self._labels is not None
        assert self._squared_norms is not None

        for start in range(0, len(images), self.batch_size):
            batch = images[start : start + self.batch_size]
            # Евклидово расстояние в квадрате: ||x-y||² = ||x||² + ||y||² - 2xy.
            distances = (
                np.sum(batch * batch, axis=1)[:, None]
                + self._squared_norms[None, :]
                - 2.0 * batch @ self._images.T
            )
            nearest_indices = np.argpartition(distances, self.k - 1, axis=1)[:, : self.k]
            nearest_labels = self._labels[nearest_indices]

            probabilities = np.empty((len(batch), 10), dtype=np.float64)
            for row, candidates in enumerate(nearest_labels):
                probabilities[row] = np.bincount(candidates, minlength=10) / self.k
            yield probabilities
