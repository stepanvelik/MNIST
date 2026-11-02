"""Простое распознавание рукописных цифр методом k-ближайших соседей."""

from .image import load_digit_image, prepare_digit_image
from .knn import KNNClassifier
from .mnist import load_mnist

__all__ = ["KNNClassifier", "load_digit_image", "load_mnist", "prepare_digit_image"]
