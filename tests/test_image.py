from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageDraw

from digit_recognizer import load_digit_image, prepare_digit_image


class ImagePreprocessingTest(unittest.TestCase):
    def test_black_digit_on_white_background_is_converted_to_mnist_format(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "digit.png"
            image = Image.new("L", (100, 100), 255)
            ImageDraw.Draw(image).rectangle((40, 15, 60, 85), fill=0)
            image.save(path)

            result = load_digit_image(path)

            self.assertEqual(result.shape, (784,))
            self.assertEqual(result.dtype, np.float32)
            self.assertGreaterEqual(float(result.min()), 0.0)
            self.assertLessEqual(float(result.max()), 1.0)
            self.assertGreater(float(result.sum()), 0.0)
            self.assertLess(float(result[0]), 0.01)

    def test_smoothing_can_be_disabled(self) -> None:
        image = Image.new("L", (101, 103), 255)
        ImageDraw.Draw(image).line((10, 10, 90, 90), fill=0, width=7)
        smoothed = prepare_digit_image(image, smoothing=True)
        nearest = prepare_digit_image(image, smoothing=False)
        self.assertEqual(smoothed.shape, nearest.shape)
        self.assertGreater(np.count_nonzero((smoothed > 0) & (smoothed < 1)), 0)
        self.assertFalse(np.array_equal(smoothed, nearest))

    def test_empty_image_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.png"
            Image.new("L", (28, 28), 255).save(path)
            with self.assertRaisesRegex(ValueError, "Сначала нарисуйте цифру"):
                load_digit_image(path)


if __name__ == "__main__":
    unittest.main()
