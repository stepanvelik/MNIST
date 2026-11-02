import gzip
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from digit_recognizer import load_mnist


class MnistLoadingTest(unittest.TestCase):
    def test_reads_idx_and_applies_limit_and_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            images = bytes([0, 255, 128, 64])
            labels = bytes([3, 8])
            with gzip.open(data_dir / "train-images-idx3-ubyte.gz", "wb") as file:
                file.write(struct.pack(">IIII", 2051, 2, 1, 2) + images)
            with gzip.open(data_dir / "train-labels-idx1-ubyte.gz", "wb") as file:
                file.write(struct.pack(">II", 2049, 2) + labels)

            loaded_images, loaded_labels = load_mnist("train", data_dir, limit=1)

            self.assertEqual(loaded_images.shape, (1, 2))
            np.testing.assert_allclose(loaded_images, [[0.0, 1.0]])
            np.testing.assert_array_equal(loaded_labels, [3])


if __name__ == "__main__":
    unittest.main()
