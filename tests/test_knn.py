import unittest

import numpy as np

from digit_recognizer import KNNClassifier


class KNNClassifierTest(unittest.TestCase):
    def setUp(self) -> None:
        self.images = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.0, 0.1],
                [1.0, 1.0],
                [0.9, 1.0],
                [1.0, 0.9],
            ],
            dtype=np.float32,
        )
        self.labels = np.array([0, 0, 0, 1, 1, 1])

    def test_predicts_one_and_many_images(self) -> None:
        classifier = KNNClassifier(k=3, batch_size=1).fit(self.images, self.labels)
        result = classifier.predict(np.array([[0.05, 0.05], [0.95, 0.95]]))
        np.testing.assert_array_equal(result, [0, 1])
        np.testing.assert_array_equal(classifier.predict(self.images[3]), [1])

    def test_calculates_confidence_from_neighbour_votes(self) -> None:
        classifier = KNNClassifier(k=3).fit(self.images, self.labels)
        probabilities = classifier.predict_proba(np.array([0.05, 0.05]))[0]
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)
        self.assertAlmostEqual(float(probabilities[0]), 1.0)
        self.assertAlmostEqual(float(probabilities[1]), 0.0)

    def test_rejects_invalid_settings_and_unfitted_predict(self) -> None:
        with self.assertRaises(ValueError):
            KNNClassifier(k=0)
        with self.assertRaises(RuntimeError):
            KNNClassifier().predict(self.images)
        with self.assertRaises(ValueError):
            KNNClassifier(k=7).fit(self.images, self.labels)


if __name__ == "__main__":
    unittest.main()
