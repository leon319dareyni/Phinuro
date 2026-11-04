import unittest

from src.layer import DenseLayer


class TestDenseLayer(unittest.TestCase):

    def setUp(self):
        self.layer = DenseLayer(
            weights=[
                [0.4, -0.2],
                [0.1, 0.3],
                [-0.5, 0.7],
            ],
            biases=[0.1, -0.1, 0.2],
        )

    def test_output_count(self):
        outputs = self.layer.forward([0.5,0.8])

        self.assertEqual(len(outputs), 3)

    def test_expected_outputs(self):
        outputs = self.layer.forward([0.5,0.8])

        expected = [
            0.5349429452,
            0.5473576181,
            0.6248064745,
        ]

        for actual, target in zip(outputs,expected):
            self.assertAlmostEqual(actual,target,places=5)


    def test_wrong_input_size(self):
        with self.assertRaises(ValueError):
            self.layer.forward([0.5])

    def test_mismatched_bias_count(self):
        with self.assertRaises(ValueError):
            DenseLayer(
                weights=[[0.2, 0.3], [0.4, 0.5]],
                biases=[0.1],
            )

    def test_inconsistent_weight_dimensions(self):
        with self.assertRaises(ValueError):
            DenseLayer(
                weights=[[0.2, 0.3], [0.4]],
                biases=[0.1, 0.2],
            )


if __name__ == "__main__":
    unittest.main()
