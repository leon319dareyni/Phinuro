import unittest

from src.neuron import neuron, sigmoid, weighted_sum


class TestSigmoid(unittest.TestCase):

    def test_sigmoid_at_zero(self):
        self.assertAlmostEqual(sigmoid(0.0), 0.5)

    def test_sigmoid_at_positive_value(self):
        result = sigmoid(2.0)

        self.assertGreater(result, 0.0)
        self.assertLess(result, 1.0)


class TestWeightedSum(unittest.TestCase):

    def test_weighted_sum(self):
        result = weighted_sum(
            0.5, 0.8, 0.4, -0.2, 0.1
        )

        self.assertAlmostEqual(result, 0.14)


class TestNeuron(unittest.TestCase):

    def test_neuron_output(self):
        result = neuron(
            0.5, 0.8, 0.4, -0.2, 0.1
        )

        self.assertAlmostEqual(
            result, 0.5349429451582145
        )


if __name__ == "__main__":
    unittest.main()