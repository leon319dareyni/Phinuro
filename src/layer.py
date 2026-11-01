"""Dense layer implemention for Phinuro."""

from src.neuron import sigmoid

class DenseLayer:
    """A fully connected layer of artifical neurons."""

    def __init__(
            self,
            weights: list[list[float]],
            biases: list[float],
    )-> None:
        if not weights:
            raise ValueError("A layer must contain at least one neuron.")

        if len(weights) != len(biases):
            raise ValueError(
                "The number of weight vectors must match the number of biases."
            )

        input_size = len(weights[0])

        if input_size == 0:
            raise ValueError("Each neuron must have at least one input.")

        if any(len(neuron_weights) != input_size for neuron_weights in weights):
            raise ValueError(
                "All neurons in a dense layer must have the same input size."
            )

        self.weightes = weights
        self.biases = biases
        self.input_size = input_size
        self.neuron_count = len(weights)

    def forward(
            self,
            inputs: list[float],
    )-> list[float]:
        """Calculate the outputs of every neuron in the layer."""

        if len(inputs) != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} inputs, got {len(inputs)}"
            )

        outputs = []

        for neuron_weights, bias in zip(self.weightes,self.biases):
            z = sum(
                weight * value 
                for weight, value in zip(neuron_weights,self.input_size)
            )

            z += bias 

            outputs.append(sigmoid(z))

        return outputs
    
