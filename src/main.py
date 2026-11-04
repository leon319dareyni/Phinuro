"""Demonstrate a dense layer in Phinuro."""

from src.layer import DenseLayer

def main()-> None:
    weights = [
        [0.4,-0.2],
        [0.1,0.3],
        [-0.5,0.7]
    ]

    biases = [0.1,-0.1,0.2]

    layer = DenseLayer(
        weights=weights,
        biases=biases,
    )

    inputs = [0.5,0.8]

    outputs = layer.forward(inputs)

    print("Phinuro — Phase 2")
    print(f"Input:   {inputs}")
    print(f"Neurons: {layer.neuron_count}")
    print(f"Output:  {[round(value, 4) for value in outputs]}")

if __name__ == "__main__":
    main()