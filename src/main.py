"""Run the phase 1 Phinuro demostration."""

from src.neuron import sigmoid, weighted_sum

def main()-> None:
    x1 = 0.5
    x2 = 0.8

    w1 = 0.4
    w2 = -0.2
    bias = 0.1

    z = weighted_sum(x1,x2,w1,w2,bias)
    output = sigmoid(z)

    print("Phinuro — Phase 1")
    print(f"Weighted sum (z): {z:.4f}")
    print(f"Sigmoid output:   {output:.4f}")


if __name__ == "__main__":
    main()