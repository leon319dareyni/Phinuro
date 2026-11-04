# Phinuro 🧠

### Building Neural Networks From Scratch

Phinuro is an educational project focused on understanding how neural networks work by implementing their fundamental components from the ground up.

Rather than relying on machine-learning frameworks to perform the core computations, Phinuro starts with simple Python implementations and builds toward a small neural-network library. Each phase introduces new concepts alongside the mathematics and tests needed to verify them.

> **Current status:** Phase 2 — Multiple Neurons and Dense Layers

---

## Project objectives

- Understand how artificial neurons process numerical inputs.
- Implement neural-network components and algorithms in Python.
- Connect the code to the underlying mathematics.
- Build forward propagation, loss functions, and backpropagation step by step.
- Implement parameter optimization without automatic differentiation.
- Test mathematical correctness and explore model behavior through experiments.

## Technology stack

| Technology | Purpose |
|---|---|
| Python 3.10+ | Core implementation |
| Python `math` module | Mathematical operations and sigmoid activation |
| Python `unittest` | Automated tests |
| Git and GitHub | Version control and project hosting |

**Dependencies:** Phinuro currently uses only Python's standard library. NumPy, PyTorch, TensorFlow, and other machine-learning frameworks are not used in the current phases.

---

# Development roadmap

- [x] **Phase 1 — Single Artificial Neuron**
- [x] **Phase 2 — Multiple Neurons and Dense Layers**
- [ ] Phase 3 — Loss Functions
- [ ] Phase 4 — Backpropagation
- [ ] Phase 5 — Gradient Descent and Optimization
- [ ] Phase 6 — Training Engine and Dataset Handling
- [ ] Phase 7 — Model Evaluation and Visualization
- [ ] Phase 8 — Model Saving and Loading
- [ ] Phase 9 — Advanced Architectures and Experiments

The roadmap may evolve as the project develops.

---

# Phase 1 — Single Artificial Neuron

Phase 1 implemented a single neuron with two inputs, two manually assigned weights, a bias, and a sigmoid activation function.

The neuron performs a forward pass in two steps.

### 1. Weighted sum

\[
z = w_1x_1 + w_2x_2 + b
\]

Where \(x_1\) and \(x_2\) are inputs, \(w_1\) and \(w_2\) are their weights, and \(b\) is the bias.

### 2. Sigmoid activation

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

The sigmoid function maps any finite real input to a value strictly between zero and one.

### 3. Neuron output

\[
\hat{y} = \sigma(w_1x_1 + w_2x_2 + b)
\]

The weights and bias in Phase 1 are supplied manually. The neuron does not learn from data.

---

# Phase 2 — Multiple Neurons and Dense Layers

## Overview

Phase 2 extends the single-neuron implementation into a **dense layer** containing multiple neurons. Every neuron receives the same input vector, but each neuron has its own weight vector and bias.

The current demonstration takes two input features and produces three outputs.

## Mathematical foundation

Let the input vector be:

\[
\mathbf{x} =
\begin{bmatrix}
0.5 \\
0.8
\end{bmatrix}
\]

The layer has three neurons. Their weights are stored in a matrix:

\[
W =
\begin{bmatrix}
0.4 & -0.2 \\
0.1 & 0.3 \\
-0.5 & 0.7
\end{bmatrix}
\]

The bias vector is:

\[
\mathbf{b} =
\begin{bmatrix}
0.1 \\
-0.1 \\
0.2
\end{bmatrix}
\]

The complete layer calculation is:

\[
\mathbf{z} = W\mathbf{x} + \mathbf{b}
\]

The dimensions are compatible:

\[
(3 \times 2)(2 \times 1) + (3 \times 1) = (3 \times 1)
\]

The layer then applies sigmoid separately to each element:

\[
\mathbf{a} = \sigma(\mathbf{z})
\]

Here, \(\sigma\) is applied element-wise. The result is one activated output for each neuron.

## Example calculation

Using the inputs, weights, and biases above:

\[
z_1 = (0.4)(0.5) + (-0.2)(0.8) + 0.1 = 0.14
\]

\[
z_2 = (0.1)(0.5) + (0.3)(0.8) - 0.1 = 0.19
\]

\[
z_3 = (-0.5)(0.5) + (0.7)(0.8) + 0.2 = 0.51
\]

Therefore:

\[
\mathbf{z} =
\begin{bmatrix}
0.14 \\
0.19 \\
0.51
\end{bmatrix}
\qquad
\mathbf{a} \approx
\begin{bmatrix}
0.5349 \\
0.5474 \\
0.6248
\end{bmatrix}
\]

The layer transforms **two input values into three output values**.

## Implementation details

The `DenseLayer` class in `src/layer.py` stores:

- `weights`: one weight vector per neuron.
- `biases`: one bias per neuron.
- `input_size`: the number of values expected by each neuron.
- `neuron_count`: the number of neurons in the layer.

Its `forward()` method validates the input size, computes the weighted sum for every neuron, adds the corresponding bias, applies sigmoid, and returns the list of outputs.

The implementation uses Python lists and loops so the calculations remain visible. Matrix-based operations with NumPy can be introduced later as a deliberate implementation step.

**Important:** Phase 2 still does not train the network. Weights and biases are manually supplied. Loss functions, gradient calculation, backpropagation, and parameter updates are future phases.

---

# Project structure

```text
Phinuro/
│
├── README.md
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── neuron.py
│   ├── layer.py
│   └── main.py
│
└── tests/
    ├── __init__.py
    ├── test_neuron.py
    └── test_layer.py
```

| File | Responsibility |
|---|---|
| `README.md` | Project documentation, mathematics, and setup instructions |
| `.gitignore` | Excludes generated files and local environments from version control |
| `src/__init__.py` | Marks `src` as a Python package |
| `src/neuron.py` | Implements the sigmoid function, weighted sum, and single-neuron forward pass |
| `src/layer.py` | Implements a dense layer containing multiple neurons |
| `src/main.py` | Runs the current phase demonstration |
| `tests/test_neuron.py` | Tests the Phase 1 neuron calculations |
| `tests/test_layer.py` | Tests dense-layer outputs and input-dimension validation |

---

# Installation and usage

## Requirements

- Python 3.10 or newer recommended.
- Git, if cloning the repository.

No third-party Python packages are required for the current phases.

## 1. Clone the repository

Replace the placeholder with the URL of your GitHub repository.

```bash
git clone https://github.com/leon319dareyni/Phinuro
cd Phinuro
```

## 2. Create and activate a virtual environment

On Linux and macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

## 3. Run the demonstration

Run this command from the repository root:

```bash
python -m src.main
```

Expected output:

```text
Phinuro — Phase 2
Input:   [0.5, 0.8]
Neurons: 3
Output:  [0.5349, 0.5474, 0.6248]
```

## 4. Run the test suite

```bash
python -m unittest discover -s tests -v
```

The tests check the single-neuron calculations, dense-layer output values, output count, and handling of invalid input dimensions.

---

# Development principles

1. **Understand before abstracting.** Understand the mathematical operation before hiding it behind a larger interface.
2. **Implement core mechanics manually.** Avoid delegating the central learning algorithms to frameworks while developing the educational foundation.
3. **Test mathematical correctness.** Compare outputs with known values and test important properties and invalid cases.
4. **Build incrementally.** Each phase should extend the existing implementation without unnecessary rewrites.
5. **Experiment and observe.** Change inputs and parameters, predict the result, run the code, and investigate differences.

---

# License

Phinuro is currently an educational and experimental project. No open-source license has been selected yet; until one is added to the repository, no particular open-source license is implied.

---

**Phinuro — Understand the mathematics. Build the intelligence.**
