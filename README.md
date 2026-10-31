# Phinuro 🧠
### Building Neural Networks From Scratch

Phinuro is an educational project dedicated to understanding artificial intelligence and neural networks by implementing their fundamental components from scratch.

The goal is to move beyond simply using machine learning frameworks and understand the mathematics, algorithms, and computational processes that make neural networks work.

Instead of relying on frameworks such as PyTorch or TensorFlow, Phinuro begins with basic Python implementations and gradually evolves into a neural network library.

> **Current status:** Phase 1 — Single Artificial Neuron

---

## 🎯 Project Objectives

Phinuro aims to provide a practical, mathematical understanding of neural networks through incremental implementation.

The main objectives are:

- Understand how artificial neurons process information.
- Implement neural network components using Python.
- Explore the mathematical foundations of machine learning.
- Implement forward propagation, loss functions, and backpropagation.
- Build learning algorithms without automatic differentiation.
- Visualize training processes and model behavior.
- Develop a lightweight neural network library for educational purposes.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| `math` | Mathematical operations and activation functions |
| `unittest` | Automated testing |

**Dependencies:** No third-party Python packages are required for Phase 1.

---

# 📚 Development Roadmap

Phinuro is being developed in incremental phases, with each phase introducing new concepts and capabilities.

- [x] **Phase 1 — Single Artificial Neuron**
- [ ] Phase 2 — Multiple Neurons and Dense Layers
- [ ] Phase 3 — Loss Functions
- [ ] Phase 4 — Backpropagation
- [ ] Phase 5 — Gradient Descent and Optimization
- [ ] Phase 6 — Training Engine and Dataset Handling
- [ ] Phase 7 — Model Evaluation and Visualization
- [ ] Phase 8 — Model Saving and Loading
- [ ] Phase 9 — Advanced Architectures and Experiments

The roadmap may evolve as the project develops.

---

# 🧮 Phase 1: Single Artificial Neuron

## Overview

The first phase implements a single artificial neuron with two inputs, two weights, a bias, and a sigmoid activation function.

The neuron performs a forward pass:

1. Receives input values.
2. Multiplies each input by its corresponding weight.
3. Adds the weighted values and bias.
4. Applies the sigmoid activation function.
5. Returns the resulting output.

At this stage, the parameters are manually assigned. The neuron does not learn from data.

## Mathematical Foundation

### 1. Weighted Sum

The neuron first calculates a weighted sum:

\[
z = w_1x_1 + w_2x_2 + b
\]

Where:

- \(x_1, x_2\): Input features.
- \(w_1, w_2\): Weights associated with the inputs.
- \(b\): Bias.
- \(z\): Weighted sum before activation.

Weights determine how strongly each input contributes to the result, while the bias provides an additional adjustable offset.

### 2. Sigmoid Activation

The weighted sum is passed through the sigmoid function:

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

The sigmoid function maps every finite real-valued input to a value strictly between zero and one.

This transformation is useful for understanding nonlinear activation functions and will be important when we later implement binary classification.

### 3. Neuron Output

The complete forward pass is:

\[
\hat{y} = \sigma(w_1x_1 + w_2x_2 + b)
\]

Here, \(\hat{y}\) represents the neuron's output.

Although sigmoid outputs can be interpreted as probabilities in an appropriately trained binary classification model, this first implementation simply produces an activated numerical output.

---

## 🧪 Example Calculation

Consider the following inputs and parameters:

| Parameter | Value |
|---|---:|
| \(x_1\) | 0.5 |
| \(x_2\) | 0.8 |
| \(w_1\) | 0.4 |
| \(w_2\) | -0.2 |
| \(b\) | 0.1 |

### Step 1: Calculate the weighted sum

\[
z = (0.5)(0.4) + (0.8)(-0.2) + 0.1
\]

\[
z = 0.2 - 0.16 + 0.1
\]

\[
z = 0.14
\]

### Step 2: Apply the sigmoid function

\[
\hat{y} = \frac{1}{1 + e^{-0.14}}
\]

\[
\boxed{\hat{y} \approx 0.5349}
\]

The neuron produces an output of approximately **0.5349**.

---

# 📁 Project Structure

```text
Phinuro/
│
├── README.md
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── neuron.py
│   └── main.py
│
└── tests/
    ├── __init__.py
    └── test_neuron.py
```

### Directory Responsibilities

| File | Description |
|---|---|
| `README.md` | Project documentation |
| `.gitignore` | Excludes generated files and virtual environments from version control |
| `src/__init__.py` | Defines the source directory as a Python package |
| `src/neuron.py` | Implements the neuron and its mathematical operations |
| `src/main.py` | Runs the Phase 1 demonstration |
| `tests/test_neuron.py` | Contains automated tests for the implementation |

---

# 🚀 Installation and Usage

## Requirements

- Python 3.10 or newer recommended.
- Git, if you want to clone and contribute to the repository.

No external Python packages are needed for Phase 1.

## 1. Clone the Repository

Replace the placeholder with your actual GitHub repository URL.

```bash
git clone https://github.com/leon319dareyni/Phinuro.git
cd Phinuro
```

## 2. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows, activate the environment using:

```powershell
.venv\Scripts\Activate.ps1
```

## 3. Run the Neuron

From the project root:

```bash
python -m src.main
```

Expected output:

```text
NeuralCore — Phase 1
Weighted sum (z): 0.1400
Sigmoid output:   0.5349
```

## 4. Run the Tests

```bash
python -m unittest discover -s tests -v
```

The test suite checks the expected behavior of the mathematical operations and neuron output.

---

# 🔬 Learning Principles

Phinuro follows several principles throughout its development:

**1. Understand before abstracting**

Every major mathematical operation should be understood before being incorporated into a larger component.

**2. Implement before relying on frameworks**

Core algorithms will initially be implemented manually rather than delegated to machine learning frameworks.

**3. Test mathematical correctness**

Implementations should be tested against known results and mathematical properties.

**4. Build incrementally**

Each phase should extend the previous implementation without introducing unnecessary complexity.

**5. Learn through experimentation**

Changing parameters, observing outputs, and investigating unexpected behavior are essential parts of the learning process.

---

# 🗺️ Future Direction

After Phase 1, Phinuro will progress from a single neuron to interconnected layers of neurons.

Future development will introduce:

- Dense layers and matrix-based computation.
- ReLU, tanh, and additional activation functions.
- Regression and classification loss functions.
- Manual gradient calculation.
- Backpropagation using the chain rule.
- Gradient descent and other optimization algorithms.
- Mini-batch training and validation.
- Training metrics and decision-boundary visualization.
- Model serialization and reproducible experiments.

The long-term goal is to build a small but understandable neural network library that demonstrates the fundamental mechanics of deep learning.

---

# 📜 License

The project is currently intended for educational and experimental purposes.

A formal open-source license can be added when one is selected. Until then, no particular open-source license is implied.

---

**Phinuro — Understand the mathematics. Build the intelligence.**