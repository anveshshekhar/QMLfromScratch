---

# QML From Scratch

### Objective

This repository has been developed to act as a Centralized Implementation Library for Core QML Concepts. Instead of relying on library dependent implementations, I have focused on Learning & Implementing fundamentals concepts such as encoding schemes, parameterized quantum circuits, and hybrid optimization loops to provide deep insight into the mechanics of quantum-classical integration.

### Core Concepts Implemented

| Category | Concept |
| --- | --- |
| **Encoding** | Angle Embedding |
| **Optimization** | Parameter-Shift Rule |
| **Hybrid** | Variational Quantum Classifier |
| **Advanced** | Quantum Kernel Methods |

### Repository Structure

```text
qml-from-scratch/
├── data/                 # Datasets for benchmarking
├── src/
│   ├── encoding/         # Data re-uploading and feature maps
│   ├── circuits/         # Parametrized Quantum Circuits (Ansatz designs)
│   ├── kernels/          # Quantum Kernel implementations
│   ├── optimizers/       # Custom gradient descent and parameter-shift logic
│   └── utils/            # Plotting and state vector visualizations
├── notebooks/            # Explanatory tutorials for each implementation
├── tests/                # Unit tests for quantum circuit outputs
└── README.md

```

### Technical Stack

* **Language:** Python 3.10+
* **Quantum:** Qiskit (Core), PennyLane (Autodiff Engine)
* **ML Backend:** JAX (for high-performance gradient computation)

### Performance and Methodology

This repository prioritizes transparency in the learning process. All implementations are designed to be modular, allowing for the substitution of specific quantum backends (e.g., local simulators vs. cloud providers) and classical optimizers. The gradient computation relies on the parameter-shift rule to ensure analytical accuracy in quantum landscapes, avoiding numerical approximations where possible.

### Getting Started

1. Clone the repository: `git clone [[url](https://github.com/anveshshekhar/QMLfromScratch/)]`
2. Install dependencies: `pip install -r requirements.txt`
3. Execute a module: `python src/optimizers/gradient_calc.py`

