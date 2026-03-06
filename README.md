# QML From Scratch: Hybrid Quantum Classification

### Objective

This repository serves as a centralized library for implementing Quantum Machine Learning (QML) fundamentals from the ground up. By moving beyond high-level "black box" libraries, this project focuses on the mechanics of _Quantum Classical Integration_, including custom encoding schemes, Variational Quantum Circuit (VQC) architectures, and hybrid optimization loops.

---

### Experimental Results

Through iterative development, this project reached the following performance benchmarks on an MNIST Dataset:

| Architecture | Ansatz Depth | Accuracy | Status |
| --- | --- | --- | --- |
| *Variational Ansatz* | 3 Layers | 67.5% | Under-fit |
| *Variational Ansatz* | 4 Layers | *90.0%* | *Optimal* |
| *Variational Ansatz* | 5 Layers | 65.0% | Over-fit/Barren Plateau |
| *Quantum Kernel SVM* | N/A | *100%* | *Baseline* |

---

### Key Technical Achievements

* **Kernel-Based Separability**: Achieved 100% accuracy using a Quantum Kernel SVM, confirming perfect linear separability of the dataset in the quantum feature space.
* **Optimal Ansatz Architecture**: Identified a **4-layer `TwoLocalAnsatz**` as the "Goldilocks" depth. This configuration balances sufficient expressive power with the stability required to avoid vanishing gradients.
* **Optimization Precision**: Refined the training pipeline by scaling the learning rate from `0.05` to `0.02` to resolve convergence jitter, ultimately pushing the Ansatz accuracy to *90.0%*.
* **Convergence Analysis**: Utilized cost function history to identify the optimal stopping point (Epoch 25), mitigating the risk of overfitting during long-duration training runs.

---

### Repository Structure

```text
qml-from-scratch/
├── data/               # MNIST subset datasets
├── src/
│   ├── encoding/       # AngleEmbedding and feature map logic
│   ├── circuits/       # Parametrized Quantum Circuits (TwoLocalAnsatz)
│   ├── kernels/        # Quantum Kernel SVM implementations
│   ├── optimizers/     # JAX-based hybrid training loops
│   └── utils/          # Confusion matrix and cost-history plotting
├── train_model.py      # Main training orchestration
├── evaluate_models.py  # Model validation and performance logging
└── README.md

```

### Technical Stack

* **Language:** Python 3.10+
* **Quantum/Autodiff:** PennyLane (Lightning Qubit), Qiskit
* **ML Backend:** JAX (High-performance gradient computation)

---

### Getting Started

1. **Clone:** `git clone https://github.com/anveshshekhar/QMLfromScratch/`
2. **Setup:** `pip install -r requirements.txt`
3. **Train:** `python train_model.py` 
4. **Evaluate:** `python evaluate_models.py`

---
