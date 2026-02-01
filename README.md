# iQuHack 2026: Superquantum Challenge

**Team Name:** 67 Qubits
**Team Members:** Adam Godel, Yebin Song, Nico Jackson, Travis Meyer, Timothy Wright  
**Affiliation:** Boston University, Boston, MA

---

## Project Write-up & Presentation Slides

Our complete technical writeup is available in the `writeup/` directory:

- **PDF**: [`writeup/main.pdf`](writeup/main.pdf) - **Start here!** This is the compiled document with all our implementations, optimizations, and experimental results.

- **Slides**: Access our slides [here!](https://docs.google.com/presentation/d/1q9z32plF9qOmsmKggezTibO6qC0Dm3mvM3ZvNwCfbd8/edit?usp=sharing)

## Project Overview

This repository contains our implementations of **12 unitary operators** using the **Clifford + T gate set**, where the T gate is the most computationally expensive gate to execute. Our goal was to optimize quantum circuits by:

1. **Minimizing T-gate count** - Reducing the "quantumness" and computational cost
2. **Minimizing approximation error** - Achieving high fidelity to target unitaries
3. **Exploring the tradeoff** - Balancing these competing objectives

---

## Repository Structure

### `unitary/` - Circuit Implementations

Contains Python implementations for all 12 unitaries:

- **`unitary1.py` through `unitary12.py`**: Individual implementations for each target unitary
- **`optim.py`**: Main optimization framework for building and synthesizing circuits
- **`utils.py`**: Utility functions for rotation gates (Rz, Ry, Rx)
- **`test.py`**: Testing framework with expected unitary matrices and distance calculations
- **`plot.py`**: Plotting utilities for T-count vs. distance analysis

### `qasm/` - QASM Circuit Files

Contains OpenQASM 3.0 representations of all optimized circuits:

```
qasm/unitary1.qasm
qasm/unitary2.qasm
...
qasm/unitary12.qasm
```

These files can be loaded into Qiskit or other quantum computing frameworks.

### `plots/` - Visualization Results

Contains generated plots showing the T-count vs. distance tradeoffs:

- `unitary_2_distance_vs_tcount.png`
- `unitary_3_distance_vs_tcount.png`
- `unitary_4_distance_vs_tcount.png`
- `unitary_6_distance_vs_tcount.png`
- `unitary7_plot.png` (fidelity-based)
- `unitary10_distance_vs_tcount.png`

### `writeup/` - Technical Documentation

The complete LaTeX writeup.

---

## How to Use

### Prerequisites

```bash
pip install qiskit
pip install gridsynth
pip install rmsynth
pip install numpy matplotlib scipy
```

### Running Individual Unitaries

Each unitary can be executed independently:

```bash
cd unitary
python3 unitary2.py   # Controlled-Ry(π/7) gate
python3 unitary7.py   # State preparation
python3 unitary11.py  # Four-qubit diagonal unitary
```

### Running Optimizations

To optimize a specific unitary with custom parameters:

```bash
cd unitary
python3 optim.py
```

Edit the file to select which unitary to optimize and adjust epsilon values.

### Generating Plots

To generate T-count vs. distance plots:

```bash
cd unitary
python3 plot.py
```

Modify the `constructions_to_analyze` list in the `if __name__ == "__main__"` block to select which unitaries to plot.

### Testing Implementations

To verify circuit correctness:

```bash
cd unitary
python3 test.py
```

This will load QASM files, compute circuit unitaries, and compare them against expected results.

---

## References

- **Gridsynth**: Ross, N. J., & Selinger, P. (2014). Optimal ancilla-free Clifford+T approximation of z-rotations.
- **Rmsynth**: Superquantum rmsynth package for phase polynomial optimization
- **Qiskit**: IBM's open-source quantum computing framework

## License

This project was created for the iQuHack 2026 hackathon. See individual package licenses for gridsynth, rmsynth, and Qiskit.

---

**For questions or issues, please open an issue on GitHub or contact the team.**
