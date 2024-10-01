# Matrix Product Operator (MPO) Simulation for Noisy Random Circuit Sampling

This project is a Python-based simulation of quantum circuits using Matrix Product Operator (MPO) techniques. The package provides tools for initializing MPOs, applying single- and two-qubit gate operations, adding noise, and calculating various observables such as entanglement entropy and total probability.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [References](#references)

## Features

- **Initialization of MPO**: Initialize the MPO with product state $\ket{0^n}$ or maximally mixed states.
- **Gate Operations**: Apply single-qudit and two-qudit operations to update the MPO tensors.
- **Noise Channels**: Add depolarizing and amplitude damping channels to simulate realistic quantum systems.
- **Entanglement Entropy Calculation**: Calculate the entanglement entropy across different bipartitions.
- **Random Unitary Generation**: Generate random two-qudit unitaries and apply them in the circuit.

## Project Structure
The project follows a modular design pattern, with each aspect of the MPO simulation separated into its own module:
MPO-Quantum-Circuit-Simulation/
```
├── src/
│   ├── __init__.py         # Combines all components into a cohesive MPO class
│   ├── core.py             # Core implementation of the MPO class
│   ├── initialization.py   # Methods for initializing the MPO
│   ├── updates.py          # Methods for applying single- and two-qubit gate updates
│   ├── observables.py      # Methods for calculating observables like entanglement entropy
│   └── random_circuits.py  # Utility functions for applying noisy gates on MPO
├── examples/
│   └── NoisyRCS.py         # Example usage of the MPO class
├── README.md               
└── LICENSE                 
```

## References

Please refer to the [references.bib](references.bib) file.
