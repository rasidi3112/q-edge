"""
Q-Edge: Federated Hybrid Quantum-Neural Network Platform
=========================================================

A production-ready platform for mobile-edge environments combining:
- Federated Learning (FL)
- Quantum Machine Learning (QML)
- Post-Quantum Cryptography (PQC)

Author: Ahmad Rasidi (Roy)
License: Apache-2.0
"""

__version__ = "1.0.0"
__author__ = "Ahmad Rasidi (Roy)"
__email__ = "roy@qedge.ai"

from src.quantum.aggregator import QuantumGlobalAggregator
from src.quantum.circuits import VariationalQuantumCircuit
from src.quantum.kernels import QuantumKernelAlignment

__all__ = [
    "QuantumGlobalAggregator",
    "VariationalQuantumCircuit", 
    "QuantumKernelAlignment",
    "__version__",
]
