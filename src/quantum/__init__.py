"""
Quantum Computing Module
========================

This module contains all quantum computing components including:
- Variational Quantum Circuits (VQC)
- Quantum Kernel Alignment (QKA)
- Quantum Error Mitigation (QEM)
- Azure Quantum Integration
"""

from src.quantum.aggregator import QuantumGlobalAggregator
from src.quantum.circuits import VariationalQuantumCircuit
from src.quantum.kernels import QuantumKernelAlignment
from src.quantum.error_mitigation import ZeroNoiseExtrapolation
from src.quantum.azure_connector import AzureQuantumConnector

__all__ = [
    "QuantumGlobalAggregator",
    "VariationalQuantumCircuit",
    "QuantumKernelAlignment",
    "ZeroNoiseExtrapolation",
    "AzureQuantumConnector",
]
