"""
Mobile Edge Simulation Module
=============================

This module provides simulations for mobile edge devices participating
in the Federated Hybrid Quantum-Neural Network platform.

Features:
- Flower (flwr) based federated learning client
- PySyft integration for privacy-preserving ML
- PQC-secured communication simulation
- Mobile device resource constraints simulation
"""

from src.mobile.fl_client import MobileFlowerClient
from src.mobile.pqc_transport import PQCTransportLayer

__all__ = [
    "MobileFlowerClient",
    "PQCTransportLayer",
]
