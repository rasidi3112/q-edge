"""
Q-Edge Backend Module
====================

FastAPI backend for the Federated Hybrid Quantum-Neural Network platform.

Features:
- Post-Quantum Cryptography (PQC) authentication
- Azure Key Vault secret management
- Celery-based async quantum job processing
- RESTful API for federated learning coordination
"""

from src.backend.main import app, run_server
from src.backend.security import PQCAuthMiddleware
from src.backend.celery_app import celery_app

__all__ = [
    "app",
    "run_server",
    "PQCAuthMiddleware",
    "celery_app",
]
