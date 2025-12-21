"""
Q-Edge Test Configuration
=========================

Pytest configuration and fixtures for the test suite.
"""

import pytest
import asyncio
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration."""
    return {
        "n_qubits": 4,
        "n_layers": 2,
        "shots": 100,
        "seed": 42,
    }


@pytest.fixture
def random_weights():
    """Generate random model weights for testing."""
    import numpy as np
    np.random.seed(42)
    return np.random.randn(100)


@pytest.fixture
def sample_updates():
    """Generate sample FL updates for testing."""
    import numpy as np
    from src.quantum.aggregator import LocalModelUpdate
    
    np.random.seed(42)
    return [
        LocalModelUpdate(
            client_id=f"client_{i}",
            weights=np.random.randn(100),
            n_samples=100 + i * 10,
            local_loss=0.5 - i * 0.05,
        )
        for i in range(5)
    ]


# Enable async tests
pytest_plugins = ('pytest_asyncio',)
