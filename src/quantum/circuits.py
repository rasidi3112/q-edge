"""
Variational Quantum Circuits (VQC) for Q-Edge Platform
======================================================

This module implements parameterized quantum circuits for the global
model aggregation in the Federated Hybrid Quantum-Neural Network.

Mathematical Foundation:
    The VQC implements a unitary operation U(θ) on n qubits:
    
    |ψ(θ)⟩ = U(θ)|0⟩^⊗n
    
    where U(θ) = ∏ᵢ Uᵢ(θᵢ) is a product of parameterized gates.

Author: Ahmad Rasidi (Roy)
License: Apache-2.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pennylane as qml
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class EntanglementPattern(Enum):
    """Entanglement topology patterns for VQC layers."""
    
    LINEAR = "linear"       # Nearest-neighbor entanglement
    CIRCULAR = "circular"   # Ring topology
    FULL = "full"           # All-to-all connectivity
    STAR = "star"           # Hub-and-spoke pattern
    BRICK = "brick"         # Brick-layer pattern


class AnsatzType(Enum):
    """Types of variational ansatz architectures."""
    
    HARDWARE_EFFICIENT = "hardware_efficient"
    STRONGLY_ENTANGLING = "strongly_entangling"
    ALTERNATING_LAYER = "alternating_layer"
    QNN_LAYER = "qnn_layer"
    IQP_EMBEDDING = "iqp_embedding"


@dataclass
class VQCConfig:
    """Configuration for Variational Quantum Circuit.
    
    Attributes:
        n_qubits: Number of qubits in the circuit.
        n_layers: Number of variational layers.
        ansatz_type: Type of variational ansatz.
        entanglement: Entanglement pattern between qubits.
        rotation_gates: Sequence of rotation gates per qubit.
        data_reuploading: Whether to use data re-uploading strategy.
        measurement_basis: Basis for final measurement (pauli_z, pauli_x, etc.).
    """
    
    n_qubits: int = 8
    n_layers: int = 4
    ansatz_type: AnsatzType = AnsatzType.STRONGLY_ENTANGLING
    entanglement: EntanglementPattern = EntanglementPattern.FULL
    rotation_gates: Sequence[str] = field(default_factory=lambda: ["RY", "RZ"])
    data_reuploading: bool = True
    measurement_basis: str = "pauli_z"
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.n_qubits < 2:
            raise ValueError("VQC requires at least 2 qubits")
        if self.n_layers < 1:
            raise ValueError("VQC requires at least 1 layer")
        
        valid_gates = {"RX", "RY", "RZ", "Rot"}
        for gate in self.rotation_gates:
            if gate not in valid_gates:
                raise ValueError(f"Invalid rotation gate: {gate}. Valid: {valid_gates}")


class VariationalQuantumCircuit:
    """
    Variational Quantum Circuit for Federated Learning Global Aggregation.
    
    This class implements a parameterized quantum circuit that can be used
    as a trainable model for processing aggregated weights from mobile devices
    in the federated learning pipeline.
    
    The circuit architecture follows the design principles from:
    - Schuld et al. (2021): "Effect of data encoding on the expressive power 
      of variational quantum machine learning models"
    - Cerezo et al. (2021): "Variational quantum algorithms"
    
    Mathematical Representation:
        The circuit implements the following transformation:
        
        ⟨O⟩ = ⟨0|^⊗n U†(θ) O U(θ) |0⟩^⊗n
        
        where:
        - U(θ) is the parameterized unitary
        - O is the observable (typically ⊗ᵢ Zᵢ)
        - θ are the trainable parameters
    
    Attributes:
        config: VQC configuration parameters.
        device: PennyLane quantum device.
        params: Current circuit parameters.
        
    Example:
        >>> config = VQCConfig(n_qubits=4, n_layers=3)
        >>> vqc = VariationalQuantumCircuit(config)
        >>> features = np.random.randn(4)
        >>> output = vqc.forward(features)
    """
    
    def __init__(
        self,
        config: VQCConfig,
        device_name: str = "default.qubit",
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the Variational Quantum Circuit.
        
        Args:
            config: VQC configuration object.
            device_name: PennyLane device name (default.qubit, lightning.qubit, etc.).
            shots: Number of measurement shots (None for exact simulation).
            seed: Random seed for reproducibility.
        """
        self.config = config
        self._shots = shots
        self._seed = seed
        
        # Initialize random number generator
        self._rng = np.random.default_rng(seed)
        
        # Create PennyLane device
        self.device = qml.device(
            device_name,
            wires=config.n_qubits,
            shots=shots,
        )
        
        # Calculate parameter shape based on ansatz
        self._param_shape = self._calculate_param_shape()
        
        # Initialize parameters using He initialization
        self.params = self._initialize_parameters()
        
        # Build the quantum circuit
        self._circuit = self._build_circuit()
        
        logger.info(
            f"Initialized VQC with {config.n_qubits} qubits, "
            f"{config.n_layers} layers, {self.num_params} parameters"
        )
    
    def _calculate_param_shape(self) -> tuple[int, ...]:
        """Calculate the shape of parameter tensor based on ansatz type."""
        n_qubits = self.config.n_qubits
        n_layers = self.config.n_layers
        
        if self.config.ansatz_type == AnsatzType.STRONGLY_ENTANGLING:
            # StronglyEntanglingLayers uses 3 rotations per qubit per layer
            return (n_layers, n_qubits, 3)
        elif self.config.ansatz_type == AnsatzType.HARDWARE_EFFICIENT:
            # Hardware efficient: 2 rotations per qubit per layer
            return (n_layers, n_qubits, 2)
        elif self.config.ansatz_type == AnsatzType.QNN_LAYER:
            # QNN layer: Single rotation + entanglement
            return (n_layers, n_qubits)
        else:
            # Default: rotations per layer
            n_rotations = len(self.config.rotation_gates)
            return (n_layers, n_qubits, n_rotations)
    
    def _initialize_parameters(self) -> NDArray[np.float64]:
        """
        Initialize circuit parameters using He initialization.
        
        He initialization helps prevent vanishing/exploding gradients
        in deep variational circuits.
        
        Returns:
            Initialized parameter array.
        """
        # He initialization: scale by sqrt(2/n)
        fan_in = np.prod(self._param_shape[1:])
        scale = np.sqrt(2.0 / fan_in)
        
        params = self._rng.normal(0, scale, self._param_shape)
        return params.astype(np.float64)
    
    def _build_circuit(self) -> qml.QNode:
        """
        Build the quantum circuit as a PennyLane QNode.
        
        Returns:
            Compiled QNode representing the variational circuit.
        """
        @qml.qnode(self.device, interface="autograd", diff_method="parameter-shift")
        def circuit(
            params: NDArray[np.float64],
            features: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            """
            Execute the variational quantum circuit.
            
            Args:
                params: Trainable circuit parameters.
                features: Input features to encode.
                
            Returns:
                Expectation values of Pauli-Z observables.
            """
            # Feature encoding using angle embedding
            self._encode_features(features)
            
            # Apply variational layers
            for layer in range(self.config.n_layers):
                # Data re-uploading (if enabled)
                if self.config.data_reuploading and layer > 0:
                    self._encode_features(features)
                
                # Apply variational layer
                self._apply_variational_layer(params[layer], layer)
                
                # Apply entanglement
                self._apply_entanglement(layer)
            
            # Measurement
            return self._measure()
        
        return circuit
    
    def _encode_features(self, features: NDArray[np.float64]) -> None:
        """
        Encode classical features into quantum state using angle embedding.
        
        The encoding uses the transformation:
            |0⟩ → cos(x/2)|0⟩ + sin(x/2)|1⟩
        
        Args:
            features: Classical feature vector.
        """
        n_features = min(len(features), self.config.n_qubits)
        
        for i in range(n_features):
            # Angle embedding with RY rotation
            qml.RY(features[i], wires=i)
    
    def _apply_variational_layer(
        self,
        layer_params: NDArray[np.float64],
        layer_idx: int,
    ) -> None:
        """
        Apply a single variational layer to all qubits.
        
        Args:
            layer_params: Parameters for this layer.
            layer_idx: Index of the current layer.
        """
        if self.config.ansatz_type == AnsatzType.STRONGLY_ENTANGLING:
            # Apply Rot gates (RZ-RY-RZ decomposition)
            for qubit in range(self.config.n_qubits):
                qml.Rot(
                    layer_params[qubit, 0],
                    layer_params[qubit, 1],
                    layer_params[qubit, 2],
                    wires=qubit,
                )
        else:
            # Apply specified rotation gates
            for qubit in range(self.config.n_qubits):
                for gate_idx, gate_name in enumerate(self.config.rotation_gates):
                    gate = getattr(qml, gate_name)
                    if gate_name == "Rot":
                        gate(
                            layer_params[qubit, gate_idx * 3],
                            layer_params[qubit, gate_idx * 3 + 1],
                            layer_params[qubit, gate_idx * 3 + 2],
                            wires=qubit,
                        )
                    else:
                        gate(layer_params[qubit, gate_idx], wires=qubit)
    
    def _apply_entanglement(self, layer_idx: int) -> None:
        """
        Apply entangling gates according to the configured pattern.
        
        Args:
            layer_idx: Index of the current layer.
        """
        n_qubits = self.config.n_qubits
        pattern = self.config.entanglement
        
        if pattern == EntanglementPattern.LINEAR:
            # Linear chain: (0,1), (1,2), (2,3), ...
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                
        elif pattern == EntanglementPattern.CIRCULAR:
            # Ring: Linear + (n-1, 0)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[n_qubits - 1, 0])
            
        elif pattern == EntanglementPattern.FULL:
            # All-to-all connectivity
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    qml.CNOT(wires=[i, j])
                    
        elif pattern == EntanglementPattern.STAR:
            # Hub-and-spoke (qubit 0 is hub)
            for i in range(1, n_qubits):
                qml.CNOT(wires=[0, i])
                
        elif pattern == EntanglementPattern.BRICK:
            # Brick layer pattern
            offset = layer_idx % 2
            for i in range(offset, n_qubits - 1, 2):
                qml.CNOT(wires=[i, i + 1])
    
    def _measure(self) -> NDArray[np.float64]:
        """
        Perform measurement on all qubits.
        
        Returns:
            Expectation values of Pauli-Z on all qubits.
        """
        return qml.probs(wires=range(self.config.n_qubits))
    
    @property
    def num_params(self) -> int:
        """Total number of trainable parameters."""
        return int(np.prod(self._param_shape))
    
    def forward(
        self,
        features: NDArray[np.float64],
        params: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        """
        Forward pass through the quantum circuit.
        
        Args:
            features: Input feature vector of shape (n_qubits,).
            params: Optional parameter override. Uses stored params if None.
            
        Returns:
            Circuit output probabilities of shape (2^n_qubits,).
        """
        if params is None:
            params = self.params
            
        # Ensure features match qubit count
        if len(features) < self.config.n_qubits:
            features = np.pad(
                features,
                (0, self.config.n_qubits - len(features)),
                mode='constant',
            )
        elif len(features) > self.config.n_qubits:
            features = features[:self.config.n_qubits]
        
        return self._circuit(params, features)
    
    def batch_forward(
        self,
        features_batch: NDArray[np.float64],
        params: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        """
        Process a batch of feature vectors.
        
        Args:
            features_batch: Batch of features of shape (batch_size, n_features).
            params: Optional parameter override.
            
        Returns:
            Batch of circuit outputs of shape (batch_size, 2^n_qubits).
        """
        results = []
        for features in features_batch:
            output = self.forward(features, params)
            results.append(output)
        return np.array(results)
    
    def get_circuit_depth(self) -> int:
        """
        Calculate the circuit depth (number of layers in the longest path).
        
        Returns:
            Circuit depth.
        """
        # Approximate depth calculation
        depth_per_layer = 2  # Rotation + entanglement
        if self.config.entanglement == EntanglementPattern.FULL:
            depth_per_layer += self.config.n_qubits - 1
        
        return self.config.n_layers * depth_per_layer + 1  # +1 for encoding
    
    def get_gate_count(self) -> dict[str, int]:
        """
        Count the number of each gate type in the circuit.
        
        Returns:
            Dictionary mapping gate names to counts.
        """
        gate_counts: dict[str, int] = {}
        
        # Rotation gates
        n_rotations = self.config.n_qubits * self.config.n_layers
        if self.config.ansatz_type == AnsatzType.STRONGLY_ENTANGLING:
            gate_counts["Rot"] = n_rotations
        else:
            for gate in self.config.rotation_gates:
                gate_counts[gate] = n_rotations
        
        # Entanglement gates
        if self.config.entanglement == EntanglementPattern.LINEAR:
            n_cnots = (self.config.n_qubits - 1) * self.config.n_layers
        elif self.config.entanglement == EntanglementPattern.CIRCULAR:
            n_cnots = self.config.n_qubits * self.config.n_layers
        elif self.config.entanglement == EntanglementPattern.FULL:
            n_cnots = (self.config.n_qubits * (self.config.n_qubits - 1) // 2) * self.config.n_layers
        elif self.config.entanglement == EntanglementPattern.STAR:
            n_cnots = (self.config.n_qubits - 1) * self.config.n_layers
        else:
            n_cnots = (self.config.n_qubits // 2) * self.config.n_layers
        
        gate_counts["CNOT"] = n_cnots
        gate_counts["RY"] = self.config.n_qubits  # Feature encoding
        
        return gate_counts
    
    def draw(self, style: str = "mpl") -> Any:
        """
        Draw the quantum circuit.
        
        Args:
            style: Drawing style ('mpl', 'text', or 'console').
            
        Returns:
            Circuit visualization.
        """
        # Create a sample execution for drawing
        sample_features = np.zeros(self.config.n_qubits)
        
        if style == "text":
            return qml.draw(self._circuit)(self.params, sample_features)
        else:
            return qml.draw_mpl(self._circuit)(self.params, sample_features)
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize circuit configuration and parameters to dictionary."""
        return {
            "config": {
                "n_qubits": self.config.n_qubits,
                "n_layers": self.config.n_layers,
                "ansatz_type": self.config.ansatz_type.value,
                "entanglement": self.config.entanglement.value,
                "rotation_gates": list(self.config.rotation_gates),
                "data_reuploading": self.config.data_reuploading,
            },
            "params": self.params.tolist(),
            "param_shape": list(self._param_shape),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VariationalQuantumCircuit":
        """Deserialize circuit from dictionary."""
        config = VQCConfig(
            n_qubits=data["config"]["n_qubits"],
            n_layers=data["config"]["n_layers"],
            ansatz_type=AnsatzType(data["config"]["ansatz_type"]),
            entanglement=EntanglementPattern(data["config"]["entanglement"]),
            rotation_gates=tuple(data["config"]["rotation_gates"]),
            data_reuploading=data["config"]["data_reuploading"],
        )
        
        circuit = cls(config)
        circuit.params = np.array(data["params"])
        return circuit
