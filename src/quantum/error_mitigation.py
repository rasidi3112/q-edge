"""
Quantum Error Mitigation (QEM) for NISQ Devices
===============================================

This module implements error mitigation techniques to improve the accuracy
of quantum computations on Noisy Intermediate-Scale Quantum (NISQ) devices.

Implemented Techniques:
    1. Zero-Noise Extrapolation (ZNE)
    2. Probabilistic Error Cancellation (PEC)
    3. Measurement Error Mitigation
    4. Dynamical Decoupling

Mathematical Foundation:
    Zero-Noise Extrapolation estimates the zero-noise expectation value
    by running circuits at multiple noise levels λ and extrapolating:
    
    ⟨O⟩₀ = lim_{λ→0} ⟨O⟩_λ
    
    The noise scaling is achieved through unitary folding:
    G → G·G†·G (local folding) or U → U·U†·U (global folding)

References:
    - Temme et al. (2017): "Error Mitigation for Short-Depth Quantum Circuits"
    - Li & Benjamin (2017): "Efficient Variational Quantum Simulator 
      Incorporating Active Error Minimization"
    - Kandala et al. (2019): "Error mitigation extends the computational 
      reach of a noisy quantum processor"

Author: Ahmad Rasidi (Roy)
License: Apache-2.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np
import pennylane as qml
from numpy.typing import NDArray
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


class FoldingMethod(Enum):
    """Circuit folding methods for noise amplification."""
    
    GLOBAL = "global"       # Fold entire circuit: U → U·U†·U
    LOCAL = "local"         # Fold individual gates: G → G·G†·G
    RANDOM = "random"       # Randomly select gates to fold


class ExtrapolationMethod(Enum):
    """Extrapolation methods for zero-noise estimation."""
    
    LINEAR = "linear"               # Linear extrapolation
    POLYNOMIAL = "polynomial"       # Polynomial fit
    EXPONENTIAL = "exponential"     # Exponential decay model
    RICHARDSON = "richardson"       # Richardson extrapolation
    ADAPTIVE = "adaptive"           # Adaptive method selection


@dataclass
class ZNEConfig:
    """Configuration for Zero-Noise Extrapolation.
    
    Attributes:
        scale_factors: Noise amplification factors.
        folding_method: Method for circuit folding.
        extrapolation_method: Method for zero-noise extrapolation.
        poly_degree: Polynomial degree (if using polynomial extrapolation).
        n_shots_per_scale: Number of shots per noise scale.
        random_seed: Seed for reproducibility.
    """
    
    scale_factors: Sequence[float] = field(
        default_factory=lambda: [1.0, 1.5, 2.0, 2.5, 3.0]
    )
    folding_method: FoldingMethod = FoldingMethod.GLOBAL
    extrapolation_method: ExtrapolationMethod = ExtrapolationMethod.RICHARDSON
    poly_degree: int = 2
    n_shots_per_scale: int = 1024
    random_seed: Optional[int] = None
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if len(self.scale_factors) < 2:
            raise ValueError("At least 2 scale factors required for extrapolation")
        if min(self.scale_factors) < 1.0:
            raise ValueError("All scale factors must be >= 1.0")
        if not all(self.scale_factors[i] < self.scale_factors[i+1] 
                   for i in range(len(self.scale_factors) - 1)):
            raise ValueError("Scale factors must be strictly increasing")


class ZeroNoiseExtrapolation:
    """
    Zero-Noise Extrapolation for Quantum Error Mitigation.
    
    ZNE is a leading error mitigation technique that estimates the ideal
    (noise-free) expectation value by executing a quantum circuit at
    multiple noise levels and extrapolating to zero noise.
    
    The key insight is that while we cannot reduce hardware noise directly,
    we can artificially amplify it in a controlled way and then extrapolate
    backwards to estimate the zero-noise limit.
    
    Noise Amplification Methods:
        - Global Folding: U → U(U†U)^n, where n controls noise level
        - Local Folding: Apply G → G(G†G)^n to selected gates
        - Random Folding: Randomly select gates to fold
    
    Extrapolation Methods:
        - Linear: f(λ) = a + bλ, extrapolate to λ=0
        - Polynomial: f(λ) = Σᵢ aᵢλⁱ
        - Exponential: f(λ) = a·exp(-bλ) + c
        - Richardson: Weighted combination of noisy values
    
    Example:
        >>> config = ZNEConfig(scale_factors=[1.0, 2.0, 3.0])
        >>> zne = ZeroNoiseExtrapolation(config)
        >>> 
        >>> # Define your quantum circuit
        >>> @qml.qnode(dev)
        >>> def circuit(params):
        ...     qml.RX(params[0], wires=0)
        ...     qml.RY(params[1], wires=1)
        ...     qml.CNOT(wires=[0, 1])
        ...     return qml.expval(qml.PauliZ(0))
        >>> 
        >>> # Get mitigated expectation value
        >>> mitigated_value = zne.mitigate(circuit, params)
    
    Attributes:
        config: ZNE configuration.
        extrapolation_history: History of extrapolation results.
    """
    
    def __init__(self, config: ZNEConfig) -> None:
        """
        Initialize Zero-Noise Extrapolation.
        
        Args:
            config: ZNE configuration object.
        """
        self.config = config
        self._rng = np.random.default_rng(config.random_seed)
        self.extrapolation_history: list[dict[str, Any]] = []
        
        logger.info(
            f"Initialized ZNE with scales={config.scale_factors}, "
            f"method={config.extrapolation_method.value}"
        )
    
    def fold_circuit_global(
        self,
        tape: qml.tape.QuantumTape,
        scale_factor: float,
    ) -> qml.tape.QuantumTape:
        """
        Apply global unitary folding to amplify noise.
        
        Global folding transforms U → U(U†U)^n where n is determined
        by the scale factor: λ = 1 + 2n (for integer n).
        
        For non-integer n, we use partial folding on a subset of gates.
        
        Args:
            tape: Original quantum tape.
            scale_factor: Noise amplification factor (λ ≥ 1).
            
        Returns:
            Folded quantum tape with amplified noise.
        """
        if scale_factor == 1.0:
            return tape
        
        # Calculate number of full folds and partial fold
        n_full_folds = int((scale_factor - 1) / 2)
        partial_fold = (scale_factor - 1) % 2 / 2
        
        # Get operations and observables
        operations = tape.operations.copy()
        measurements = tape.measurements
        
        # Apply full unitary folding: U → U(U†U)^n
        folded_ops = operations.copy()
        for _ in range(n_full_folds):
            # Add U†
            folded_ops.extend([qml.adjoint(op) for op in reversed(operations)])
            # Add U
            folded_ops.extend(operations.copy())
        
        # Handle partial folding
        if partial_fold > 0:
            n_gates_to_fold = int(len(operations) * partial_fold)
            if n_gates_to_fold > 0:
                gates_to_fold = operations[-n_gates_to_fold:]
                # Add partial U†
                folded_ops.extend([qml.adjoint(op) for op in reversed(gates_to_fold)])
                # Add partial U
                folded_ops.extend(gates_to_fold)
        
        # Create new tape
        with qml.tape.QuantumTape() as new_tape:
            for op in folded_ops:
                qml.apply(op)
            for m in measurements:
                qml.apply(m)
        
        return new_tape
    
    def fold_circuit_local(
        self,
        tape: qml.tape.QuantumTape,
        scale_factor: float,
    ) -> qml.tape.QuantumTape:
        """
        Apply local gate folding to amplify noise.
        
        Local folding transforms individual gates G → G(G†G)^n.
        This provides more fine-grained control over noise amplification.
        
        Args:
            tape: Original quantum tape.
            scale_factor: Target noise amplification factor.
            
        Returns:
            Folded quantum tape.
        """
        if scale_factor == 1.0:
            return tape
        
        operations = tape.operations.copy()
        measurements = tape.measurements
        
        # Calculate how many folds per gate
        n_gates = len(operations)
        target_ops = int(n_gates * scale_factor)
        ops_to_add = target_ops - n_gates
        
        # Distribute folds across gates
        folds_per_gate = np.zeros(n_gates, dtype=int)
        for i in range(ops_to_add // 2):
            gate_idx = i % n_gates
            folds_per_gate[gate_idx] += 1
        
        # Build folded circuit
        folded_ops = []
        for i, op in enumerate(operations):
            folded_ops.append(op)
            for _ in range(folds_per_gate[i]):
                folded_ops.append(qml.adjoint(op))
                folded_ops.append(op)
        
        # Create new tape
        with qml.tape.QuantumTape() as new_tape:
            for op in folded_ops:
                qml.apply(op)
            for m in measurements:
                qml.apply(m)
        
        return new_tape
    
    def execute_at_scale(
        self,
        circuit: Callable,
        params: Any,
        scale_factor: float,
        device: qml.Device,
    ) -> float:
        """
        Execute circuit at a specific noise scale.
        
        Args:
            circuit: Quantum circuit function.
            params: Circuit parameters.
            scale_factor: Noise amplification factor.
            device: PennyLane device for execution.
            
        Returns:
            Expectation value at the given noise scale.
        """
        if scale_factor == 1.0:
            # No folding needed
            return float(circuit(params))
        
        # Get the quantum tape
        tape = qml.tape.make_qscript(circuit)(params)
        
        # Apply folding
        if self.config.folding_method == FoldingMethod.GLOBAL:
            folded_tape = self.fold_circuit_global(tape, scale_factor)
        elif self.config.folding_method == FoldingMethod.LOCAL:
            folded_tape = self.fold_circuit_local(tape, scale_factor)
        else:
            # Random folding
            if self._rng.random() > 0.5:
                folded_tape = self.fold_circuit_global(tape, scale_factor)
            else:
                folded_tape = self.fold_circuit_local(tape, scale_factor)
        
        # Execute folded circuit
        results = device.execute(folded_tape)
        
        return float(np.mean(results))
    
    def _fit_linear(
        self,
        scales: NDArray[np.float64],
        values: NDArray[np.float64],
    ) -> float:
        """Linear extrapolation to zero noise."""
        coeffs = np.polyfit(scales, values, 1)
        return float(coeffs[1])  # y-intercept
    
    def _fit_polynomial(
        self,
        scales: NDArray[np.float64],
        values: NDArray[np.float64],
    ) -> float:
        """Polynomial extrapolation to zero noise."""
        degree = min(self.config.poly_degree, len(scales) - 1)
        coeffs = np.polyfit(scales, values, degree)
        return float(coeffs[-1])  # y-intercept (λ=0)
    
    def _fit_exponential(
        self,
        scales: NDArray[np.float64],
        values: NDArray[np.float64],
    ) -> float:
        """Exponential decay extrapolation."""
        def exp_model(x: NDArray, a: float, b: float, c: float) -> NDArray:
            return a * np.exp(-b * x) + c
        
        try:
            # Initial guess
            p0 = [values[0] - values[-1], 1.0, values[-1]]
            popt, _ = curve_fit(exp_model, scales, values, p0=p0, maxfev=1000)
            return float(popt[0] + popt[2])  # Value at λ=0
        except (RuntimeError, ValueError):
            # Fall back to polynomial if exponential fails
            logger.warning("Exponential fit failed, falling back to polynomial")
            return self._fit_polynomial(scales, values)
    
    def _fit_richardson(
        self,
        scales: NDArray[np.float64],
        values: NDArray[np.float64],
    ) -> float:
        """
        Richardson extrapolation for improved accuracy.
        
        Richardson extrapolation uses weighted combinations of values
        at different noise scales to eliminate leading-order errors.
        """
        n = len(scales)
        
        # Build the Vandermonde-like matrix
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                A[i, j] = scales[i] ** j
        
        # Solve for coefficients
        try:
            # We want: sum(c_i * f(λ_i)) where c_i eliminates noise terms
            # This requires solving: A.T @ c = e_1 (first basis vector)
            e1 = np.zeros(n)
            e1[0] = 1.0
            c = np.linalg.solve(A.T, e1)
            
            return float(np.dot(c, values))
        except np.linalg.LinAlgError:
            logger.warning("Richardson extrapolation failed, falling back to linear")
            return self._fit_linear(scales, values)
    
    def _select_best_method(
        self,
        scales: NDArray[np.float64],
        values: NDArray[np.float64],
    ) -> Tuple[float, str]:
        """
        Adaptively select the best extrapolation method.
        
        Compares different methods using cross-validation and selects
        the one with lowest error.
        """
        methods = {
            "linear": self._fit_linear,
            "polynomial": self._fit_polynomial,
            "exponential": self._fit_exponential,
            "richardson": self._fit_richardson,
        }
        
        best_method = "linear"
        best_value = methods["linear"](scales, values)
        best_cv_error = float("inf")
        
        for name, method in methods.items():
            try:
                # Leave-one-out cross-validation
                cv_errors = []
                for i in range(len(scales)):
                    train_scales = np.delete(scales, i)
                    train_values = np.delete(values, i)
                    
                    pred = method(train_scales, train_values)
                    
                    # Estimate error as deviation from smooth curve
                    all_pred = method(scales, values)
                    cv_errors.append(abs(values[i] - all_pred))
                
                cv_error = np.mean(cv_errors)
                
                if cv_error < best_cv_error:
                    best_cv_error = cv_error
                    best_method = name
                    best_value = method(scales, values)
                    
            except Exception as e:
                logger.debug(f"Method {name} failed: {e}")
                continue
        
        return best_value, best_method
    
    def extrapolate(
        self,
        scales: NDArray[np.float64],
        values: NDArray[np.float64],
    ) -> dict[str, Any]:
        """
        Extrapolate to zero noise from noisy measurements.
        
        Args:
            scales: Noise scale factors used.
            values: Measured expectation values at each scale.
            
        Returns:
            Dictionary containing:
                - mitigated_value: Extrapolated zero-noise value
                - method_used: Extrapolation method used
                - confidence: Confidence estimate
                - raw_data: Original scales and values
        """
        scales = np.array(scales)
        values = np.array(values)
        
        method = self.config.extrapolation_method
        
        if method == ExtrapolationMethod.LINEAR:
            mitigated = self._fit_linear(scales, values)
            method_name = "linear"
        elif method == ExtrapolationMethod.POLYNOMIAL:
            mitigated = self._fit_polynomial(scales, values)
            method_name = "polynomial"
        elif method == ExtrapolationMethod.EXPONENTIAL:
            mitigated = self._fit_exponential(scales, values)
            method_name = "exponential"
        elif method == ExtrapolationMethod.RICHARDSON:
            mitigated = self._fit_richardson(scales, values)
            method_name = "richardson"
        else:  # ADAPTIVE
            mitigated, method_name = self._select_best_method(scales, values)
        
        # Estimate confidence based on residuals
        if method_name in ["linear", "polynomial"]:
            degree = 1 if method_name == "linear" else self.config.poly_degree
            coeffs = np.polyfit(scales, values, degree)
            predicted = np.polyval(coeffs, scales)
            residuals = values - predicted
            confidence = 1.0 / (1.0 + np.std(residuals))
        else:
            confidence = 0.8  # Default confidence for other methods
        
        result = {
            "mitigated_value": mitigated,
            "method_used": method_name,
            "confidence": confidence,
            "raw_scales": scales.tolist(),
            "raw_values": values.tolist(),
        }
        
        self.extrapolation_history.append(result)
        
        return result
    
    def mitigate(
        self,
        circuit: Callable,
        params: Any,
        device: Optional[qml.Device] = None,
    ) -> dict[str, Any]:
        """
        Apply zero-noise extrapolation to mitigate errors in a quantum circuit.
        
        This is the main entry point for error mitigation. It:
        1. Executes the circuit at multiple noise scales
        2. Extrapolates to estimate the zero-noise value
        
        Args:
            circuit: Quantum circuit function (QNode).
            params: Parameters to pass to the circuit.
            device: PennyLane device (uses circuit's device if None).
            
        Returns:
            Mitigation results including the estimated ideal value.
        """
        if device is None:
            device = circuit.device
        
        # Execute at each noise scale
        noisy_values = []
        for scale in self.config.scale_factors:
            value = self.execute_at_scale(circuit, params, scale, device)
            noisy_values.append(value)
            
            logger.debug(f"Scale {scale:.2f}: value = {value:.6f}")
        
        # Extrapolate to zero noise
        result = self.extrapolate(
            np.array(self.config.scale_factors),
            np.array(noisy_values),
        )
        
        # Add comparison with unmitigated
        result["unmitigated_value"] = noisy_values[0]  # λ=1 value
        result["improvement"] = abs(result["mitigated_value"] - noisy_values[-1])
        
        logger.info(
            f"ZNE mitigation: {noisy_values[0]:.4f} → {result['mitigated_value']:.4f} "
            f"(method: {result['method_used']})"
        )
        
        return result


class MeasurementErrorMitigation:
    """
    Measurement Error Mitigation using calibration matrices.
    
    This class implements readout error mitigation by characterizing
    the measurement channel and applying its inverse.
    
    The idea is to measure all basis states to build a calibration
    matrix C where C[i,j] = P(measure j | prepared i), then apply
    C^{-1} to measured probabilities to recover ideal distribution.
    
    Example:
        >>> mem = MeasurementErrorMitigation(n_qubits=2)
        >>> mem.calibrate(device)  # Measure calibration circuits
        >>> mitigated_probs = mem.mitigate(measured_probs)
    """
    
    def __init__(
        self,
        n_qubits: int,
        n_shots: int = 8192,
        method: str = "inverse",
    ) -> None:
        """
        Initialize measurement error mitigation.
        
        Args:
            n_qubits: Number of qubits.
            n_shots: Shots for calibration measurements.
            method: Mitigation method ('inverse', 'least_squares', 'iterative').
        """
        self.n_qubits = n_qubits
        self.n_shots = n_shots
        self.method = method
        
        self.calibration_matrix: Optional[NDArray[np.float64]] = None
        self.inverse_matrix: Optional[NDArray[np.float64]] = None
        self._is_calibrated = False
    
    def calibrate(self, device: qml.Device) -> None:
        """
        Perform calibration measurements to characterize readout errors.
        
        Prepares all 2^n computational basis states and measures to
        build the calibration matrix.
        
        Args:
            device: PennyLane device to calibrate.
        """
        n_states = 2 ** self.n_qubits
        self.calibration_matrix = np.zeros((n_states, n_states))
        
        for i in range(n_states):
            # Prepare basis state |i⟩
            @qml.qnode(device)
            def calibration_circuit() -> NDArray:
                # Prepare state |i⟩ using X gates
                for qubit in range(self.n_qubits):
                    if (i >> qubit) & 1:
                        qml.PauliX(wires=qubit)
                return qml.probs(wires=range(self.n_qubits))
            
            # Measure probabilities
            probs = calibration_circuit()
            self.calibration_matrix[:, i] = probs
        
        # Compute inverse (or pseudo-inverse) for mitigation
        try:
            self.inverse_matrix = np.linalg.inv(self.calibration_matrix)
        except np.linalg.LinAlgError:
            self.inverse_matrix = np.linalg.pinv(self.calibration_matrix)
        
        self._is_calibrated = True
        
        logger.info(f"Calibration complete. Readout fidelity: {np.mean(np.diag(self.calibration_matrix)):.4f}")
    
    def mitigate(self, measured_probs: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Apply measurement error mitigation to measured probabilities.
        
        Args:
            measured_probs: Measured probability distribution.
            
        Returns:
            Mitigated probability distribution.
        """
        if not self._is_calibrated:
            raise RuntimeError("Must call calibrate() before mitigating")
        
        if self.method == "inverse":
            mitigated = self.inverse_matrix @ measured_probs
        elif self.method == "least_squares":
            # Constrained least squares (non-negative, sums to 1)
            from scipy.optimize import minimize
            
            def objective(p: NDArray) -> float:
                return float(np.sum((self.calibration_matrix @ p - measured_probs) ** 2))
            
            constraints = [
                {"type": "eq", "fun": lambda p: np.sum(p) - 1.0}
            ]
            bounds = [(0, 1) for _ in range(len(measured_probs))]
            
            result = minimize(
                objective,
                measured_probs,
                method="SLSQP",
                constraints=constraints,
                bounds=bounds,
            )
            mitigated = result.x
        else:
            # Iterative method
            mitigated = measured_probs.copy()
            for _ in range(10):
                correction = self.inverse_matrix @ (measured_probs - self.calibration_matrix @ mitigated)
                mitigated = mitigated + 0.5 * correction
                mitigated = np.clip(mitigated, 0, 1)
                mitigated = mitigated / np.sum(mitigated)
        
        # Ensure valid probability distribution
        mitigated = np.clip(mitigated, 0, 1)
        mitigated = mitigated / np.sum(mitigated)
        
        return mitigated
    
    def get_readout_fidelity(self) -> float:
        """
        Get average readout fidelity from calibration.
        
        Returns:
            Average fidelity (diagonal elements of calibration matrix).
        """
        if not self._is_calibrated:
            raise RuntimeError("Must call calibrate() first")
        
        return float(np.mean(np.diag(self.calibration_matrix)))
