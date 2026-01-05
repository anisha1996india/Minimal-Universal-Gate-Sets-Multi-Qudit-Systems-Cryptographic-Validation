
from utils.grover_tools import *

def grover_circuit_with_traditional_gates(q0, q1):

    # --- Circuit for constructing Grover's circuit with actual matrices --- #

    circuit = cirq.Circuit([
        QuditHGate(d=4)(q0),
        QuditHGate(d=4)(q1),
        UfGate()(q0,q1),
        QuditHGate(d=4)(q0),
        QuditHGate(d=4)(q1),
        U0Gate()(q0,q1),
        QuditHGate(d=4)(q0),
        QuditHGate(d=4)(q1),
        cirq.measure(q0, q1)
    ])

    return circuit


def grover_circuit_with_universal_gates(q0, q1):

    # --- Circuit for constructing Grover's circuit with decomposed matrices ---

    H_unitary = QuditHGate(d=4)._unitary_()
    M1, M2 = reckon_decompose_unitary(H_unitary) #decomposing hadamard matrices

    # The order for reconstruction is R_L, R_{L-1}, ..., R_1
    # So R_matrices_H[i] is R_{i+1} in the decomposition M = R1 @ ... @ RL @ Phi.
    repeated_h_ops_phi = [ArbitraryGate(d=4,matrix=M2)(q0), ArbitraryGate(d=4,matrix=M2)(q1)]
    repeated_h_ops_R = [op for M in M1[::-1] for op in [ArbitraryGate(d=4,matrix=M)(q0), ArbitraryGate(d=4,matrix=M)(q1)]]

    all_operations = (
        repeated_h_ops_phi+
        repeated_h_ops_R+
        [UfGate()(q0, q1)] +
        repeated_h_ops_phi+
        repeated_h_ops_R+
        [U0Gate()(q0, q1)] +
        repeated_h_ops_phi+
        repeated_h_ops_R+
        [cirq.measure(q0, q1)]
    )

    circuit = cirq.Circuit(all_operations)

    return circuit
