
from utils.qkd_tools import *

# --- Decomposing Qutrit Gates with Universal Gate Set (Setting Global Variables) ---

H_unitary = QuditHGate(d=3)._unitary_()
R_matrices_H, Phi_Qutrit_H = reckon_decompose_unitary(H_unitary) #decomposing hadamard matrices

Qutrit_0swap1_Gate_unitary = Qutrit_0swap1_Gate(d=3)._unitary_()
R_matrices_Qutrit_0swap1, Phi_Qutrit_0swap1 = reckon_decompose_unitary(Qutrit_0swap1_Gate_unitary) #decomposing Qutrit_0swap1_Gate matrices

Qutrit_0swap2_Gate_unitary = Qutrit_0swap2_Gate(d=3)._unitary_()
R_matrices_Qutrit_0swap2, Phi_Qutrit_0swap2 = reckon_decompose_unitary(Qutrit_0swap2_Gate_unitary) #decomposing Qutrit_0swap2_Gate matrices

# --- Alice's Side (Using Operations Constructible from S) ---

def alice_prepares_bb84_state(bit: int, basis_choice: str): # Corrected function name to match call
    """
    Alice prepares a qubit state for BB84 using operations constructible from S.
    (Conceptually, these operations are built from PHASE_1 and T_elements).

    Args:
        bit (int): The bit Alice wants to send (0 or 1).
        basis_choice (str): Alice's chosen basis ('rectilinear' or 'diagonal').

    Returns:
        cirq.Circuit: The quantum circuit for Alice's action.
        cirq.LineQubit: The qubit Alice uses.
        str: The basis Alice chose.  <-- ADDED THIS RETURN VALUE
        int: The bit Alice encoded.
    """

    qubit = cirq.LineQid(0, dimension=3)
    
    # QKD Circuit using Traditional Qutrit Gates

    circuit_traditional_gates = cirq.Circuit()

    # For rectilinear basis:
    # choice 0 -> |0> (No operation needed)
    # choice 1 -> |1> (apply Qutrit_0swap1_Gate)
    # choice 2 -> |2> (apply Qutrit_0swap2_Gate)    
    if basis_choice == 'rectilinear':
        if bit == 1:
            circuit_traditional_gates.append(Qutrit_0swap1_Gate(d=3)(qubit))
        if bit == 2:
            circuit_traditional_gates.append(Qutrit_0swap2_Gate(d=3)(qubit))

    # For diagonal basis:
    # Bit 0 -> (|0> + |1>)/sqrt(2) (Equivalent to Hadamard)
    # Bit 1 -> (|0> - |1>)/sqrt(2) (Equivalent to X then Hadamard)
    elif basis_choice == 'diagonal':
        if bit == 0:
            circuit_traditional_gates.append(QuditHGate(d=3)(qubit))
            circuit_traditional_gates.append(QuditHGate(d=3)(qubit))
            circuit_traditional_gates.append(QuditHGate(d=3)(qubit))
        if bit == 1:
            circuit_traditional_gates.append(Qutrit_0swap1_Gate(d=3)(qubit))
            circuit_traditional_gates.append(QuditHGate(d=3)(qubit))
            circuit_traditional_gates.append(QuditHGate(d=3)(qubit))
            circuit_traditional_gates.append(QuditHGate(d=3)(qubit))
        if bit == 2:
            circuit_traditional_gates.append(Qutrit_0swap2_Gate(d=3)(qubit))
            circuit_traditional_gates.append(QuditHGate(d=3)(qubit))
            circuit_traditional_gates.append(QuditHGate(d=3)(qubit))
            circuit_traditional_gates.append(QuditHGate(d=3)(qubit))
    else:
        raise ValueError("Invalid basis choice for Alice.")


    # QKD Circuit using Universal Qutrit Gates

    circuit_universal_gates = cirq.Circuit()
    
    # The order for reconstruction is R_L, R_{L-1}, ..., R_1
    # So R_matrices_H[i] is R_{i+1} in the decomposition M = R1 @ ... @ RL @ Phi.
    repeated_h_ops_phi = [ArbitraryGate(d=3,matrix=Phi_Qutrit_H)(qubit)]
    repeated_h_ops_R = [op for M in R_matrices_H[::-1] for op in [ArbitraryGate(d=3,matrix=M)(qubit)]]
    
    repeated_Qutrit_0swap1_Gate_ops_phi = [ArbitraryGate(d=3,matrix=Phi_Qutrit_0swap1)(qubit)]
    repeated_Qutrit_0swap1_Gate_ops_R = [op for M in R_matrices_Qutrit_0swap1[::-1] for op in [ArbitraryGate(d=3,matrix=M)(qubit)]]
    
    repeated_Qutrit_0swap2_Gate_ops_phi = [ArbitraryGate(d=3,matrix=Phi_Qutrit_0swap2)(qubit)]
    repeated_Qutrit_0swap2_Gate_ops_R = [op for M in R_matrices_Qutrit_0swap2[::-1] for op in [ArbitraryGate(d=3,matrix=M)(qubit)]]  

    # For rectilinear basis:
    # choice 0 -> |0> (No operation needed)
    # choice 1 -> |1> (apply Qutrit_0swap1_Gate)
    # choice 2 -> |2> (apply Qutrit_0swap2_Gate)    
    if basis_choice == 'rectilinear':
        if bit == 1:
            circuit_universal_gates.append(repeated_Qutrit_0swap1_Gate_ops_phi)
            circuit_universal_gates.append(repeated_Qutrit_0swap1_Gate_ops_R)
        if bit == 2:
            circuit_universal_gates.append(repeated_Qutrit_0swap2_Gate_ops_phi)
            circuit_universal_gates.append(repeated_Qutrit_0swap2_Gate_ops_R)
    # For diagonal basis:
    # Bit 0 -> (|0> + |1>)/sqrt(2) (Equivalent to Hadamard)
    # Bit 1 -> (|0> - |1>)/sqrt(2) (Equivalent to X then Hadamard)
    elif basis_choice == 'diagonal':
        if bit == 0:
            circuit_universal_gates.append(repeated_h_ops_phi)
            circuit_universal_gates.append(repeated_h_ops_R)
            circuit_universal_gates.append(repeated_h_ops_phi)
            circuit_universal_gates.append(repeated_h_ops_R)
            circuit_universal_gates.append(repeated_h_ops_phi)
            circuit_universal_gates.append(repeated_h_ops_R)
        if bit == 1:
            circuit_universal_gates.append(repeated_Qutrit_0swap1_Gate_ops_phi)
            circuit_universal_gates.append(repeated_Qutrit_0swap1_Gate_ops_R)
            circuit_universal_gates.append(repeated_h_ops_phi)
            circuit_universal_gates.append(repeated_h_ops_R)
            circuit_universal_gates.append(repeated_h_ops_phi)
            circuit_universal_gates.append(repeated_h_ops_R)
            circuit_universal_gates.append(repeated_h_ops_phi)
            circuit_universal_gates.append(repeated_h_ops_R)
        if bit == 2:
            circuit_universal_gates.append(repeated_Qutrit_0swap2_Gate_ops_phi)
            circuit_universal_gates.append(repeated_Qutrit_0swap2_Gate_ops_R)
            circuit_universal_gates.append(repeated_h_ops_phi)
            circuit_universal_gates.append(repeated_h_ops_R)
            circuit_universal_gates.append(repeated_h_ops_phi)
            circuit_universal_gates.append(repeated_h_ops_R)
            circuit_universal_gates.append(repeated_h_ops_phi)
            circuit_universal_gates.append(repeated_h_ops_R)
    else:
        raise ValueError("Invalid basis choice for Alice.")

    # Return the basis choice made by Alice
    return circuit_traditional_gates, circuit_universal_gates, qubit, basis_choice, bit 

# --- Bob's Side (Using Operations Constructible from S for Measurement) ---
def bob_measures_state(basis_choice: str, qubit: cirq.LineQubit): 
    """
    Bob measures in his chosen basis using operations constructible from S.
    Measurement in diagonal basis is achieved by applying Hadamard before measuring.

    Args:
        basis_choice (str): Bob's chosen basis ('rectilinear' or 'diagonal').
        qubit (cirq.LineQubit): The qubit Bob receives from Alice.

    Returns:
        cirq.Circuit: The quantum circuit for Bob's measurement.
        cirq.LineQubit: The qubit Bob measured.
        str: The basis Bob chose.  <-- ADDED THIS RETURN VALUE
    """

    measurement_key = f'{basis_choice}_measurement' # Unique key for the measurement
    
    # --- Operations constructible from S for measurement ---
    # To measure in a specific basis, we transform the state such that
    # the desired basis states align with the computational (rectilinear) basis,
    # and then perform a standard rectilinear measurement.

    # QKD Circuit using Traditional Qutrit Gates

    circuit_traditional_gates = cirq.Circuit()

    if basis_choice == 'rectilinear':
        # Measure in the rectilinear (Z) basis. No extra gates needed as
        # Cirq's measure is in the Z-basis.
        pass

    elif basis_choice == 'diagonal':
        # To measure in the diagonal basis, we need to apply an operation
        # that maps the diagonal basis states to the rectilinear basis states.
        # The Hadamard gate (H) does this: H|(+)> = |0>, H|(-)> = |1>.
        # So, apply Hadamard before measurement. H is constructible from S.
        circuit_traditional_gates.append(QuditHGate(d=3)(qubit))

    else:
        raise ValueError("Invalid basis choice for Bob.")

    # Perform the measurement in the (now aligned) rectilinear basis.
    circuit_traditional_gates.append(cirq.measure(qubit, key=measurement_key))

    # QKD Circuit using Universal Qutrit Gates

    circuit_universal_gates = cirq.Circuit()
    
    # The order for reconstruction is R_L, R_{L-1}, ..., R_1
    # So R_matrices_H[i] is R_{i+1} in the decomposition M = R1 @ ... @ RL @ Phi.
    repeated_h_ops_phi = [ArbitraryGate(d=3,matrix=Phi_Qutrit_H)(qubit)]
    repeated_h_ops_R = [op for M in R_matrices_H[::-1] for op in [ArbitraryGate(d=3,matrix=M)(qubit)]]
    
    if basis_choice == 'rectilinear':
        # Measure in the rectilinear (Z) basis. No extra gates needed as
        # Cirq's measure is in the Z-basis.
        pass

    elif basis_choice == 'diagonal':
        # To measure in the diagonal basis, we need to apply an operation
        # that maps the diagonal basis states to the rectilinear basis states.
        # The Hadamard gate (H) does this: H|(+)> = |0>, H|(-)> = |1>.
        # So, apply Hadamard before measurement. H is constructible from S.
        circuit_universal_gates.append(repeated_h_ops_phi)
        circuit_universal_gates.append(repeated_h_ops_R)
    else:
        raise ValueError("Invalid basis choice for Bob.")

    circuit_universal_gates.append(cirq.measure(qubit, key=measurement_key))

    return circuit_traditional_gates, circuit_universal_gates, qubit, basis_choice 

