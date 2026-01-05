
# Import Libraries
import cirq
import cmath
import numpy as np
from utils.tools import *

# ... qutrit based state generation operators ...
class Qutrit_0swap2_Gate(cirq.Gate):
    """
    A custom gate that swaps the |0> and |2> states of a qutrit,
    leaving |1> unchanged.
    """
    def __init__(self, d: int):
        self.name = "Qutrit_0swap2_Gate"

    def _qid_shape_(self):
        return (3,) # It operates on a single qutrit

    def _unitary_(self):
        QUTRIT_SWAP_MATRIX = np.array([
            [0, 0, 1],  # First row: maps to |0> (from |2>)
            [0, 1, 0],  # Second row: maps to |1> (from |1>)
            [1, 0, 0]   # Third row: maps to |2> (from |0>)
        ])
        # Return the matrix for this gate
        return QUTRIT_SWAP_MATRIX

    def _circuit_diagram_info_(self, args):
        return self.name # Name for diagram
    
class Qutrit_0swap1_Gate(cirq.Gate):
    """
    A custom gate that swaps the |0> and |1> states of a qutrit,
    leaving |2> unchanged.
    """
    def __init__(self, d: int):
        self.name = "Qutrit_0swap1_Gate"

    def _qid_shape_(self):
        return (3,) # It operates on a single qutrit

    def _unitary_(self):
        QUTRIT_SWAP_MATRIX = np.array([
            [0, 1, 0], # First row: maps to |1> (from |0>)
            [1, 0, 0], # Second row: maps to |0> (from |1>)
            [0, 0, 1] # Third row: maps to |2> (from |2>)
        ])
        # Return the matrix for this gate
        return QUTRIT_SWAP_MATRIX

    def _circuit_diagram_info_(self, args):
        return self.name # Name for diagram
    
class QuditHGate(cirq.Gate):
    """A generalized Hadamard-like gate for qudits.

    This gate acts on a single d-level system (qudit).
    It implements a specific unitary transformation, which in the original
    example was a normalized DFT matrix. For this script's purpose,
    we assume it's a component for setting up the search space.

    Args:
        d (int): The number of levels in the qudit system.
                 This dictates the dimension of the Hilbert space.
    """
    def __init__(self, d: int):
        """Initializes the QuditHGate for a qudit of 'd' levels."""
        super().__init__()
        if not isinstance(d, int) or d <= 0:
            raise ValueError("The number of levels 'd' must be a positive integer.")
        self._d = d
        self.name = f'Qu{d}H' # Shorter name for circuit diagrams
        # This gate acts on a single qudit.
        self.num_qubits = 1

    def _qid_shape_(self):
        """Returns the shape of the qudit this gate acts on."""
        return (self._d,)

    def _unitary_(self):
        """Returns the unitary matrix for the gate.

        This specific implementation generates a normalized Discrete Fourier
        Transform (DFT) matrix of size d x d. This is a common component
        in quantum algorithms and signal processing.
        """
        d = self._d
        # Create indices i and j for the matrix elements.
        i, j = np.meshgrid(np.arange(d), np.arange(d))
        # The exponent determines the phase shift, based on (i * j) mod d.
        exponent = (i * j) % d
        # Calculate complex roots of unity and normalize.
        roots_of_unity = np.exp(2 * cmath.pi * 1j * exponent / d)
        normalized_matrix = roots_of_unity / np.sqrt(d)
        return normalized_matrix

    def _circuit_diagram_info_(self, args):
        """Provides information for circuit diagrams."""
        return f'Qu{self._d}H'
    
# # --- Universal Decomposition ---

# # Reck's Decomposition method to decomposes an unitary matrix into a product of R_i matrices and a diagonal phase matrix.
# def reckon_decompose_unitary(M):
#     """
#     Decomposes an N x N unitary matrix M into a product of R_k matrices and a diagonal
#     phase matrix Phi, such that M = R_1 @ R_2 @ ... @ R_L @ Phi.

#     This algorithm is based on the Reck's decomposition method, which uses a sequence
#     of Givens rotations to transform the unitary matrix into a diagonal matrix.

#     Args:
#         M (np.ndarray): An N x N complex numpy array representing a unitary matrix.

#     Returns:
#         tuple: A tuple containing:
#             - R_matrices (list): A list of N x N numpy arrays, where each array is
#                                  an R_k matrix (identity matrix with a 2x2 unitary
#                                  block). The matrices are ordered such that
#                                  M = R_matrices[0] @ R_matrices[1] @ ... @ R_matrices[-1] @ Phi.
#             - Phi (np.ndarray): An N x N diagonal numpy array representing the
#                                  final phase shift matrix.

#     Raises:
#         ValueError: If the input matrix is not square or is not unitary.
#     """
#     N = M.shape[0]
#     if M.shape[1] != N:
#         raise ValueError("Input matrix must be square.")
    
#     # Check if M is unitary (M @ M.conj().T should be identity)
#     if not np.allclose(M @ M.conj().T, np.eye(N)):
#         raise ValueError("Input matrix is not unitary.")

#     R_matrices = []
#     U_current = M.astype(complex) # Ensure complex dtype for computations

#     # Iterate through columns from left to right (j)
#     for j in range(N - 1):
#         # Iterate through rows from bottom up to j+1 (i)
#         # to zero out elements below the diagonal in column j
#         for i in range(N - 1, j, -1):
#             u = U_current[j, j]
#             v = U_current[i, j]

#             # If the element to zero is already very small, skip
#             if np.isclose(v, 0.0):
#                 continue

#             r = np.sqrt(np.abs(u)**2 + np.abs(v)**2)

#             # Construct the 2x2 Givens rotation block G_block
#             # G_block @ [[u],[v]] = [[r],[0]]
#             c = u.conjugate() / r
#             s = v.conjugate() / r
            
#             # The 2x2 block that zeros v when applied to [[u],[v]]
#             G_block = np.array([[c, s],
#                                 [-s.conjugate(), c.conjugate()]], dtype=complex)

#             # Construct the N x N R_inv_matrix (Givens rotation matrix)
#             R_inv_matrix = np.eye(N, dtype=complex)
#             R_inv_matrix[np.ix_([j, i], [j, i])] = G_block

#             # Apply the rotation to U_current
#             U_current = R_inv_matrix @ U_current

#             # Store the R_k matrix (which is the inverse of R_inv_matrix, i.e., its conjugate transpose)
#             R_matrices.append(R_inv_matrix.conj().T)

#     Phi = U_current
    
#     # Round small values to zero for cleaner diagonal matrix
#     # and ensure phases are correct
#     for k in range(N):
#         if not np.isclose(np.abs(Phi[k, k]), 1.0):
#              # This should not happen if M is unitary and calculations are precise
#             print(f"Warning: Diagonal element Phi[{k},{k}] has magnitude {np.abs(Phi[k,k])} != 1.")
#         # Make off-diagonal elements exactly zero if close to zero
#         for l in range(N):
#             if k != l and np.isclose(Phi[k, l], 0.0):
#                 Phi[k, l] = 0.0

#     # Decomposition of Phi Matrix into product of elements from T_elements and Phase1
#     # Elements of T_elements are finally appended under R_matrices   
#     Phi_balance = np.eye(N, dtype=complex)
#     if np.linalg.det(Phi) != 1: 
#         v = 1
#         for i in range(len(Phi)): 
#             v = v * Phi[i][i]
#         phi0 = v.conjugate()
#         Phi_balance[0][0] = phi0
#         for i in range(1, len(Phi)): Phi_balance[i][i] = Phi[i][i]
#         Phi = np.eye(N, dtype=complex)
#         Phi[0][0] = v
#         R_matrices.append(Phi_balance)

#     return R_matrices, Phi

