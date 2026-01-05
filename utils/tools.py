
# Import Libraries
import cirq
import random
import numpy as np

# --- To define arbitrary operator from arbitrary matrix ---

# The ArbitraryGate is used to apply the decomposed matrices from Reck's method.
# It's a generic gate that can take any unitary matrix.
class ArbitraryGate(cirq.Gate):
    """A generic gate that applies a given arbitrary unitary matrix.

    This is useful for applying decomposed gates (e.g., from Reck's algorithm).

    Args:
        d (int): The dimension of the qudit system this gate operates on.
                 Note: For multi-qubit/qudit gates, this parameter might need
                 adjustment or a different approach to specify the total system.
                 Here, it's mainly for diagram labeling.
        matrix (np.ndarray): The N x N unitary matrix to apply.
    """
    def __init__(self, d: int, matrix: np.ndarray):
        super().__init__()
        if not isinstance(d, int) or d <= 0:
            raise ValueError("The number of levels 'd' must be a positive integer.")
        if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] != d:
             # This check is done to carry out data sanity check on the matrix dimension
            pass 

        self._d = d
        self._matrix = matrix
        self.name = f'Arb{d}' # Generic name for arbitrary gates
        self.num_qubits = 1 # QuditHGate acts on 1 qudit, so its decomposition is also on 1 qudit

    def _qid_shape_(self):
        """Returns the shape of the qudit this gate acts on."""
        return (self._d,) # Use the stored instance value of d

    def _unitary_(self):
        """Returns the unitary matrix for the gate."""
        d = self._d  # Retrieve the d value for this specific instance
        matrix = self._matrix
        return np.array(matrix)

    def _circuit_diagram_info_(self, args):
        """Provides information for circuit diagrams."""
        return f'Qu{self._d}M'
    
# --- Universal Decomposition ---

# Reck's Decomposition method to decomposes an unitary matrix into a product of R_i matrices and a diagonal phase matrix.
def reckon_decompose_unitary(M):
    """
    Decomposes an N x N unitary matrix M into a product of R_k matrices and a diagonal
    phase matrix Phi, such that M = R_1 @ R_2 @ ... @ R_L @ Phi.

    This algorithm is based on the Reck's decomposition method, which uses a sequence
    of Givens rotations to transform the unitary matrix into a diagonal matrix.

    Args:
        M (np.ndarray): An N x N complex numpy array representing a unitary matrix.

    Returns:
        tuple: A tuple containing:
            - R_matrices (list): A list of N x N numpy arrays, where each array is
                                 an R_k matrix (identity matrix with a 2x2 unitary
                                 block). The matrices are ordered such that
                                 M = R_matrices[0] @ R_matrices[1] @ ... @ R_matrices[-1] @ Phi.
            - Phi (np.ndarray): An N x N diagonal numpy array representing the
                                 final phase shift matrix.

    Raises:
        ValueError: If the input matrix is not square or is not unitary.
    """
    N = M.shape[0]
    if M.shape[1] != N:
        raise ValueError("Input matrix must be square.")
    
    # Check if M is unitary (M @ M.conj().T should be identity)
    if not np.allclose(M @ M.conj().T, np.eye(N)):
        raise ValueError("Input matrix is not unitary.")

    R_matrices = []
    U_current = M.astype(complex) # Ensure complex dtype for computations

    # Iterate through columns from left to right (j)
    for j in range(N - 1):
        # Iterate through rows from bottom up to j+1 (i)
        # to zero out elements below the diagonal in column j
        for i in range(N - 1, j, -1):
            u = U_current[j, j]
            v = U_current[i, j]

            # If the element to zero is already very small, skip
            if np.isclose(v, 0.0):
                continue

            r = np.sqrt(np.abs(u)**2 + np.abs(v)**2)

            # Construct the 2x2 Givens rotation block G_block
            # G_block @ [[u],[v]] = [[r],[0]]
            c = u.conjugate() / r
            s = v.conjugate() / r
            
            # The 2x2 block that zeros v when applied to [[u],[v]]
            G_block = np.array([[c, s],
                                [-s.conjugate(), c.conjugate()]], dtype=complex)

            # Construct the N x N R_inv_matrix (Givens rotation matrix)
            R_inv_matrix = np.eye(N, dtype=complex)
            R_inv_matrix[np.ix_([j, i], [j, i])] = G_block

            # Apply the rotation to U_current
            U_current = R_inv_matrix @ U_current

            # Store the R_k matrix (which is the inverse of R_inv_matrix, i.e., its conjugate transpose)
            R_matrices.append(R_inv_matrix.conj().T)

    Phi = U_current
    
    # Round small values to zero for cleaner diagonal matrix
    # and ensure phases are correct
    for k in range(N):
        if not np.isclose(np.abs(Phi[k, k]), 1.0):
             # This should not happen if M is unitary and calculations are precise
            print(f"Warning: Diagonal element Phi[{k},{k}] has magnitude {np.abs(Phi[k,k])} != 1.")
        # Make off-diagonal elements exactly zero if close to zero
        for l in range(N):
            if k != l and np.isclose(Phi[k, l], 0.0):
                Phi[k, l] = 0.0

    # Decomposition of Phi Matrix into product of elements from T_elements and Phase1
    # Elements of T_elements are finally appended under R_matrices   
    Phi_balance = np.eye(N, dtype=complex)
    if np.linalg.det(Phi) != 1: 
        v = 1
        for i in range(len(Phi)): 
            v = v * Phi[i][i]
        phi0 = v.conjugate()
        Phi_balance[0][0] = phi0
        for i in range(1, len(Phi)): Phi_balance[i][i] = Phi[i][i]
        Phi = np.eye(N, dtype=complex)
        Phi[0][0] = v
        R_matrices.append(Phi_balance)

    return R_matrices, Phi


# Li-Roberts-Yin's Decomposition of unitary matrices and quantum gates
def li_roberts_yin_decompose(U):
    """
    Simplified Li Roberts Yin decomposition:
    Break unitary matrix into fully-controlled single-qubit gates.
    Returns: list of (target_qubit, controls, U2_matrix)
    """
    U = np.array(U, dtype=complex)
    n = int(np.log2(U.shape[0]))
    gates = []

    # Work column by column
    for col in range(U.shape[0]):
        for target in range(n):
            # For each control pattern (other qubits fixed)
            for ctrl_pattern in range(2**(n-1)):
                # Build indices for pair differing at 'target'
                bits = [(ctrl_pattern >> i) & 1 for i in range(n-1)]
                controls = {}
                idx0_bits, idx1_bits = [], []
                bitpos = 0
                for q in range(n):
                    if q == target:
                        idx0_bits.append(0)
                        idx1_bits.append(1)
                    else:
                        val = bits[bitpos]
                        controls[q] = val
                        idx0_bits.append(val)
                        idx1_bits.append(val)
                        bitpos += 1
                idx0 = sum(b << i for i, b in enumerate(idx0_bits))
                idx1 = sum(b << i for i, b in enumerate(idx1_bits))

                a, b = U[idx0, col], U[idx1, col]
                if abs(b) < 1e-12:  # nothing to zero
                    continue

                # Build 2x2 Givens rotation
                r = np.sqrt(abs(a)**2 + abs(b)**2)
                c, s = a/r, b/r
                G = np.array([[c, s], [-np.conj(s), np.conj(c)]], dtype=complex)

                # Apply to U
                U[[idx0, idx1], :] = G @ U[[idx0, idx1], :]

                # Record gate
                gates.append((target, controls, G))

    return gates


# --- Matrix Tools ---

def generate_random_unitary_matrix(n):
    """
    Generates an n x n random unitary complex matrix using the QR decomposition
    of a Ginibre random matrix.
    """
    # Create an n x n matrix with standard complex Gaussian entries (Ginibre matrix)
    # Each real and imaginary part drawn from N(0, 1/2)
    # np.random.randn generates samples from a standard normal distribution N(0, 1)
    A = (np.random.randn(n, n) + 1j * np.random.randn(n, n)) / np.sqrt(2.0)

    # Perform QR decomposition
    Q, R = np.linalg.qr(A)

    # The matrix Q is unitary
    return Q


def generate_sorted_random_numbers(count=5, start=2, end=20):
    """
    Generate 'count' unique random integers between 'start' and 'end' (inclusive),
    sorted in increasing order.
    """
    numbers = random.sample(range(start, end + 1), count)
    return sorted(numbers)


# ------------------------------
# Core matrix similarity metrics
# ------------------------------

def operator_fidelity(U_target: np.ndarray, U_approx: np.ndarray) -> float:
    """
    Operator fidelity for (ideally unitary) matrices:
        F = (1/m) * |Tr(U_target^\\dagger U_approx)|
    Returns a value in [0, 1] when both are unitary.
    """
    m = U_target.shape[0]
    return float(np.abs(np.trace(U_target.conj().T @ U_approx)) / m)


def trace_similarity(U_target: np.ndarray, U_approx: np.ndarray) -> float:
    """
    Normalized trace overlap ('trace similarity'):
        S = |Tr(U_target^\\dagger U_approx)| / sqrt(Tr(U_target^\\dagger U_target) * Tr(U_approx^\\dagger U_approx))
    For unitaries, Tr(U^\\dagger U) = m, hence S == operator_fidelity.
    """
    num = np.abs(np.trace(U_target.conj().T @ U_approx))
    den = np.sqrt(
        np.trace(U_target.conj().T @ U_target).real *
        np.trace(U_approx.conj().T @ U_approx).real
    )
    return float(num / den) if den != 0 else 0.0


def frobenius_error(U_target: np.ndarray, U_approx: np.ndarray) -> float:
    """
    Frobenius norm (Euclidean) error: ||U_target - U_approx||_F.
    Lower is better; zero means identical matrices.
    """
    diff = U_target - U_approx
    return float(np.linalg.norm(diff, ord='fro'))


def spectral_norm_error(U_target: np.ndarray, U_approx: np.ndarray) -> float:
    """
    Spectral (operator) norm error: ||U_target - U_approx||_2
    i.e., largest singular value of the difference.
    """
    diff = U_target - U_approx
    svals = np.linalg.svd(diff, compute_uv=False)
    return float(svals[0])  # largest singular value


def trace_distance(U_target: np.ndarray, U_approx: np.ndarray) -> float:
    """
    Trace distance (Schatten-1 norm based):
        D = (1/2) * ||U_target - U_approx||_1
    where ||·||_1 is the sum of singular values (trace norm).
    """
    diff = U_target - U_approx
    svals = np.linalg.svd(diff, compute_uv=False)
    return 0.5 * float(np.sum(svals))


def hilbert_schmidt_inner(U_target: np.ndarray, U_approx: np.ndarray, normalized: bool = True) -> complex:
    """
    Hilbert–Schmidt inner product:
        <U_target, U_approx> = Tr(U_target^\\dagger U_approx)
    If normalized=True, returns Tr(U_target^\\dagger U_approx)/m,
    which for unitaries is (complex) and whose magnitude equals operator fidelity.
    """
    m = U_target.shape[0]
    val = np.trace(U_target.conj().T @ U_approx)
    return val / m if normalized else val


def average_gate_fidelity(U_target: np.ndarray, U_approx: np.ndarray) -> float:
    """
    Average gate fidelity (widely used benchmark metric):
        F_avg = (|Tr(U_target^\\dagger U_approx)|^2 + m) / (m * (m + 1))
    Assumes both are m×m unitaries (or near-unitary).
    """
    m = U_target.shape[0]
    t = np.trace(U_target.conj().T @ U_approx)
    return float((np.abs(t) ** 2 + m) / (m * (m + 1)))


# -----------------------------
# Unitary-specific diagnostics
# -----------------------------

def _circular_min_arc_length(phases: np.ndarray) -> float:
    """
    Helper: Given an array of angles in [-π, π), compute the length of the
    minimal arc on the unit circle that covers all angles. (Result in [0, 2π].)
    """
    # Sort phases and compute circular gaps
    ph = np.sort(phases)
    diffs = np.diff(ph)
    # include wrap-around gap
    last_gap = (ph[0] + 2 * np.pi) - ph[-1]
    gaps = np.concatenate([diffs, [last_gap]])
    # The complement of the largest gap is the minimal covering arc
    max_gap = np.max(gaps)
    return float(2 * np.pi - max_gap)


def unitary_eigenphase_spread(U_target: np.ndarray, U_approx: np.ndarray) -> float:
    """
    Compute the eigenphase 'spread' for W = U_target^\\dagger U_approx.
    Steps:
      1) W = U_target^\\dagger U_approx (unitary if both are unitary)
      2) eigenvalues λ_k of W lie on the unit circle; let θ_k = arg(λ_k) in [-π, π)
      3) return minimal covering arc length Δθ ∈ [0, 2π]
    Smaller Δθ indicates closer unitary action modulo a global phase.
    """
    W = U_target.conj().T @ U_approx
    evals = np.linalg.eigvals(W)
    phases = np.angle(evals)  # in [-π, π)
    return _circular_min_arc_length(phases)


def diamond_norm_unitary_estimate(U_target: np.ndarray, U_approx: np.ndarray) -> float:
    """
    Practical numerical estimate of the diamond-norm distance between the
    unitary channels Ad_U and Ad_V (i.e., ρ -> U ρ U^† vs V ρ V^†).

    For unitaries, a convenient proxy is:
        || Ad_U - Ad_V ||_diamond  ≈  2 * sin(Δθ / 2)
    where Δθ is the minimal covering arc length of eigenphases of W = U_target^\\dagger U_approx
    (after optimal global phase alignment, which this Δθ effectively encodes).

    NOTE: This is an estimate commonly used in practice; for exact values one
    would solve an SDP or use specialized formulas. As Δθ -> 0, the estimate -> 0;
    as Δθ -> π, the estimate -> 2 (maximum channel distinguishability).
    """
    delta_theta = unitary_eigenphase_spread(U_target, U_approx)
    # Clamp to [0, π] for the sinusoidal estimate symmetry
    delta_theta = min(delta_theta, np.pi)
    return float(2.0 * np.sin(delta_theta / 2.0))


