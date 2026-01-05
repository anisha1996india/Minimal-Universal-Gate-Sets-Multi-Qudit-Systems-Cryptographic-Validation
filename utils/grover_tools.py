
# Import Libraries
import os
import cirq
import cmath
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from utils.tools import *

# --- Configuration for the Circuit ---
# Define the dimension of the qudit system.
# Use QU_DIMENSION = d for d-dimensional Hilbert space.
QU_DIMENSION = 4

# Define the number of qubits (or qudits in this case).
# Use NUM_QUDITS = n for circuit operating on n qudits.
NUM_QUDITS = 2

# Define the number of repetitions for the simulation.
# A higher number of repetitions leads to a more accurate histogram.
SIMULATION_REPETITIONS = 5000

# Generate the U0 and Uf matrices globally for use in gate definitions.
def matrices():
    """Generates the unitary matrices for the U0 and Uf gates.

    These matrices are designed to mimic the oracle (U0) and diffusion/amplification
    operator (Uf) in Grover's algorithm, operating within a 16-dimensional Hilbert space.

    U0: Flips the phase of the |0> state (all qubits in |0>).
        In this context, it's applied to a 16-dimensional space.
    Uf: Marks a specific state (index 4 in the 16-dim space) by flipping its phase.

    Returns:
        tuple: A tuple containing two numpy arrays: (U0_arr, Uf_arr).
    """
    # The identity matrix for a 16-dimensional space.
    # This space is assumed to be the tensor product of QU_DIMENSION^NUM_QUDITS states.
    U0_arr = np.identity(QU_DIMENSION**NUM_QUDITS, dtype=complex)
    Uf_arr = np.identity(QU_DIMENSION**NUM_QUDITS, dtype=complex)

    # U0: Flip the phase of the |0...0> state (the first basis vector).
    U0_arr[0, 0] = -1

    # Uf: Mark a specific state by flipping its phase.
    # The 'positions' variable determines which state is marked.
    # Currently hardcoded to index 4. This is a simplification for the example.
    positions = [4] # Index of the state to be marked by Uf
    for elem1 in positions:
        Uf_arr[elem1, elem1] = -Uf_arr[elem1, elem1]

    return U0_arr, Uf_arr

# Defining U0 and Uf matrices
U0_arr, Uf_arr = matrices()

# Qudit Hadamard Gate
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

#U0 and Uf gates derived from U0 and Uf matrices defined before
class U0Gate(cirq.Gate):
    """A custom multi-qudit gate
    """
    
#     def __init__(self):
#         super().__init__()  # Correct call to super()
#         self.num_qubits = 4
#         self.name = 'QuartHGate' 

    def _qid_shape_(self):
        # By implementing this method this gate implements the
        # cirq.qid_shape protocol and will return the tuple (3,)
        # when cirq.qid_shape acts on an instance of this class.
        # This indicates that the gate acts on a single qutrit.
        return (QU_DIMENSION,QU_DIMENSION)

    def _unitary_(self):
        # Since the gate acts on three level systems it has a unitary 
        # effect which is a three by three unitary matrix.
        return np.array(U0_arr)

    def _circuit_diagram_info_(self, args):
        return '[+1]'

class UfGate(cirq.Gate):
    """A custom multi-qudit gate
    """
    
#     def __init__(self):
#         super().__init__()  # Correct call to super()
#         self.num_qubits = 4
#         self.name = 'QuartHGate' 

    def _qid_shape_(self):
        # By implementing this method this gate implements the
        # cirq.qid_shape protocol and will return the tuple (3,)
        # when cirq.qid_shape acts on an instance of this class.
        # This indicates that the gate acts on a single qutrit.
        return (QU_DIMENSION,QU_DIMENSION)

    def _unitary_(self):
        # Since the gate acts on three level systems it has a unitary 
        # effect which is a three by three unitary matrix.
        return np.array(Uf_arr)

    def _circuit_diagram_info_(self, args):
        return '[+1]'

# Measuring the state where the Grover's Algorithm has amplified the probability
def marked_state_measurement(circuit, gates = "traditional_gates"):

    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=SIMULATION_REPETITIONS)

    measurement_outcomes = result.measurements['q(0) (d=4),q(1) (d=4)'] # This will be a list of lists, e.g., [[0, 0], [1, 0], ...]
    histogram_data = []
    for outcome_pair in measurement_outcomes:
        # outcome_pair will be something like [0, 1] representing q0's result and q1's result
        # Convert each integer to its string representation and join them.
        histogram_data.append("".join(map(str, outcome_pair[::-1])))
    counts = Counter(histogram_data)

    # --- Histogram Visualization ---

    # Prepare data for plotting
    # Sort by count (descending) to highlight the most frequent outcomes.
    sorted_items_by_count = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    labels_by_count = [item[0] for item in sorted_items_by_count]
    values_by_count = [item[1] for item in sorted_items_by_count]

    # Set a visually pleasing style for the plot.
    plt.style.use('seaborn-v0_8-whitegrid')

    # Create the figure and axes for the plot.
    fig, ax = plt.subplots(figsize=(10, 6)) # Adjust figure size for readability.

    # Plot the bars of the histogram.
    ax.bar(labels_by_count, values_by_count, color='skyblue', width=0.7)

    # Add labels and a title for clarity.
    ax.set_xlabel("Measurement Outcome (q0q1)", fontsize=24)
    ax.set_ylabel("Frequency", fontsize=18)
    if gates == "universal_gates":
        ax.set_title("Measurement Outcome Distribution \n (Grover's Circuit with Universal Gates)", fontsize=18, fontweight='bold')
    else:
        ax.set_title("Measurement Outcome Distribution \n (Grover's Circuit with Traditional Gates)", fontsize=18, fontweight='bold')


    # Rotate x-axis labels to prevent overlap.
    plt.xticks(rotation=90, ha='right', fontsize=18)

    # Add value labels on top of each bar for exact counts.
    for i, v in enumerate(values_by_count):
        ax.text(i, v + 10, str(v), ha='center', va='bottom', fontsize=18)

    # Improve layout to prevent elements from being cut off.
    plt.tight_layout()

    output_folder = "Grover_Measurements"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if gates == "universal_gates":
        imageLocation = f"{output_folder}/Grovers Circuit with Universal Gates.png"
    else:
        imageLocation = f"{output_folder}/Grovers Circuit with Traditional Gates.png"

    # Download the Image.
    plt.savefig(imageLocation)

    # Display the plot.
    # plt.show()

    # Release the Image.
    plt.close()

    response = {
        "measurement_outcomes" : f"Grover's algorithm has amplified the probability of the state {sorted_items_by_count[0][0]}",
        "states_with_most_probability" : ". ".join(str(item[0])+" - "+str(item[1]) for item in sorted_items_by_count[:5]),
        "histogram_location" : imageLocation
    }
    return response

