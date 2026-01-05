
'''
The python script aims to simulate a quantum circuit that mimics the aspects of Grover's algorithm, 
specifically involving multi-level quantum systems (quards) and custom universal gates replacing traditional gates. 
Furthermore, it provides visual outcomes of the measurement outputs through a histogram to explain the output.
'''

import json
import random
from utils.qkd_tools import *
from utils.circuits_qkd import *
from utils.grover_tools import *
from utils.circuits_grover import *


# --- Grover's Algorithm Execution with Traditional gates and Decomposed gates ---

def grover_universal_decomposition():

    """
    Grover's Algorithm Execution with Traditional gates and Decomposed gates using the paper's universal gate set S = PHASE_1 U T_elements.
    """

    # Define the qubits. We are using cirq.LineQid to represent multi-level systems.
    # q0 and q1 are 4-dimensional qudits.
    q0 = cirq.LineQid(0, dimension=QU_DIMENSION)
    q1 = cirq.LineQid(1, dimension=QU_DIMENSION)

    try :
        grover_circuit_actual_matrices = grover_circuit_with_traditional_gates(q0, q1)
        grover_response = marked_state_measurement(
            grover_circuit_actual_matrices
        )

        grover_circuit_decomposed_matrices = grover_circuit_with_universal_gates(q0, q1)
        grover_universal_gates_response = marked_state_measurement(
            grover_circuit_decomposed_matrices, 
            gates = "universal_gates"
        )

        return json.dumps(
            {
                "status" : 200,
                "Grovers Circuit with Traditional Gates" : grover_response,
                "Grovers Circuit with Universal Gates" : grover_universal_gates_response
            },
            indent = 2)

    except Exception as e:
        return json.dumps(
            {
                "status" : 500,
                "response" : f'''
                Error occured while measuring marked state using Grover circuit with Traditional Gates : {e} 
                '''
            },
            indent = 2)


# --- Simulation of Quantum Key Distribution ---

def qkd_universal_decomposition(num_rounds: int):
    
    """
    Simulates QKD (BB84) using operations constructible from the paper's universal gate set S = PHASE_1 U T_elements.
    """

    # Storing Simulation Logs
    simulationLogs = []

    simulationLogs.append("--- Simulating QKD Protocol (BB84 with Universal Gates) ---")

    alice_basis_choices = []
    alice_sent_bits = []
    bob_basis_choices = []
    bob_measured_bits_traditional = []
    bob_measured_bits_universal = []
    shared_key = []

    try:
        # Initialize Cirq simulator
        simulator = cirq.Simulator()

        # Count the shared key measurement matching where the bases matches betweeb Bob and Alice
        bob_measurement_match_count = 0

        for round_num in range(num_rounds):

            simulationLogs.append(f"\n--- QKD Round {round_num + 1}/{num_rounds} ---")

            # 1. Alice prepares a bit and basis using operations from S
            alice_bit = random.choice([0, 1, 2])
            alice_basis = random.choice(['rectilinear', 'diagonal'])
            simulationLogs.append(f"Alice: Preparing bit {alice_bit} in basis '{alice_basis}'.")

            # Corrected call to use the function that returns the basis
            alice_circuit_traditional_gates, alice_circuit_universal_gates, alice_qubit, chosen_alice_basis, encoded_alice_bit = alice_prepares_bb84_state(alice_bit, alice_basis)
            
            alice_basis_choices.append(chosen_alice_basis)
            alice_sent_bits.append(encoded_alice_bit)

            # 2. Bob chooses a basis to measure using operations from S
            bob_basis = random.choice(['rectilinear', 'diagonal'])
            simulationLogs.append(f"Bob: Choosing basis '{bob_basis}'.")

            bob_circuit_traditional_gates, bob_circuit_universal_basis, bob_qubit, chosen_bob_basis = bob_measures_state(bob_basis, alice_qubit)
            bob_basis_choices.append(chosen_bob_basis)

            # 3. Combine circuits for simulation
            entanglement_traditional_basis = alice_circuit_traditional_gates + bob_circuit_traditional_gates
            entanglement_universal_basis = alice_circuit_universal_gates + bob_circuit_universal_basis
            simulationLogs.append(str(entanglement_traditional_basis))
            simulationLogs.append(str(entanglement_universal_basis))

            # 4. Simulate the circuit
            result_traditional_basis = simulator.run(entanglement_traditional_basis, repetitions=1)
            result_universal_basis = simulator.run(entanglement_universal_basis, repetitions=1)

            # 5. Extract Bob's measurement outcome
            bob_measurement_traditional_circuit = None
            bob_measurement_universal_circuit = None
            measurement_key = f'{chosen_bob_basis}_measurement' # Use Bob's chosen basis for key
            bob_measurement_raw_result_traditional_circuit = result_traditional_basis.measurements.get(measurement_key)
            bob_measurement_raw_result_universal_circuit = result_universal_basis.measurements.get(measurement_key)
            
            bob_measurement_traditional_circuit = bob_measurement_raw_result_traditional_circuit[0][0]
            bob_measured_bits_traditional.append(bob_measurement_traditional_circuit)
            simulationLogs.append(f"Bob: Measured bit {bob_measurement_traditional_circuit} in basis '{chosen_bob_basis}'.")
            bob_measurement_universal_circuit = bob_measurement_raw_result_universal_circuit[0][0]
            bob_measured_bits_universal.append(bob_measurement_universal_circuit)
            simulationLogs.append(f"Bob: Measured bit {bob_measurement_universal_circuit} in basis '{chosen_bob_basis}'.")
            
            # 6. Alice and Bob compare bases and establish key bits
            shared_key_bit = None
            if chosen_alice_basis == chosen_bob_basis:
                simulationLogs.append("Alice and Bob used the same basis.")
                shared_key_bit = encoded_alice_bit
                shared_key.append(shared_key_bit)
                simulationLogs.append(f"  Key established for this round: {shared_key_bit}")

                if bob_measurement_traditional_circuit == bob_measurement_universal_circuit:
                    bob_measurement_match_count += 1

            else:
                pass
                simulationLogs.append("Alice and Bob used different bases or measurement result invalid. No key bit established for this round.")

        simulationLogs.append("\n--- QKD Simulation Complete ---")
        simulationLogs.append(f"Total rounds simulated: {num_rounds}")
        simulationLogs.append(f"Number of key bits established: {len(shared_key)}")
        simulationLogs.append(f"Shared Secret Key: {''.join(map(str, shared_key))}")
        simulationLogs.append(f"Bob's key with actual circuit: {bob_measured_bits_traditional}, and with universal gates' circuit: {bob_measured_bits_universal}")

        # --- Visualization of Basis Choices ---
        alice_basis_counts = Counter(alice_basis_choices)
        bob_basis_counts = Counter(bob_basis_choices)

        simulationLogs.append("\n--- Basis Statistics ---")
        simulationLogs.append(f"Alice's basis choices: {dict(alice_basis_counts)}")
        simulationLogs.append(f"Bob's basis choices: {dict(bob_basis_counts)}")

        # Plotting basis choices
        plt.figure(figsize=(8, 5))
        plt.bar(['Alice (Rect)', 'Alice (Diag)', 'Bob (Rect)', 'Bob (Diag)'],
                [alice_basis_counts.get('rectilinear', 0),
                alice_basis_counts.get('diagonal', 0),
                bob_basis_counts.get('rectilinear', 0),
                bob_basis_counts.get('diagonal', 0)],
                color=['skyblue', 'lightblue', 'lightgreen', 'lightcyan'])
        plt.ylabel("Number of Times Chosen")
        plt.title("Basis Choice Distribution")

        # Display the Image. 
        # plt.show()

        output_folder = "QKD_Measurements"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        imageLocation = f"{output_folder}/QKD Basis Choice Distribution.png"

        # Download the Image.
        plt.savefig(imageLocation)
        simulationLogs.append(f"Histogram successfully stored to '{imageLocation}'.")

        simulationLogsReport = "\n".join([str(item) for item in simulationLogs])
        fileLocation = f"{output_folder}/QKD Simulation Outcomes.txt"
        with open(fileLocation, "w", encoding="utf-8") as file:
            file.write(simulationLogsReport)

        simulationLogs.append(f"Text successfully written to '{fileLocation}'.")

        response = {
            "Total Rounds Simulated": num_rounds,
            "Number of Key Bits Established": len(shared_key),
            "Shared Secret Key": ''.join(map(str, shared_key)),
            "Impression" : f'''The choosen bases of Alice and Bob has matched {len(shared_key)} times and {bob_measurement_match_count} many times the circuits with traditional and universal gates has yielded same measurement.''',
            "QKD Simulation Logs" : fileLocation,
            "QKD Basis Choice Plot" : imageLocation
        }

        return json.dumps(
            {
                "status" : 200,
                "QKD Simulation Response" : response
            },
            indent = 2
        )
    
    except Exception as e:
        return json.dumps(
            {
                "status" : 500,
                "response" : f'''
                Error occured while simulating Quantum key distribution : {e} 
                '''
            },
            indent = 2)


if __name__ == "__main__":

    # Grover's Algorithm with Universal Decomposition
    print(" ---- ---- Grover's Algorithm with Universal Decomposition ---- ---- ")
    grover_response = grover_universal_decomposition()
    print(grover_response)

    # QKD's Algorithm with Universal Decomposition
    print(" ---- ---- Quantum Key Distribution with Universal Decomposition ---- ---- ")
    qkd_response = qkd_universal_decomposition(num_rounds = 100)
    print(qkd_response)

