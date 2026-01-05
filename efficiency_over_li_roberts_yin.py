
import os
import pandas as pd
from utils.tools import *

# def count_reck(matrix):
#     R_matrices, Phi = reckon_decompose_unitary(matrix)
#     return(len(R_matrices)+1)

# def count_cnott(matrix, N):
#     Q, _ = np.linalg.qr(matrix)
#     gates = li_roberts_yin_decompose(Q)
#     return(len(gates))

def gate_comparison():

    gateComparisonLogs = []
    comparison_table = pd.DataFrame(columns=[
        'Qudit States', 'Trials Count', 'Li-Robert-Yin Method - Avg Gate Count', 'Recks Method - Avg Gate Count'
        ]
    )
    gateComparisonLogs.append("\n--- Efficiency of proposed Model over Li-Robets-Yin Decomposition ---")
    
    trial = 10
    arr = generate_sorted_random_numbers(count=10, start=2, end=20)
    gateComparisonLogs.append(f"Randomly Generated Choice of Qudit States {arr}.")

    for N in arr:
        gateComparisonLogs.append(f"--- No of States in Qudits {N} ---")
        reck_decomposition_count = 0
        li_robert_yin_decomposition_count = 0

        for trial_run_index in range(trial):

            gateComparisonLogs.append(f"Trial Run Index {trial_run_index}")
            matrix = generate_random_unitary_matrix(N)
            gateComparisonLogs.append(f"Randomly Generated Matrix \n{matrix}")

            R_matrices, Phi = reckon_decompose_unitary(matrix)
            reck_decomposition_count += len(R_matrices)+1
            gateComparisonLogs.append(f"Reck's Decomposition method to decomposes the matrix into a product of R_i matrices and a diagonal phase matrix.")
            gateComparisonLogs.append(f"R_Matrices \n{R_matrices}")
            gateComparisonLogs.append(f"Diagonal Phase Matrix \n{Phi}")

            Q, _ = np.linalg.qr(matrix)
            gates = li_roberts_yin_decompose(Q)
            li_robert_yin_decomposition_count += len(gates)
            gateComparisonLogs.append(f"Li Roberts Yin decomposition breaks unitary matrix into fully-controlled single-qubit gates.")
            gateComparisonLogs.append(f"Decomposed Matrices \n{gates}")

        gateComparisonLogs.append(f"Execution result after {trial} trials")
        gateComparisonLogs.append(f"Gate count for Reck's decomposition \n{reck_decomposition_count}")
        gateComparisonLogs.append(f"Gate count for Reck's ecomposition  \n{li_robert_yin_decomposition_count}")

        comparison_table.loc[len(comparison_table)] = [N, trial, li_robert_yin_decomposition_count/trial, reck_decomposition_count/trial]

    comparison_table['Qudit States'] = comparison_table['Qudit States'].astype(int)
    comparison_table['Trials Count'] = comparison_table['Trials Count'].astype(int)
    # pd.set_option('display.float_format', lambda x: f'{x:.2f}')

    gateComparisonLogs.append(f"Final Comparison Table  \n{comparison_table}")

    efficiencyLogsReport = "\n\n".join([str(item) for item in gateComparisonLogs])
    output_folder = "Efficiency_Report"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    fileLocation = f"{output_folder}/Efficiency Over Li Roberts Yin Decomposition.txt"
    with open(fileLocation, "w", encoding="utf-8") as file:
        file.write(efficiencyLogsReport)

    return comparison_table


if __name__ == "__main__":

    # Average Gate Comparison between Reck's Decomposition and Li-Robert-Yin's Decomposition
    print(" ---- ---- Average Gate Comparison between Reck's Decomposition and Li-Robert-Yin's Decomposition ---- ---- ")
    response_table = gate_comparison()
    print(response_table)

