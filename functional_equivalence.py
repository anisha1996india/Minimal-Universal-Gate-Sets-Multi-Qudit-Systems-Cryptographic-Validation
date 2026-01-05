
import os
import pandas as pd
from utils.tools import *


def randomMatrixDecompRecons(n):
    matrix = generate_random_unitary_matrix(n)
    R_matrices, Phi = reckon_decompose_unitary(matrix)
    reconstructed = Phi
    if R_matrices:
        # The R_matrices are stored in the order they are multiplied from left
        # M = R_matrices[0] @ R_matrices[1] @ ... @ R_matrices[-1] @ Phi
        if len(R_matrices)>1: reconstructed = np.linalg.multi_dot(R_matrices) @ Phi
        else: reconstructed = R_matrices @ Phi
    return(matrix, R_matrices, Phi, reconstructed)


def decomposition_accuracy(trials = 10):

    # Storing Accuracy Validation Logs
    validationLogs = []
    validationLogs.append("\n--- Functional Equivalence : Validation of Decomposition Accuracy ---")

    table = pd.DataFrame(
        columns = [
            'Qudit States', 
            'Trials Count', 
            'Operator fidelity', 
            'Trace similarity', 
            'Frobenius error', 
            'Spectral norm error', 
            'Eigenphase minimal arc'
            ]
        )

    arr = generate_sorted_random_numbers()
    validationLogs.append(f"Randomly Generated Choice of Qudit States {arr}.")

    for item in arr:

        validationLogs.append(f"--- No of States in Qudits {item} ---")
        fidelity_score_cumulated = 0
        trace_similarity_score_cumulated = 0
        frobenius_error_cumulated = 0
        spectral_norm_error_cumulated = 0
        eigenphase_spread_cumulated = 0
        
        for trial_run_index in range(trials):
            validationLogs.append(f"Trial Run Index {trial_run_index}")
            matrix, R_matrices, Phi, reconstructed = randomMatrixDecompRecons(item)
            validationLogs.append(f"Randomly Generated Matrix \n{matrix}")
            validationLogs.append(f"Reck's Decomposition method to decomposes the matrix into a product of R_i matrices and a diagonal phase matrix.")
            validationLogs.append(f"R_Matrices \n{R_matrices}")
            validationLogs.append(f"Diagonal Phase Matrix \n{Phi}")

            fidelity_score  = operator_fidelity(matrix, reconstructed)
            trace_similarity_score  = trace_similarity(matrix, reconstructed)
            frobenius_error_value = frobenius_error(matrix, reconstructed)
            spectral_norm_error_value = spectral_norm_error(matrix, reconstructed)
            eigenphase_spread_value = unitary_eigenphase_spread(matrix, reconstructed)   # radians, in [0, 2π]
            gate_fidelity_score = average_gate_fidelity(matrix, reconstructed)
            gate_fidelity_normalized_score = hilbert_schmidt_inner(matrix, reconstructed, normalized=True)
            diamond_norm_estimate = diamond_norm_unitary_estimate(matrix, reconstructed)          # in [0, 2]
            
            validationLogs.append(f"fidelity_score :  {fidelity_score}")
            validationLogs.append(f"trace_similarity_score : {trace_similarity_score}")
            validationLogs.append(f"frobenius_error_value : {frobenius_error_value}")
            validationLogs.append(f"spectral_norm_error_value : {spectral_norm_error_value}")
            validationLogs.append(f"eigenphase_spread_value : {eigenphase_spread_value}")
            validationLogs.append(f"gate_fidelity_score : {gate_fidelity_score}")
            validationLogs.append(f"gate_fidelity_normalized_score : {gate_fidelity_normalized_score}")
            validationLogs.append(f"diamond_norm_estimate : {diamond_norm_estimate}")

            fidelity_score_cumulated += fidelity_score 
            trace_similarity_score_cumulated += trace_similarity_score 
            frobenius_error_cumulated += frobenius_error_value 
            spectral_norm_error_cumulated += spectral_norm_error_value 
            eigenphase_spread_cumulated += eigenphase_spread_value 

        table.loc[len(table)] = [
            item, 
            trials, 
            fidelity_score_cumulated/trials, 
            trace_similarity_score_cumulated/trials, 
            frobenius_error_cumulated/trials, 
            spectral_norm_error_cumulated/trials, 
            eigenphase_spread_cumulated/trials
        ]

    table['Qudit States'] = table['Qudit States'].astype(int)
    table['Trials Count'] = table['Trials Count'].astype(int)
    # pd.set_option('display.float_format', lambda x: f'{x:.16f}')

    validationLogsReport = "\n\n".join([str(item) for item in validationLogs])
    output_folder = "Functional_Equivalence"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    fileLocation = f"{output_folder}/Functional Equivalence Report.txt"
    with open(fileLocation, "w", encoding="utf-8") as file:
        file.write(validationLogsReport)

    return table


if __name__ == "__main__":

    # Validation of Decomposition Accuracy
    print(" ---- ---- Validation of Decomposition Accuracy ---- ---- ")
    response_table = decomposition_accuracy()
    print(response_table)

