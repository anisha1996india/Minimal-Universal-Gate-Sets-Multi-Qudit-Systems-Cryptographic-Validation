# Minimal-Universal-Gate-Sets-Multi-Qudit-Systems-Cryptographic-Validation

The code repository alligns with the research paper titled [**Practically Implementable Minimal Universal Gate Sets for Multi-Qudit Systems with Cryptographic Validation**](https://doi.org/10.1007/978-3-032-13301-4_17), authored by [Anisha Dutta](https://github.com/anisha1996india), [Sayantan Chakraborty](https://github.com/ChakSayantan), Chandan Goswami, and Prof. Avishek Adhikari. The research paper was presented and published at Indocrypt 2025 conference, which was held at IIIT Bhubaneswar on December, 2025. The work provides an algorithms for decomposing arbitrary multi-qudit operations using a rigorously proven minimal universal gate set. The implementation validates the proposed framework through functional equivalence tests and practical demonstrations, including Grover’s algorithm and QKD circuit simulations, confirming its cryptographic applicability, also establishing the efficiency across another widely acceptable higher-dimensional quantum systems. 

## 1. Introduction

This repository provides a comprehensive implementation and validation framework for the minimal universal gate set proposed for multi-qudit systems, as described in the research paper “Practically Implementable Minimal Universal Gate Sets for Multi-Qudit Systems with Cryptographic Validation.”

Unlike conventional qubit-based circuits, qudit circuits operate on higher-dimensional Hilbert spaces, enabling more compact, expressive, and scalable quantum operations. The repository implements a Python-based algorithmic decomposition of arbitrary multi-qudit unitary matrices into the rigorously proven minimal universal gate set `S = PHASE1 ∪ T_elements`. The decomposition is based on Reck’s approach and includes detailed time and space complexity analyses in the correcponding research paper.

The implementation validates the functional equivalence of the decomposed circuits against their original unitary forms using fidelity, trace similarity, and matrix norm metrics—all confirming near-perfect agreement within machine precision. Further, an efficiency comparison with the widely accepted Li–Roberts–Yin (LRY) decomposition demonstrates that the proposed method achieves significantly reduced gate counts and better scalability for higher-dimensional systems.

Two cryptographically significant algorithms — **Grover’s search** and **Quantum Key Distribution (QKD)** — are implemented using both traditional and decomposed gate models. The simulations confirm that the decomposed circuits reproduce identical logical and cryptographic outcomes, thereby validating the proposed gate set’s practical implementability, hardware-agnostic portability, and efficiency for realistic quantum cryptographic workflows.

## 2. Repository Structure

```
.
├── cryptographic_simulation.py
├── functional_equivalence.py
├── efficiency_over_li_roberts_yin.py
├── utils/tools.py
├── utils/circuits_grover.py
├── utils/grover_tools.py
├── utils/circuits_qkd.py
├── utils/qkd_tools.py
├── requirements.txt
├── Grover_Measurements/Grovers Circuit with Traditional Gates.png (Histogram generated during execution)
├── Grover_Measurements/Grovers Circuit with Universal Gates.png (Histogram generated during execution)
├── QKD_Measurements/QKD Basis Choice Distribution.png (Distribution plot generated during execution)
├── QKD_Measurements/QKD Simulation Outcomes.txt (Simulation log generated during execution)
├── Functional_Equivalence/Functional Equivalence Report.txt (Validation log generated during execution)
├── Efficiency_Report/Efficiency Over Li Roberts Yin Decomposition.txt (Gate Comparison logs generated during execution)
```

- **`cryptographic_simulation.py`** → Entry point to run Grover’s Algorithm and QKD simulations.  
- **`functional_equivalence.py`** → Implementation to check functional equivalence of proposed decomposition.  
- **`efficiency_over_li_roberts_yin.py`** → gate count comparison between proposed decomposition and Li-Roberts-Yin decomposition.
- **`tools.py`** → General utility tools used for the above mentioned workflows.  
- **`grover_tools.py`** → Core utilities for Grover’s algorithm (Hadamard, oracle, diffusion, decomposition).  
- **`circuits_grover.py`** → Builders for Grover circuits (traditional and universal).  
- **`qkd_tools.py`** → Qutrit swap gates, Hadamard generalizations, decomposition utilities.  
- **`circuits_qkd.py`** → QKD simulation circuits for Alice and Bob.  
- **`requirements.txt`** → Packages / Libraries required to create enviroment to execute above files.  
- **Output folders** store logs, histograms, and distribution plots.  

## 3. Technical Flow : Cryptographic Algorithms Simulations

- Execution begins with **`cryptographic_simulation.py`**.  
- For **Grover’s algorithm**: two 4-dimensional qudits are initialized.  
  - Traditional circuit → Generalized Hadamard, Uf, U0, measurement.  
  - Decomposed circuit → Reck’s decomposition of Hadamard-like gates into 2×2 rotations + PHASE1 gates.  
- For **QKD**:  
  - Alice encodes states in rectilinear/diagonal basis.  
  - Bob randomly selects measurement basis.  
  - Both traditional and universal gates are used.  
  - Keys are extracted where bases match.  
- Outputs include:  
  - Histograms of Grover amplified states.  
  - Basis choice distribution plots.  
  - Logs with per-round QKD results.  
- Grover’s algorithm validates **attack feasibility** in qudit cryptanalysis.  
- QKD demonstrates **defensive protocol correctness** under decomposition.  
- Equivalence testing confirms minimal gate sets can support cryptography securely.  

### 3A. Grover’s Algorithm Validation

#### Traditional Circuit

Constructed with generalized qudit gates:  

```python
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
```

#### Universal Circuit

Constructed using Reck’s decomposition:  

```python
H_unitary = QuditHGate(d=4)._unitary_()
R_matrices, Phi = reckon_decompose_unitary(H_unitary)
ops = [ArbitraryGate(d=4, matrix=Phi)(q0), ArbitraryGate(d=4, matrix=Phi)(q1)]
for M in R_matrices[::-1]:
    ops += [ArbitraryGate(d=4, matrix=M)(q0), ArbitraryGate(d=4, matrix=M)(q1)]
```

#### Results

- **Traditional Histogram**  
  ![Grover Traditional](./Grover_Measurements/Grovers%20Circuit%20with%20Traditional%20Gates.png)  

- **Universal Histogram**  
  ![Grover Universal](./Grover_Measurements/Grovers%20Circuit%20with%20Universal%20Gates.png)  

**Interpretation:** Both amplify the same marked state, confirming equivalence.  
The universal version has greater depth but functional correctness is preserved.  

### 3B. Quantum Key Distribution (QKD) Validation

- Alice chooses random trit (0/1/2) and basis (rectilinear/diagonal).  
- Bob chooses random basis (rectilinear/diagonal).  
- Keys established when bases align.  

#### Basis Choice Distribution  

![QKD Basis](./QKD_Measurements/QKD%20Basis%20Choice%20Distribution.png)  

#### Example Logs  

```
--- QKD Round 29/100 ---
Alice: Preparing bit 2 in basis 'diagonal'.
Bob: Choosing basis 'diagonal'.
0(d=3): --- Qutrit_0swap2_Gate --- Qu3H --- Qu3H --- M('diagonal_measurement')
0(d=3): --- Qu3M x3 --- Qu3M x5 --- Qu3M x5 --- M('diagonal_measurement')
Bob: Measured bit 2 in basis 'diagonal'.
Bob: Measured bit 2 in basis 'diagonal'.
Alice and Bob used the same basis.
Key established for this round: 2
```

In Round 29 of the QKD simulation, within the traditional setup, Alice randomly selected the trit “2” and the diagonal basis, while Bob independently chose the same diagonal basis. On Alice’s side, this resulted in a circuit beginning with the 0swap2 gate to encode the symbol, followed by a Hadamard gate to realize the diagonal basis. Bob, matching the basis choice, appended a Hadamard gate before measurement. 
In case of universal decomposed gates, all the gates are decomposed into unitary matrices of the proposed forms and then appended subsequently replacing the traditional gates. The 0swap2 gate has been replaced with one R_ij, one Phi balance and one Phi matrices, while the Hadamard gate has been replaced with three R_ij, one Phi balance and one Phi matrices.
Both measurements yielded the value “2”, and since the bases aligned, a key bit was successfully established, exactly as predicted by the principles of QKD.


```
--- QKD Round 67/100 ---
Alice: Preparing bit 0 in basis 'rectilinear'.
Bob: Choosing basis 'diagonal'.
20 Anonymous Submission
0(d=3): --- Qu3H --- M('rectilinear_measurement')
0(d=3): --- Qu3M --- Qu3M --- Qu3M --- Qu3M --- Qu3M --- M('diagonal_measurement')
Bob: Measured bit 1 in basis 'diagonal'.
Bob: Measured bit 2 in basis 'diagonal'.
Alice and Bob used different bases or measurement result invalid.
No key bit established for this round.
```

In Round 67 of the QKD simulation, within the traditional setup, Alice randomly selected the trit “0” and the rectilinear basis, while Bob independently chose the diagonal basis. Because of Alice’s choices, this resulted in a circuit that keeps the state unaltered. Bob, matching the basis choice, appended a Hadamard gate before measurement. 
In case of universal decomposed gates, the only Hadamard gate in the circuit has been replaced with three R_ij, one Phi balance and one Phi matrices. We see both measurements yielded different values of measured bit, although the choice of bases being different, this round will anyway not yield any shared key bit.

#### Summary  

- 100 rounds simulated.  
- Around 50 shared key bits established.  
- Both traditional and universal gates produced identical outcomes whenever choices of bases matches.  

### 3C. Implementation Details

- **Reck’s decomposition** expresses arbitrary unitaries as products of 2×2 rotations and a diagonal phase.  
- Custom Cirq gates (`QuditHGate`, `Qutrit_0swap1`, `Qutrit_0swap2`) are decomposed into `ArbitraryGate` instances.  
- Circuits use `cirq.LineQid` with dimension=3 (QKD) or 4 (Grover).  
- Outputs:  
  - Histograms (`Matplotlib`)  
  - Logs (plain text)  

### 3D. Results & Outputs

#### Sample Grover JSON Output  

```json
{
  "status": 200,
  "Grovers Circuit with Traditional Gates": {
    "measurement_outcomes": "Grover's algorithm has amplified the probability of the state 03",
    "states_with_most_probability": "03 - 2404. 10 - 204. 12 - 188. 21 - 185. 33 - 184",
    "histogram_location": "Grover_Measurements/Grovers Circuit with Traditional Gates.png"
  },
  "Grovers Circuit with Universal Gates": {
    "measurement_outcomes": "Grover's algorithm has amplified the probability of the state 03",
    "states_with_most_probability": "03 - 2400. 30 - 200. 00 - 200. 23 - 190. 20 - 187",
    "histogram_location": "Grover_Measurements/Grovers Circuit with Universal Gates.png"
  }
}
```

#### Sample QKD JSON Output  

```json
{
  "status": 200,
  "QKD Simulation Response": {
    "Total Rounds Simulated": 100,
    "Number of Key Bits Established": 56,
    "Shared Secret Key": "12201220120122220110210202122200201211121012212020010001",
    "Impression": "The choosen bases of Alice and Bob has matched 56 times and 56 many times the circuits with traditional and universal gates has yielded same measurement.",
    "QKD Simulation Logs": "QKD_Measurements/QKD Simulation Outcomes.txt",
    "QKD Basis Choice Plot": "QKD_Measurements/QKD Basis Choice Distribution.png"
  }
}
```

## 4. Numerical Validation of Qudit Gate Decomposition Accuracy

This section validates that arbitrary multi-qudit unitary operators, when decomposed using the proposed minimal universal gate set $S = \text{PHASE1} \cup T_{\text{elements}}$ remain **functionally equivalent** to their original (undecomposed) representations. The goal is to empirically demonstrate that the our proposed decomposition constructed solely from PHASE1 gates and embedded 2-dimensional subspace rotations:

- Preserves the action of the original unitary operator,
- Introduces only negligible numerical error due to floating-point precision,
- Scales correctly across increasing qudit dimensions.

#### Numerical Measures Validated

- **Operator Fidelity** – Measures overlap between \(U\) and \(\tilde{U}\); ideal value is 1.
- **Trace Similarity** – Phase-invariant similarity measure.
- **Frobenius Norm Error** – \(\|U - \tilde{U}\|_F\), total reconstruction error.
- **Spectral Norm Error** – Worst-case singular value deviation.
- **Eigenphase Minimal Arc Distance** – Confirms phase consistency modulo \(2\pi\).

Together, these metrics verify equivalence up to machine precision. Across all tested dimensions and trials:

- Operator fidelity and trace similarity consistently evaluate to **1.0**.
- Reconstruction errors remain on the order of \(10^{-15}\), consistent with floating-point limits.
- No observable degradation is seen as qudit dimension increases.

#### Sample Output  

```
     Qudit States  Trials Count  Operator fidelity  Trace similarity  Frobenius error  Spectral norm error  Eigenphase minimal arc
0             2            10                1.0               1.0     5.390850e-16         5.132342e-16            0.000000e+00
1             5            10                1.0               1.0     1.724244e-15         1.451355e-15            0.000000e+00
2             8            10                1.0               1.0     2.322421e-15         1.865019e-15            0.000000e+00
3            19            10                1.0               1.0     5.474471e-15         4.195457e-15            2.664535e-16
4            20            10                1.0               1.0     6.196331e-15         4.821528e-15            1.776357e-16
```

## 5. Gate Count Advantage over Existing Qudit Decomposition Method

While universality guarantees correctness, **practical quantum computation demands efficiency**. Gate count directly impacts:

- Circuit depth,
- Noise accumulation,
- Execution time,
- Fault-tolerance overhead.

Thus, comparing decomposition cost is essential for real-world applicability. For increasing qudit dimensions:

1. Random unitary matrices are generated.
2. Each unitary is decomposed using:
   - **Li–Roberts–Yin (LRY) Method**, and
   - **Proposed Decomposition Method**.
3. The total number of elementary gates required by each method is recorded.
4. Results are averaged over multiple trials per dimension.

Both methods produce correct decompositions; the comparison focuses purely on **resource efficiency**. The proposed method consistently outperforms the LRY approach:

- Requires **significantly fewer gates**,
- Exhibits better scaling with dimension,
- Avoids high-arity controlled operations.

#### Sample Output  

```
   Qudit States  Trials Count  Li-Robert-Yin Method - Avg Gate Count  Recks Method - Avg Gate Count
0             6            10                                   24.0                           17.0
1             7            10                                   28.0                           23.0
2             8            10                                   88.0                           30.0
3             9            10                                   99.0                           38.0
4            12            10                                  132.0                           68.0
5            13            10                                  143.0                           80.0
6            14            10                                  154.0                           93.0
7            16            10                                  416.0                          122.0
8            18            10                                  468.0                          155.0
9            19            10                                  494.0                          173.0
```

## 6. How to Run

#### Requirements  

All necessary python libraries with matching version are provided in requirements file. 
Majorly required python libraries are following. 

- Python 3.9+  
- Cirq  
- NumPy  
- Matplotlib  

We recommend the user to create a virtual python environment and then install all the required libraries / packages inside the virtual environment only. One can refer to the following chunk of codes for so.

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

#### Run Simulations of Cryptographic Algorithms with Tradition Gates and Decomposed Gates  

```bash
python cryptographic_simulation.py
```

Outputs will be displayed in terminal itself.
Histograms, distribution plots and simulation logs will be deposited in `Grover_Measurements/` and `QKD_Measurements/` folders. 

#### Validate Decomposition Accuracy  

```bash
python functional_equivalence.py
```

For random choices of Qudit states, a detailed functional equivalence table will be displayed in terminal. For 10 trails of each cases, Operator Fidelity, Trace Similarity and Reconstruction Errors like Frobenius Error, Spectral Norm Error, Eigenphase Minimal Arc will be calculated. 

#### Compare Gate Count between LRY Method and Proposed Decomposition Method  

```bash
python efficiency_over_li_roberts_yin.py
```

Gate Comparison table will be displayed in terminal. Alongside, logs will be deposited in `Efficiency_Report/` folder.

---

## ✅ Conclusion
This repository validates the practicality of the minimal universal gate set (`PHASE1 ∪ T_elements`) by validating the decomposition accuracy through functional accuracy parameters. It also demonstrates end-to-end cryptographic protocols in a reproducible Python framework. Both **Grover's Algorithm** and **QKD Simulation** confirm match between traditional and decomposed implementations, ensuring security and scalability in cryptographic contexts. Also, the gate count comparison with a popular decomposition method, namely Li-Roberts-Yin (2012) shows the edge our proposed decomposition method can offer over existing solutions.
