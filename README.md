# dqsweep

![Pytest and Flake8 validation](https://github.com/Quang00/DQC/actions/workflows/python-app.yml/badge.svg)

This repository is designed to analyze the performance (fidelity and latency) of multiple distributed quantum experiments over a configurable quantum network. It uses quantum network simulators such as [NetSquid](https://netsquid.org/) and [SquidASM](https://github.com/QuTech-Delft/squidasm).

## Overview

The project explores the performance of several quantum distributed experiments by sweeping parameters (such as depolarization probabilities, gate times, and qubit coherence times) of a given quantum network and then assess two metrics:

- **Average Fidelity (%):** Output density matrix compared to the expected density matrix.
- **Average Simulation Time (ms):** Simulation time to execute the experiment.

The experiments implemented in this repository include:

- **Nonlocal CNOT Gate:** Implementation of a distributed CNOT gate between Alice and Bob presented in the paper [[1]](#1).
- **Nonlocal CNOT Gate with Two Teleportations:** Another approach that performs the distributed CNOT gate using two quantum teleportations.
- **Ping-Pong Teleportation:** A bidirectional teleportation protocol where a qubit is sent back and forth between Alice and Bob.
- **Distributed Quantum Fourier Transform (DQFT) on Two Qubits:** Implementation of a distributed QFT on two qubits.
- **Distributed Grover on Two Qubits:** Implementation of a distributed Grover on two qubits.

## Installation

1. **Create a Python Virtual Environment:**
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
```
2. **Install SquidASM**
```bash
export NETSQUIDPYPI_USER=user1234
export NETSQUIDPYPI_PWD=password1234
git clone git@github.com:QuTech-Delft/squidasm.git
cd squidasm
make install
```

## Usage

The experiments are executed through the `run_simulation.py` script, which performs parameter sweeps, runs the specified distributed quantum experiment, and aggregates the results. The script accomplishes the following:

- Reads a network configuration from a YAML file.
- Sweeps one or more parameters across defined ranges.
- Executes the chosen experiment multiple times for each parameter combination.
- Saves raw results in CSV format, computes correlations between parameters and performance metrics, and generates heat map visualizations.

### Running an Experiment

Choose an experiment from the following options:

- **`cnot`**: Nonlocal CNOT gate using one ebit and one bit in each direction.
- **`2_teleportations`**: Nonlocal CNOT gate using two teleportations.
- **`pingpong`**: Ping-pong teleportation.
- **`dqft2`**: Distributed Quantum Fourier Transform on 2 qubits.
- **`dgrover2`**: Distributed Grover search on 2 qubits.

#### Example Command 1: Nonlocal CNOT (Depolarization Sweep)
```bash
python -m experiments.run_simulation \
  --config configurations/perfect.yaml \
  --experiment cnot \
  --epr_rounds 10 \
  --num_experiments 100 \
  --sweep_params single_qubit_gate_depolar_prob,two_qubit_gate_depolar_prob \
  --ranges "0.0,0.8,10" "0.0,0.8,10"
```
#### Example Command 2: Distributed Grover on 2 qubits (Depolarization Sweep)
```bash
python -m experiments.run_simulation \
  --config configurations/perfect.yaml \
  --experiment dgrover2 \
  --epr_rounds 10 \
  --num_experiments 100 \
  --sweep_params single_qubit_gate_depolar_prob,two_qubit_gate_depolar_prob \
  --ranges "0.0,0.8,10" "0.0,0.8,10"
```

## Running the Simulations

1. **Prepare Configuration Files:**

   - Provide a valid quantum network configuration or use the ones provided (e.g., `perfect.yaml`, `depolarise_link.yaml`) in the `configurations/` folder.

2. **Select the Experiment and Parameters:**

   - Choose the experiment to run (options include `cnot`, `pingpong`, `dqft2`, `dgrover2`, or `2_teleportations`).
   - Define the parameters to sweep (e.g., single_qubit_gate_depolar_prob, two_qubit_gate_depolar_prob, T1, T2, etc.) along with their corresponding appropriate ranges.

3. **Execute the Simulation Script:**

   - Run the `run_simulation.py` script with appropriate command-line arguments. For example:
     ```bash
     python -m experiments.run_simulation
     ```
   - The script creates an output subdirectory (named based on the experiment and swept parameters) that contains:
     - A CSV file with raw simulation results.
     - A TXT file with parameter-to-metric correlation values.
     - Heat map plots with the performance results.

4. **Analyze the Results:**
   - Open the CSV file to review the detailed simulation data.
   - Use the heatmap images to explore how variations in the parameters affect fidelity and simulation times.

## Example Result (Heat map) from command 1: Nonlocal CNOT (Depolarization Sweep)

<img src=docs/cnot_heat_fidelity.png width="60%" height="60%">

## References
<a id="1">[1]</a> 
Eisert, Jens & Jacobs, Karel & Papadopoulos, Periklis & Plenio, M.. (2000). Optimal local implementation of nonlocal quantum gates. Phys. Rev. A. 62. 10.1103/PhysRevA.62.052317. 
