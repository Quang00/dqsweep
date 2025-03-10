# dqsweep

![Pytest and Flake8 validation](https://github.com/Quang00/DQC/actions/workflows/python-app.yml/badge.svg) [![Coverage Status](https://coveralls.io/repos/github/Quang00/dqsweep/badge.svg?branch=main)](https://coveralls.io/github/Quang00/dqsweep?branch=main)

This repository is designed to analyze the performance (fidelity and latency) of multiple distributed quantum experiments over a configurable quantum network. It uses quantum network simulators such as [NetSquid](https://netsquid.org/) and [SquidASM](https://github.com/QuTech-Delft/squidasm).

## Overview

The project explores the performance of several quantum distributed experiments by sweeping given parameters (such as depolarization probabilities, gate times, and qubit coherence times) from a quantum network and then assess two metrics:

- **Average Fidelity (%):** Density matrix output compared to the density matrix expected.
- **Average Simulation Time (ms):** Simulation time to execute the experiment.

The experiments implemented in this repository include:

- **Nonlocal CNOT Gate (`nonlocal_cnot.py`):** Implementation of a distributed CNOT gate between Alice and Bob presented in the paper [[1]](#1).
- **Nonlocal CNOT Gate with Two Teleportations (`nonlocal_cnot_2_teleportations.py`):** Another implementation of the distributed CNOT gate using two quantum teleportations.
- **Ping-Pong Teleportation (`pingpong.py`):** A bidirectional quantum teleportation where a qubit is sent back and forth between Alice and Bob.
- **Distributed Quantum Fourier Transform (DQFT) on Two Qubits (`dqft_2.py`):** Implementation of a distributed QFT on two qubits.
- **Distributed Grover on Two Qubits (`dgrover_2.py`):** Implementation of a distributed Grover on two qubits.

## Installation

1. **Create a Python Virtual Environment:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
```

2. **Install SquidASM** (The credentials from Netsquid are required)

```bash
export NETSQUIDPYPI_USER=user1234
export NETSQUIDPYPI_PWD=password1234
git clone git@github.com:QuTech-Delft/squidasm.git
make install -C squidasm
```

3. **Verify the installation**

```bash
pytest
```

## Usage

The experiments are executed through the `run_simulation.py` script, which does the parameter sweeps, runs the given distributed quantum experiment, and produces the results in a folder containing different files. To run the simulation:

1. **Configure Quantum Network:**

   - (**`--config`**): Provide a valid quantum network configuration or use the ones already provided (e.g., `perfect.yaml`, `depolarise_link.yaml`) in the `configurations/` folder. The default configuration is `perfect.yaml`.

2. **Setup the Experiment and Multiple Parameters:**

   - (**`--experiment`**): Choose the experiment to run (options include `cnot`, `pingpong`, `dqft2`, `dgrover2`, or `2_teleportations`). The default experiment is `cnot`.
   - (**`--epr_rounds`**): Specify the number of EPR rounds per simulation. The default number is `10`.
   - (**`--num_experiments`**): Specify the number of simulation runs per parameter combination. The default number is `10`.
   - (**`--sweep_params`**): Define the comma-separated list of parameter to sweep (e.g., single_qubit_gate_depolar_prob, two_qubit_gate_depolar_prob, T1, T2, etc.). The default parameters are `single_qubit_gate_depolar_prob`, `two_qubit_gate_depolar_prob`.
   - (**`--ranges`**): Provide for each swept parameter a valid range in the format "start,end,points". The default ranges are `"0.0,0.8,10"`, `"0.0,0.8,10"`.
   - (**`--output_dir`**): Define the path of the directory to save the results. The default folder is `results`.

3. **Execute the Simulation:**

   - Run the `run_simulation.py` script with appropriate command-line arguments. For example, this commad will launch the simulation with the default configuration, experiment and parameters:
     ```bash
     python -m experiments.run_simulation
     ```
   - The script creates an output subdirectory (named based on the experiment and swept parameters) that contains:
     - A CSV file with raw simulation results.
     - A TXT file with parameter-to-metric correlation values.
     - Heat map plots with the performance results.

### The Distributed Experiments

The provied experiments:

- **`cnot`**: Nonlocal CNOT gate using one ebit and one bit in each direction.
- **`2_teleportations`**: Nonlocal CNOT gate using two teleportations.
- **`pingpong`**: Ping-pong quantum teleportation between Alice and Bob.
- **`dqft2`**: Distributed Quantum Fourier Transform on 2 qubits.
- **`dgrover2`**: Distributed Grover on 2 qubits.

### Example Command 1: Nonlocal CNOT (Depolarization Sweep)

```bash
python -m experiments.run_simulation \
  --config configurations/perfect.yaml \
  --experiment cnot \
  --epr_rounds 10 \
  --num_experiments 100 \
  --sweep_params single_qubit_gate_depolar_prob,two_qubit_gate_depolar_prob \
  --ranges "0.0,0.8,10" "0.0,0.8,10"
```

### Example Result (Heat map) from command 1: Nonlocal CNOT (Depolarization Sweep)

<img src=docs/cnot_heat_fidelity.png width="60%" height="60%">

### Example Command 2: Distributed Grover on 2 qubits (Depolarization Sweep)

```bash
python -m experiments.run_simulation \
  --config configurations/perfect.yaml \
  --experiment dgrover2 \
  --epr_rounds 10 \
  --num_experiments 100 \
  --sweep_params single_qubit_gate_depolar_prob,two_qubit_gate_depolar_prob \
  --ranges "0.0,0.8,10" "0.0,0.8,10"
```

### Example Result (Heat map) from command 2: Distributed Grover on 2 qubits (Depolarization Sweep)

<img src=docs/dgrover2_heat_fidelity.png width="60%" height="60%">

## References

<a id="1">[1]</a>
Eisert, Jens & Jacobs, Karel & Papadopoulos, Periklis & Plenio, M.. (2000). Optimal local implementation of nonlocal quantum gates. Phys. Rev. A. 62. 10.1103/PhysRevA.62.052317.
