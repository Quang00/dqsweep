# dqsweep

![Pytest and Flake8 validation](https://github.com/Quang00/DQC/actions/workflows/python-app.yml/badge.svg)
[![Coverage Status](https://img.shields.io/coveralls/Quang00/dqsweep.svg?logo=Coveralls)](https://coveralls.io/r/Quang00/dqsweep)
![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-ffdd54?logo=python&logoColor=ffdd54)

This repository is designed to analyze the performance (fidelity and latency) of multiple distributed quantum experiments over a configurable quantum network. It uses quantum network simulators such as [NetSquid](https://netsquid.org/) and [SquidASM](https://github.com/QuTech-Delft/squidasm).

The research poster below introduces some distributed quantum computing concepts used in this repository:

<img src=docs/poster_distributed_QC.png width="100%" height="100%">

## Overview

The repository explores the performance of several quantum distributed experiments by sweeping given parameters (such as depolarization probabilities, gate times, and qubit coherence times) from a quantum network and then assess two metrics:

- **Average Fidelity (%):** Density matrix output compared to the density matrix expected.
- **Average Simulation Time (ms):** Simulation time to execute the experiment.

The experiments implemented in this repository include:

- **Nonlocal CNOT Gate with Two Teleportations (`nonlocal_cnot_teledata.py`):** Implementation of the distributed CNOT gate using two quantum teleportations.
- **Nonlocal CNOT Gate with Telegate (`nonlocal_cnot_telegate.py`):** Implementation of a distributed CNOT gate between Alice and Bob presented in the paper [[1]](#1).
- **Distributed Grover on Two Qubits (`dgrover_2.py`):** Implementation of a distributed Grover on two qubits with an initial pingpong quantum teleportation exchange where Alice initiates the state of Bob's qubit.
- **Distributed Grover on N Qubits (`dgrover.py`):** General implementation of a distributed Grover on n qubits with an oracle that marks the state |1...1>.
- **Distributed Quantum Fourier Transform (DQFT) on Two Qubits (`dqft_2.py`):** Implementation of a distributed QFT on two qubits.
- **Ping-Pong Teleportation (`pingpong.py`):** A bidirectional quantum teleportation where a qubit is sent back and forth between Alice and Bob.
- **Nonlocal Toffoli Gate (`nonlocal_toffoli.py`)**: Implementation of a distributed Toffoli gate between Alice, Bob and Charlie.

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

The experiments are executed through the `run_simulation.py` script, which does the parameter sweeps in parallel, runs the given distributed quantum experiment, and produces the results in a folder containing different files. To run the simulation:

1. **Configure A Quantum Network:**

- (**`--config`**): Provide a valid quantum network configuration or use the ones already provided (e.g., `perfect.yaml`, `depolarise_link.yaml`) in the `configurations/` folder. The default configuration is `perfect.yaml`.

2. **Setup the Experiment and Multiple Parameters:**

- (**`--experiment`**): Choose the experiment to run (options include `cnot_teledata`, `cnot_telegate`, `dgrover`, `dgrover2`, `dqft2`, `pingpong`, and `toffoli` ). The default experiment is `cnot_telegate`.
- (**`--epr_rounds`**): Specify the number of EPR rounds per simulation. The default number is `10`.
- (**`--num_experiments`**): Specify the number of simulation runs per parameter combination. The default number is `10`.
- (**`--sweep_params`**): Define the comma-separated list of parameter to sweep (e.g., single_qubit_gate_depolar_prob, two_qubit_gate_depolar_prob, T1, T2, etc.). The default parameters are `single_qubit_gate_depolar_prob,two_qubit_gate_depolar_prob`.
- (**`--ranges`**): Provide for each swept parameter a valid range in the format "start,end,points". The default ranges are `"0.0,0.8,10" "0.0,0.8,10"`.
- (**`--output_dir`**): Define the path of the directory to save the results. The default folder is `results`.

3. **Execute the Simulation:**

- Run the `run_simulation.py` script with the setup. The basic command to launch the simulation with the default configuration, experiment and parameters (see below for a complete command line):
  ```bash
  python -m experiments.run_simulation
  ```
- The script creates an output subdirectory (named based on the experiment and swept parameters) that contains:
  - A CSV file with raw simulation results.
  - A TXT file with parameter-to-metric correlation values.
  - Heat map plots with the performance results.

### The Distributed Experiments

The provied experiments:

- **`cnot_teledata`**: Nonlocal CNOT gate using two quantum teleportations.
- **`cnot_telegate`**: Nonlocal CNOT gate using telegate.
- **`dgrover`**: General Distributed Grover on n qubits that searches |1...1>.
- **`dgrover2`**: Distributed Grover on 2 qubits with an initial pingpong exchange between Alice and Bob.
- **`dqft2`**: Distributed Quantum Fourier Transform on 2 qubits.
- **`pingpong`**: Ping-pong quantum teleportation between Alice and Bob.
- **`toffoli`**: Nonlocal Toffoli gate using two ebits and four bits in each direction.

### Example Command 1: Nonlocal CNOT (Depolarization Sweep)

```bash
python -m experiments.run_simulation \
  --config configurations/perfect.yaml \
  --experiment cnot \
  --epr_rounds 10 \
  --num_experiments 100 \
  --sweep_params single_qubit_gate_depolar_prob,two_qubit_gate_depolar_prob \
  --ranges "0.0,0.8,10" "0.0,0.8,10" \
  --output_dir results
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
  --ranges "0.0,0.8,10" "0.0,0.8,10" \
  --output_dir results
```

### Example Result (Heat map) from command 2: Distributed Grover on 2 qubits (Depolarization Sweep)

<img src=docs/dgrover2_heat_fidelity.png width="60%" height="60%">

## References

<a id="1">[1]</a>
Eisert, Jens & Jacobs, Karel & Papadopoulos, Periklis & Plenio, M.. (2000). Optimal local implementation of nonlocal quantum gates. Phys. Rev. A. 62. 10.1103/PhysRevA.62.052317.
