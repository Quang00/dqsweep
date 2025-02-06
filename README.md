# Analysis of fidelity/latency of multiple quantum network experiments

This project is a tool to analyze the effects of multiple parameters of a quantum network using netsquid and squidasm for some experiments (Distributed CNOT, Teleportation CNOT, Ping pong teleportation).

## Installation




## Examples of command to run the program
---

```bash
python3 experiments/run_simulation.py --config configurations/config.yaml
```

```bash
python3 experiments/run_simulation.py --config configurations/generic_qdevice.yaml --plot_parameter_effects single_qubit_gate_depolar_prob two_qubit_gate_depolar_prob --num_experiments 100
```

```bash
python3 experiments/run_simulation.py --config configurations/generic_qdevice.yaml --exepriment teleportation --plot_parameter_effects single_qubit_gate_depolar_prob two_qubit_gate_depolar_prob --num_experiments 100
```

```bash
python3 experiments/run_simulation.py --config configurations/depolarise_link.yaml --plot_parameter_effects fidelity prob_success --param1_range 1.0,0.3,10 --param2_range 1.0,0.1,10 --num_experiments 100
```

```bash
python3 experiments/run_simulation.py --config configurations/config.yaml --plot_parameter_effects length T1 --param1_range 1,10,10 --param2_range 7,9,10
```
