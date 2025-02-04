# Distributed CNOT Gate Simulation and Parameter Analysis

This project simulates a distributed CNOT gate using a quantum network and allows analysis of the effect of specific quantum network parameters on the fidelity of the operation.

## Installation




## Examples of command to run the program
---

```bash
python run_simulation.py --config configurations/config.yaml
```

```bash
python run_simulation.py --config configurations/generic_qdevice.yaml --plot_parameter_effects single_qubit_gate_depolar_prob two_qubit_gate_depolar_prob --num_experiments 100
```

```bash
python run_simulation.py --config configurations/depolarise_link.yaml --plot_parameter_effects fidelity prob_success --param1_range 1.0,0.3,10 --param2_range 1.0,0.1,10 --num_experiments 100
```

```bash
python run_simulation.py --config configurations/config.yaml --plot_parameter_effects length T1 --param1_range 1,10,10 --param2_range 7,9,10
```
