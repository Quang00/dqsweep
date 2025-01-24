# Distributed CNOT Gate Simulation and Parameter Analysis

This project simulates a distributed CNOT gate using a quantum network and allows analysis of the effect of specific quantum network parameters on the fidelity of the operation.

---

```bash
python run_simulation.py --config config.yaml

python run_simulation.py --config generic_qdevice.yaml --plot_parameter_effects single_qubit_gate_depolar_prob two_qubit_gate_depolar_prob

python run_simulation.py --config depolarise_link.yaml --plot_parameter_effects fidelity prob_success --param1_range 1.0,0.3,10 --param2_range 1.0,0.1,10

python run_simulation.py --config heraldedl_link.yaml --plot_parameter_effects length p_loss_length --param1_range 1.0,20.0,10 --param2_range 0.1,0.8,10

```