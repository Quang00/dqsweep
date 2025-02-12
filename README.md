# Analysis of fidelity/latency of multiple quantum network experiments

This project is a tool to analyze the effects of multiple parameters of a quantum network using netsquid and squidasm for some experiments (Distributed CNOT, Teleportation CNOT, Ping pong teleportation).

## Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --U pip
export NETSQUIDPYPI_USER=user1234
export NETSQUIDPYPI_PWD=password1234
git clone git@github.com:QuTech-Delft/squidasm.git
cd squidasm
make install
```

## Examples of command
---

```bash
python experiments/run_simulation.py \
  --config configurations/perfect.yaml \
  --experiment cnot \
  --epr_rounds 10 \
  --num_experiments 100 \
  --sweep_params single_qubit_gate_depolar_prob,two_qubit_gate_depolar_prob \
  --ranges "0.0,0.8,10" "0.0,0.8,10"
```

```bash
python experiments/run_simulation.py \
  --config configurations/perfect.yaml \
  --experiment cnot \
  --epr_rounds 10 \
  --num_experiments 100 \
  --sweep_params single_qubit_gate_time,two_qubit_gate_time \
  --ranges "0.0,0.8,10" "0.0,0.8,10"
```

```bash
python experiments/run_simulation.py \
  --config configurations/perfect.yaml \
  --experiment pingpong \
  --epr_rounds 10 \
  --num_experiments 10 \
  --sweep_params T1,T2,single_qubit_gate_depolar_prob,two_qubit_gate_depolar_prob \
  --ranges "1e10,10e7,10" "1e10,10e7,10" "0.0,0.8,10" "0.0,0.8,10"
```
