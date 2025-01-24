from application import AliceProgram, BobProgram

from squidasm.run.stack.config import StackNetworkConfig
from squidasm.run.stack.run import run

import numpy as np

# import network configuration from file
cfg = StackNetworkConfig.from_file("nv_qdevice.yaml")

# Set a parameter, the number of epr rounds, for the programs
epr_rounds = 10
alice_program = AliceProgram(num_epr_rounds=epr_rounds)
bob_program = BobProgram(num_epr_rounds=epr_rounds)

num_times = 10
# Run the simulation. Programs argument is a mapping of network node labels to programs to run on that node
_, fidelities = run(config=cfg, programs={"Alice": alice_program, "Bob": bob_program}, num_times=num_times)
fidelities = np.array(fidelities)

print(f"Average fidielty of distributed CNOT gate between 2 nodes and {fidelities.size} experiments: {np.average(fidelities) * 100}%")
