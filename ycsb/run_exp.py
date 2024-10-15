import subprocess
from training.chop_helper import *

dist_values = [
    "0,0,0,1,0,0,0,0,0,0",
    "1,0,0,1,1,0,0,0,0,0",
    "0,0,0,1,0,1,0,1,1,0",
    "1,1,0,0,1,0,0,1,1,0"
]

# Number of repetitions for each experiment
num_runs = 5

# Base command template
base_command = "python training/flexi_policy_benchmark.py --nworkers=16 " \
               "--scale-factor 10 --state-space {space} --encoder " \
               "./encoder/default_ycsb_encoder.txt --bench-opts \"--length 10 " \
               "--access-dist {dist} --partition --opt-dist {opts}\" --workload-type ycsb"

# Run the experiments
for dist in dist_values:
    for run in range(num_runs):
        command = base_command.format(dist=dist, opts=dist_opts_str, space=N_ACCESS)
        print(command)
        print(f"Running experiment {run + 1} for DIST: {dist}")
        subprocess.run(command, shell=True)

print("All experiments completed.")
