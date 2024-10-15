import subprocess
import training.chop_helper as cp
from settings import *

# Number of repetitions for each experiment
num_runs = 5

# Base command template
base_command = "python training/flexi_policy_benchmark.py --nworkers=16 " \
               "--scale-factor 10 --state-space {space} --encoder " \
               "./encoder/default_ycsb_encoder.txt --bench-opts \"--length {txn_length} " \
               "--access-dist {dist} --partition --opt-dist {opts}\" --workload-type ycsb"

# Run the experiments
for setting in txn_length_exp + rw_rate_exp:
    for run in range(num_runs):
        update_setting(setting["opts"])
        command = base_command.format(dist=setting["access"],
                                      opts=setting["opts"],
                                      txn_length=(len(setting["opts"]) + 1) / 2,
                                      space=int((len(setting["opts"]) + 1) / 2))
        print("Current command = ", command)
        print(f"Running experiment {run + 1} for Setting: {str(setting)}")
        subprocess.run(command, shell=True)

print("All experiments completed.")
