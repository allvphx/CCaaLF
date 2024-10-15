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

result_file = "tpcc_ic3_experiment_results.txt"

base_command = "./out-perf.masstree/benchmarks/dbtest --bench ycsb --retry-aborted-transactions --parallel-loading " \
               "--backoff-aborted-transactions --scale-factor 10 --runtime 5 --encoder " \
               "./encoder/default_ycsb_encoder.txt" \
               " --bench-opts \"--length 10 --access-dist {dist} --partition --opt-dist {opts}\" " \
               "--policy ./training/samples/ic3.txt"

# Run the experiments
for dist in dist_values:
    for run in range(num_runs):
        command = base_command.format(dist=dist, opts=dist_opts_str, space=N_ACCESS)
        print(command)
        print(f"Running experiment {run + 1} for DIST: {dist}")
        subprocess.run(command, shell=True)

with open(result_file, 'w') as f:
    for dist in dist_values:
        for run in range(num_runs):
            command = base_command.format(dist=dist, opts=dist_opts_str, space=N_ACCESS)
            print(command)
            print(f"Running experiment {run + 1} for DIST: {dist}")
            result = subprocess.run(command.encode('utf-8'), shell=True, stdout=subprocess.PIPE,
                                    universal_newlines=True)
            if "RESULT" in result.stdout:
                output_line = result.stdout.strip().split("RESULT")[-1]
                f.write(f"Experiment {run + 1} for DIST: {dist} RESULT {output_line}\n")
                f.flush()
            else:
                print("panic: cannot find result inside experiment result")

print("All experiments completed.")
