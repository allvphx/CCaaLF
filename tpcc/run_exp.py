import subprocess

# Number of repetitions for each experiment
num_runs = 5

# Base command template
base_command = "python training/flexi_policy_benchmark.py --nworkers={num_threads} --scale-factor {warehouse}" \
               " --state-space {space} --encoder " \
               "./encoder/default_tpcc_encoder.txt --bench-opts \"\" --workload-type tpcc"
for wh in [1, 2, 4, 8]:
    for run in range(num_runs):
        command = base_command.format(num_threads=16, warehouse=wh, space=26)
        print("running = ", command)
        print(f"Running experiment {run + 1} for WH: {wh} TH: {1}")
        result = subprocess.run(command.encode('utf-8'), shell=True)

for th in [1, 2, 4, 8]:
    for run in range(num_runs):
        command = base_command.format(num_threads=th, warehouse=1, space=26)
        print("running = ", command)
        print(f"Running experiment {run + 1} for WH: {1} TH: {th}")
        result = subprocess.run(command.encode('utf-8'), shell=True)

print("All experiments completed.")
