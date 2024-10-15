import subprocess

# Number of repetitions for each experiment
num_runs = 5

# Base command template
base_command = "python training/flexi_policy_benchmark.py --nworkers={num_threads}" \
               " --bench-opt " \
               "\"-w 0,0,0,0,0,0,50,0,0,50 -m 0 -s 1 -a {theta}\" "

for sk in [0, 2, 4, 6]:
    for run in range(num_runs):
        command = base_command.format(num_threads=16, theta=sk)
        print("running = ", command)
        print(f"Running experiment {run + 1} for THE: {sk} TH: {16}")
        result = subprocess.run(command.encode('utf-8'), shell=True)


for th in [1, 2, 4, 8]:
    for run in range(num_runs):
        command = base_command.format(num_threads=th, theta=6)
        print("running = ", command)
        print(f"Running experiment {run + 1} for THE: {6} TH: {th}")
        result = subprocess.run(command.encode('utf-8'), shell=True)

print("All experiments completed.")
