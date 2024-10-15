import subprocess

num_runs = 1

result_file = "tpcc_ic3_experiment_results.txt"

base_command = "./out-perf.masstree/benchmarks/dbtest --bench tpcc --retry-aborted-transactions --parallel-loading " \
               "--backoff-aborted-transactions --scale-factor {warehouse} --bench-opts \"\" " \
               "--num-threads {num_threads} --runtime {runtime} --policy ./training/samples/ic3.txt"

with open(result_file, 'w') as f:
    for th in [1, 2, 4, 8, 16]:
        for run in range(num_runs):
            command = base_command.format(num_threads=th, warehouse=1, runtime=5)
            # print("running = ", command)
            print(f"Running experiment {run + 1} for WH: {1} TH: {th}")
            result = subprocess.run(command.encode('utf-8'), shell=True, stdout=subprocess.PIPE,
                                    universal_newlines=True)
            if "RESULT" in result.stdout:
                output_line = result.stdout.strip().split("RESULT")[-1]
                f.write(f"Experiment {run + 1} WH: {1} TH: {th} RESULT {output_line}\n")
                f.flush()
            else:
                print("panic: cannot find result inside experiment result")

    for wh in [2, 4, 8]:
        for run in range(num_runs):
            command = base_command.format(num_threads=16, warehouse=wh, runtime=5)
            print(f"Running experiment {run + 1} for WH: {wh} TH: {16}")
            result = subprocess.run(command.encode('utf-8'), shell=True, stdout=subprocess.PIPE,
                                    universal_newlines=True)
            if "RESULT" in result.stdout:
                output_line = result.stdout.strip().split("RESULT")[-1]
                f.write(f"Experiment {run + 1} WH: {wh} TH: {16} RESULT {output_line}\n")
                f.flush()
            else:
                print("panic: cannot find result inside experiment result")


print("All experiments completed.")
