import subprocess

result_file = "tpcc_drift.txt"

base_command = "./out-perf.masstree/benchmarks/dbtest --bench tpcc --retry-aborted-transactions --parallel-loading " \
               "--backoff-aborted-transactions --scale-factor {warehouse} --bench-opts \"\" " \
               "--num-threads {num_threads} --runtime {runtime} --policy ./test_policies/{run_num_threads}th_{" \
               "run_warehouse}wh.txt"

settings = []


def run_exp(th, wh, run_th, run_wh):
    print(f"Running experiment for WH: {wh} TH: {th}")
    command = base_command.format(runtime=1, num_threads=th, warehouse=wh,
                                  run_num_threads=run_th, run_warehouse=run_wh)
    result = subprocess.run(command.encode('utf-8'), shell=True, stdout=subprocess.PIPE,
                            universal_newlines=True)
    if "RESULT" in result.stdout:
        output_line = result.stdout.strip().split("RESULT")[-1]
        f.write(f"Experiment WH: {wh} TH: {th}, AIM WH: {run_wh} TH: {run_th} RESULT {output_line}\n")
        f.flush()
    else:
        print("panic: cannot find result inside experiment result")


with open(result_file, 'w') as f:
    for th in [1, 2, 4, 8, 16]:
        settings.append({"th": th, "wh": 1})
    for wh in [2, 4, 8]:
        settings.append({"th": 16, "wh": wh})
    for i in range(len(settings)):
        run_exp(settings[i]["th"], settings[i]["wh"],
                settings[i]["th"], settings[i]["wh"])
        for j in range(len(settings)):
            if i != j:
                run_exp(settings[i]["th"], settings[i]["wh"],
                        settings[j]["th"], settings[j]["wh"])
print("All experiments completed.")
