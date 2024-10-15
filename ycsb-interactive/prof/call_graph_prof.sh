rm -f ./callgrind.out.*
valgrind --tool=callgrind --trace-children=yes --simulate-cache=yes ../out-perf.masstree/benchmarks/dbtest --bench tpcc --retry-aborted-transactions --parallel-loading --db-type ndb-proto2 --backoff-aborted-transactions --scale-factor 1 --bench-opts "--workload-mix 50,50,0,0,0 --new-order-remote-item-pct 10" --num-threads 8 --runtime 5 --policy ../archive/investigate_conflict_tracking/5_ignore_conflicts_due_to_promote/bo14_new.txt
CALLGRIND_OUTPUT=$(ls callgrind.out.*)
PID=$(echo $CALLGRIND_OUTPUT | sed 's/callgrind.out.//')
echo $PID
callgrind_annotate --inclusive=no --auto=yes callgrind.out.$PID > callgrind_annotated.out
#../FlameGraph/stackcollapse.pl callgrind_annotated.out > out.folded