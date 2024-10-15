#!/usr/bin/env python
import argparse
import numpy as np
import utils as utils

from bo import learn
from bo_only_learn_priority import learn_priority
from pso import pso_learn

def main(args):
    np.random.seed(args.seed)

    cfg = utils.setup(args)

    command = ['./out-perf.masstree/benchmarks/dbtest --bench {} --retry-aborted-transactions --parallel-loading '
               '--db-type ndb-proto2 --backoff-aborted-transactions --scale-factor {} --bench-opts "{}" '
               '--num-threads {}'.format(
                args.workload_type, args.scale_factor, args.bench_opt, args.nworkers)]

    learn_priority(command, cfg.get('log_directory'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--base-log-dir', type=str, default='./training/bo-all',
                        help='model save location')
    parser.add_argument('--base-kid-dir', type=str, default='./training/bo',
                        help='kid policy save location')
    parser.add_argument('--expr-name', type=str, default='bo',
                        help='experiment name')
    parser.add_argument('--seed', help='RNG seed', type=int, default=42)

    # Experiment setting arguments
    parser.add_argument('--workload-type', type=str, default='tpcc',
                        choices=['tpcc', 'tpce', 'ycsb'],
                        help='number of database workers')
    parser.add_argument('--nworkers', type=int, default=8,
                        help='number of database workers'                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        )
    parser.add_argument('--scale-factor', type=int, default=1,
                        help='scale factor')
    parser.add_argument('--bench-opt', type=str, default='--workload-mix 50,50,0,0,0 --new-order-remote-item-pct 10',
                        help='benchmark info, e.g. workload mix ratio')

    main(parser.parse_args())
