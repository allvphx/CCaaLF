#!/usr/bin/env python
import argparse
import glob

import numpy as np
import utils as utils
# from bo import learn
from ng import benchmark, training

MAX_STATE_ALLOWABLE = 2000


def main(args, encoder, state_size):
    np.random.seed(args.seed)

    cfg = utils.setup(args)

    if args.state_space != 0:
        state_size = args.state_space

    if args.encoder != '':
        encoder = args.encoder

    command = ['./out-perf.masstree/benchmarks/dbtest --bench {} --retry-aborted-transactions --parallel-loading '
               ' --backoff-aborted-transactions --scale-factor {} --bench-opts "{}" --num-threads {} --encoder {}'.
               format(args.workload_type, args.scale_factor, args.bench_opts, args.nworkers, encoder)]

    return training(command, cfg.get('log_directory'), state_size, args.pickup_policy)


def evaluate_encoder(encoder="./encoder/default_encoder_tpcc.txt", state_size=0):
    # return 1, "test_load"
    if state_size > MAX_STATE_ALLOWABLE:
        return 0, "exceed"
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
    parser.add_argument('--nworkers', type=int, default=8, help='number of database workers')
    parser.add_argument('--pickup-policy', type=str,
                        default=glob.glob('./training/samples/*'),
                        help='the initial policy to start with')
    parser.add_argument('--scale-factor', type=int, default=1,
                        help='scale factor')
    parser.add_argument('--state-space', type=int, default=26,
                        help='state space for policy searching')
    parser.add_argument('--bench-opts', type=str, default='--workload-mix 45,43,4,4,4',
                        help='benchmark info, e.g. workload mix ratio')
    parser.add_argument('--encoder', type=str, default='./encoder/default_tpcc_encoder.txt',
                        help='the cc feature encoding method')
    return main(parser.parse_args(), encoder, state_size)


if __name__ == '__main__':
    evaluate_encoder()

