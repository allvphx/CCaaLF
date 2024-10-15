import os
import random
import sys
import math
import time
import shutil
import itertools
import functools
import re
import signal
import subprocess
import numpy as np
import tensorflow.compat.v1 as tf

from collections import defaultdict
from bayes_opt import BayesianOptimization, UtilityFunction

eps = 1e-5
REGEX_THPT = re.compile('throughput\(([^)]+)\)')
REGEX_ABRT = re.compile('agg_abort_rate\(([^)]+)\)')
MASK_BITS = 0
MAX_MASK = 1 << MASK_BITS
K_BITS = 3
MAX_K = 1 << K_BITS
STEP_BITS = 4
MAX_STEPS = 1 << STEP_BITS
MAX_STATE = 1 << (MASK_BITS + K_BITS + STEP_BITS)
best_seen_list = []
offset = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 0, 1, 2, 3]


def encode_state(c, k, s):
    return c | (k << MASK_BITS) | (s << (MASK_BITS + K_BITS))


def parse(return_string):
    if return_string is None: return (0.0, 0.0)
    parse_thpt = re.search(REGEX_THPT, return_string)
    parse_abrt = re.search(REGEX_ABRT, return_string)
    if parse_thpt is None or parse_abrt is None:
        return (float(.0), float(.0))
    thpt = parse_thpt.groups()[0]
    abrt = parse_abrt.groups()[0]
    return (float(thpt), float(abrt))


def run(command, die_after=0):
    print("running = ", command)
    extra = {} if die_after == 0 else {'preexec_fn': os.setsid}
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        shell=True, **extra)
    for _ in range(die_after if die_after > 0 else 600):
        if process.poll() is not None:
            break
        time.sleep(1)

    out_code = -1 if process.poll() is None else process.returncode

    if out_code < 0:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except Exception as e:
            print('{}, but continueing'.format(e))
        assert die_after != 0, 'Should only time out with die_after set'
        print('Failed with return code {}'.format(process.returncode))
        process.stdout.flush()
        return process.stdout.read().decode('utf-8')
        # return None # Timeout
    elif out_code > 0:
        print('Failed with return code {}'.format(process.returncode))
        process.stdout.flush()
        return process.stdout.read().decode('utf-8')
        # print(process.communicate())
        # return None

    process.stdout.flush()
    return process.stdout.read().decode('utf-8')


class State(object):
    def __init__(self, _policy):
        self._policy = list(_policy)
        self._hash = hash(self.__str__())

    def write_to_file(self, f):
        f.write("conflict detection (0 for deferred, 1 for immediate):\n")
        f.writelines([str(int(value * 2 - eps)) for value in self._policy[:MAX_STATE]])
        f.write("\n")
        f.write("conflict resolve (priorities):\n")
        f.writelines(["%.3f " % value for value in self._policy[MAX_STATE:2*MAX_STATE]])
        f.write("\n")
        f.write("conflict resolve (timeout):\n")
        f.writelines([str(int(value * 10 - eps)) for value in self._policy[2*MAX_STATE:]])
        f.write("\n")

    @property
    def policy(self):
        return self._policy


def encode(mask, k, step):
    return mask | (k << MASK_BITS) | (step << (MASK_BITS + K_BITS))


def shuffle(li):
    np.random.shuffle(li)
    return li


wait_die = [1.0 if i < MAX_STATE or i >= 2* MAX_STATE else 0 for i in range(3 * MAX_STATE)]
occ = [0 for i in range(3 * MAX_STATE)]
no_wait = [1.0 if i < MAX_STATE else 1.0 for i in range(3 * MAX_STATE)]
ldsf = [1.0 if i < 2*MAX_STATE else 1.0 for i in range(3*MAX_STATE)]
for i in range(MAX_MASK):
    for j in range(MAX_K):
        for k in range(MAX_STEPS):
            ldsf[MAX_STATE + encode_state(i, j, k)] = j / 10.0

mlf = [1.0 if i < 2*MAX_STATE else 1.0 for i in range(3*MAX_STATE)]
for i in range(MAX_MASK):
    for j in range(MAX_K):
        for k in range(MAX_STEPS):
            mlf[MAX_STATE + encode_state(i, j, k)] = offset[k] / 10.0

iter_ = 0
db_runtime = 5
best_seen = 0
log_rate = 1
max_non_increasing = 10
non_increase_steps = 0
training_stage = 0


def learn(command, fin_log_dir):
    print(fin_log_dir)
    tf.disable_eager_execution()
    sess = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(fin_log_dir, sess.graph)

    # Start the timer
    def black_box_function(**params):
        global db_runtime, best_seen, non_increase_steps, training_stage
        # for cases where dependency exists.
        x = [value for _, value in enumerate(params.values())]
        # if training_stage == 0:
        #     for i in range(MAX_STATE):
        #         x[i] = 1.0
        #         x[i+2*MAX_STATE] = 1.0
        # elif training_stage == 1:
        #     for i in range(MAX_STATE):
        #         x[i] = 1.0
        # for cases where dependency does not exist.
        global iter_
        base_dir = './training/bo_steps/'
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        recent_path = os.path.join(base_dir, 'step_{}.txt'.format(iter_))
        if os.path.exists(recent_path):
            os.remove(recent_path)
        policy = State(x)
        save_policy(recent_path, policy)

        iter_ += 1

        if training_stage == 0:
            command.append('--runtime {} --policy {}'.format(db_runtime, recent_path))
        elif training_stage == 1:
            command.append('--runtime {} --policy {}'.format(db_runtime, recent_path))
        else:
            command.append('--runtime {} --policy {}'.format(db_runtime, recent_path))
        sys.stdout.flush()
        # print(command)
        run_results = parse(run(' '.join(command), die_after=60))
        if run_results[0] == 0:
            print("panic: the running has been blocked for more than 1 minute")
        command.pop()
        if run_results[0] > best_seen:
            best_seen = run_results[0]
            best_seen_list.append(policy)
            non_increase_steps = 0
            save_model(fin_log_dir, policy, 'bo{}'.format(iter_))
        else:
            non_increase_steps += 1
            if non_increase_steps == max_non_increasing:
                training_stage += 1


        if iter_ % log_rate == 0:
            sc = tf.Summary()
            sc.value.add(tag='best-seen',
                         simple_value=best_seen)
            sc.value.add(tag='current',
                         simple_value=run_results[0])
            writer.add_summary(sc, iter_)
            writer.flush()
        # print("f(X) = ", run_results[0])
        return run_results[0]

    pbounds = {'x{}'.format(i): (0, 1) for i in range(3 * MAX_STATE)}

    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        random_state=42,
        verbose=1,
    )

    optimizer.probe(params=wait_die, lazy=True)
    optimizer.probe(params=no_wait, lazy=True)
    optimizer.probe(params=ldsf, lazy=True)
    optimizer.probe(params=mlf, lazy=True)
    optimizer.probe(params=occ, lazy=True)
    optimizer.set_gp_params(alpha=1e-3)
    # as suggested in http://bayesian-optimization.github.io/BayesianOptimization/advanced-tour.html
    # change alpha to accommodate noise introduced by descrete value.

    acquisition_function = UtilityFunction(kind="ucb")  # balanced exploration
    start_time = time.time()
    optimizer.maximize(
        n_iter=150,
        init_points=0,
        acquisition_function=acquisition_function
    )
    training_time = time.time() - start_time
    print("Training time for exploration: {:.2f} seconds".format(training_time))


def save_model(log_dir, policy, name):
    print('Saving model!!!!!!!')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    base_path = os.path.join(os.getcwd(), log_dir)
    recent_path = os.path.join(base_path, '{}_new.txt'.format(name))
    last_path = os.path.join(base_path, '{}_old.txt'.format(name))
    if os.path.exists(last_path):
        os.remove(last_path)
    if os.path.exists(recent_path):
        os.rename(recent_path, last_path)
    save_policy(recent_path, policy)
    print('Model saved in path: {}'.format(recent_path))


def save_policy(path, policy):
    with open(path, 'w') as f:
        policy.write_to_file(f)
