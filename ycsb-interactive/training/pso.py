import os
import sys
import time
import re
import signal
import subprocess
import numpy as np
import pyswarms as ps
import tensorflow.compat.v1 as tf
from pyswarms.utils.functions import single_obj as fx

eps = 1e-5
REGEX_THPT = re.compile('throughput\(([^)]+)\)')
REGEX_ABRT = re.compile('agg_abort_rate\(([^)]+)\)')
MASK_BITS = 2
MAX_MASK = 1 << MASK_BITS
K_BITS = 2
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
        f.write("conflict detection (0 for deferred, 1 for immediate, 2 for direct abort):\n")
        f.writelines([str(int(value * 2 - eps)) for value in self._policy[:MAX_STATE]])
        f.write("\n")
        f.write("conflict resolve (priorities):\n")
        f.writelines(["%.3f " % value for value in self._policy[MAX_STATE:2*MAX_STATE]])
        f.write("\n")
        f.write("early validation (0 for no validation, 1 for validation):\n")
        f.writelines([str(int(value * 2 - eps)) for value in self._policy[2*MAX_STATE:]])
        f.write("\n")

    @property
    def policy(self):
        return self._policy


def encode(mask, k, step):
    return mask | (k << MASK_BITS) | (step << (MASK_BITS + K_BITS))


def shuffle(li):
    np.random.shuffle(li)
    return li


wait_die = [0.5 if i < MAX_STATE else 0 for i in range(3*MAX_STATE)]
occ = [0 for i in range(3*MAX_STATE)]
occ_early_validation = [0 if i < 2*MAX_STATE else 1 for i in range(3*MAX_STATE)]
no_wait = [0.5 if i < MAX_STATE or i >= 2*MAX_STATE else 1 for i in range(3*MAX_STATE)]
ldsf = [0.5 if i < MAX_STATE else 0 for i in range(3*MAX_STATE)]
for i in range(MAX_MASK):
    for j in range(MAX_K):
        for k in range(MAX_STEPS):
            ldsf[MAX_STATE + encode_state(i, j, k)] = j / 10.0

mlf = [0.5 if i < MAX_STATE else 0 for i in range(3*MAX_STATE)]
for i in range(MAX_MASK):
    for j in range(MAX_K):
        for k in range(MAX_STEPS):
            mlf[MAX_STATE + encode_state(i, j, k)] = offset[k] / 10.0

bounded_wait = [0.5 if i < MAX_STATE else 0 for i in range(3*MAX_STATE)]
for i in range(MAX_MASK):
    for j in range(MAX_K):
        for k in range(MAX_STEPS):
            bounded_wait[MAX_STATE + encode_state(i, j, k)] = 1 if j == 0 else 0

iter_ = 0
n_particles = 0
db_runtime = 1
best_seen = 0
log_rate = 1
max_non_increasing = 10
non_increase_steps = 0
training_stage = 1


def pso_learn(command, fin_log_dir):
    tf.disable_eager_execution()
    sess = tf.Session()
    tf.summary.merge_all()
    writer = tf.summary.FileWriter(fin_log_dir, sess.graph)
    start_time = time.time()

    # Start the timer
    def black_box_function(x):
        global db_runtime, n_particles
        base_dir = './training/pso/'
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        recent_path = os.path.join(base_dir, 'particle_{}.txt'.format(n_particles))
        if os.path.exists(recent_path):
            os.remove(recent_path)
        policy = State(x)
        save_policy(recent_path, policy)
        n_particles += 1
        command.append('--runtime {} --policy {}'.format(db_runtime, recent_path))
        sys.stdout.flush()
        run_results = parse(run(' '.join(command), die_after=60))
        if run_results[0] == 0:
            print("panic: the running has been blocked for more than 1 minute")
        command.pop()
        return run_results[0]

    def obj_function(x: np.ndarray) -> np.ndarray:
        global iter_, n_particles, best_seen
        n_particles = 0
        res = []
        for i in range(x.shape[0]):
            res.append(black_box_function(x[i]))
        ips = np.argmax(res)
        if res[ips] > best_seen:
            best_seen = res[ips]
            policy = State(x[ips])
            best_seen_list.append(policy)
            save_model(fin_log_dir, policy, 'pso{}'.format(iter_))
        if iter_ % log_rate == 0:
            sc = tf.Summary()
            sc.value.add(tag='best-seen',
                         simple_value=best_seen)
            sc.value.add(tag='current',
                         simple_value=res[ips])
            writer.add_summary(sc, iter_)
            writer.flush()
        iter_ += 1
        return res[ips]


    lb = [0 for _ in range(3 * MAX_STATE)]
    ub = [1 for _ in range(3 * MAX_STATE)]
    bounds = (lb, ub)
    options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
    init_point_sets = np.array([wait_die, occ, occ_early_validation, bounded_wait, no_wait, ldsf, mlf])

    optimizer = ps.global_best.GlobalBestPSO(n_particles=7, init_pos=init_point_sets,
                                             dimensions=3*MAX_STATE, options=options, bounds=bounds)
    optimizer.optimize(obj_function, iters=300)

    training_time = time.time() - start_time
    print("Training time for exploration: {:.2f} seconds".format(training_time))


def save_model(log_dir, policy, name):
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


def save_policy(path, policy):
    with open(path, 'w') as f:
        policy.write_to_file(f)
