import json
import os
import sys
import warnings
import nevergrad as ng
import time
import numpy as np

# from chop_helper import *
from utils import run, parse
from torch.utils.tensorboard import SummaryWriter

warnings.simplefilter(action='ignore', category=FutureWarning)

MUTATION_MAX_STEP = 2
nearly_linear = 1.1


# I can learn the chop in another way!!
# 1. First round, guard point injection on all three transactions. wait access is just 0/1.
# If one transaction's guard point goes to the end --> second.


class CCLearner(object):

    def setup(self, k, v):
        assert k in ["expose", "wait", "wait_guard", "rank", "access", "timeout"]
        self.setting[k] = v
        self.set_bounds()

    def update_setting(self, setting):
        self.setting = setting
        self.patience = setting.get("patient", self.patience)
        self.set_bounds()

    def set_bounds(self):
        check_length = 0

        # Access policy parameters (adaptive field)
        access_parameters = []
        if self.setting["access"]:
            access_parameters.extend([ng.p.Choice([0, 1, 2, 3], repetitions=self.max_state)])
            check_length += self.max_state

        # Priority policy parameters (adaptive field)
        rank_parameters = []
        if self.setting["rank"]:
            rank_parameters.extend([ng.p.Scalar(lower=0, upper=1)
                                   .set_mutation(sigma=0.2, exponent=None) for _ in range(self.max_state)])
            check_length += self.max_state

        # Timeout policy parameters (adaptive field)
        timeout_parameters = []
        if self.setting["timeout"]:
            # for apple-to-apple comparison, we also add policies in PolyJuice.
            timeout_parameters.extend([ng.p.Scalar(lower=100,
                                                   upper=1000000, init=100000) for _ in range(self.max_state)])
            check_length += self.max_state

        # Combine the policies into the Instrumentation
        self.bounds = ng.p.Instrumentation(
            access=ng.p.Tuple(*access_parameters),
            rank=ng.p.Tuple(*rank_parameters),
            timeout=ng.p.Tuple(*timeout_parameters)
        )

        self.check_encoder_length = check_length

    def __init__(self, base_command, name, log_dir, starting_points, max_state, seed, log_rate=1, _runtime=1,
                 setting=None):
        if setting is None:
            setting = {"rank": False,
                       "access": False,
                       "timeout": False}
        self.max_state = max_state
        self.seed = seed
        self.current_iter = 0
        #
        self.searched_points_hash = {}
        self.setting = setting
        self.best_seen_performance = 0
        self.log_rate = log_rate
        self.start_time = time.time()
        self.check_encoder_length = 0
        self.db_runtime = _runtime
        self.training_stage = 0
        self.best_policy = None
        self.evaluated_history = []
        self.no_update_count = 0
        self.base_command = base_command
        self.name = name
        self.log_dir = log_dir
        self.starting_points = starting_points
        self.shrink_rate = 0.5
        self.patience = 100
        self.bounds = None
        self.set_bounds()
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def load_initial_policy_from_file(self, files):
        self.starting_points = []
        for p_dir in files:
            policy = Policy(_from=self, load_file=p_dir)
            self.starting_points.append(policy)

    def save_model(self, policy, name):
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        base_path = os.path.join(os.getcwd(), self.log_dir)
        recent_path = os.path.join(base_path, '{}_new.txt'.format(name))
        last_path = os.path.join(base_path, '{}_old.txt'.format(name))
        if os.path.exists(last_path):
            os.remove(last_path)
        if os.path.exists(recent_path):
            os.rename(recent_path, last_path)
        policy.save_to_path(recent_path)

    def evaluate_policy(self, policy):
        base_dir = './training/bo_steps/'
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        recent_path = os.path.join(base_dir, 'step_{}.txt'.format(self.current_iter))
        if os.path.exists(recent_path):
            os.remove(recent_path)
        policy.save_to_path(recent_path)
        self.current_iter += 1

        command = self.base_command
        command.append('--runtime {} --policy {}'.format(self.db_runtime, recent_path))
        sys.stdout.flush()
        run_results = parse(run(' '.join(command), die_after=180))
        if run_results[0] == 0:
            print("panic: the running has been blocked for more than 10s")
        command.pop()
        current_score = run_results[0]
        policy.score = current_score
        self.evaluated_history.append(policy)

        if current_score > self.best_seen_performance:
            self.no_update_count = 0
            print("Optimizer %s found better cc policy in iteration %d, spent time %f: %d TPS" %
                  (self.name, self.current_iter - 1, time.time() - self.start_time, current_score))
            self.best_seen_performance = current_score
            self.best_policy = policy
            self.save_model(policy, 'bo{}'.format(self.current_iter))
        else:
            self.no_update_count += 1

        if self.current_iter % self.log_rate == 0:
            self.writer.add_scalar('best-seen', self.best_seen_performance, self.current_iter)
            self.writer.add_scalar('current-seen', current_score, self.current_iter)
            self.writer.flush()
        return current_score

    def black_box_function(self, _access, _rank, _timeout, _expose, _wait_chop):
        base_dir = './training/bo_steps/'
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        recent_path = os.path.join(base_dir, 'step_{}.txt'.format(self.current_iter))
        if os.path.exists(recent_path):
            os.remove(recent_path)
        policy = Policy(_access, _rank, _timeout, _expose, _wait_chop, self)
        policy.save_to_path(recent_path)
        self.current_iter += 1

        command = self.base_command
        command.append('--runtime {} --policy {}'.format(self.db_runtime, recent_path))
        sys.stdout.flush()
        run_results = parse(run(' '.join(command), die_after=180))
        if run_results[0] == 0:
            print("panic: the running has been blocked for more than 10s")
        command.pop()
        current_score = run_results[0]
        policy.score = current_score

        if current_score > self.best_seen_performance:
            self.no_update_count = 0
            print("Optimizer %s found better cc policy in iteration %d, spent time %f: %d TPS" %
                  (self.name, self.current_iter - 1, time.time() - self.start_time, current_score))
            self.best_seen_performance = current_score
            self.save_model(policy, 'bo{}'.format(self.current_iter))
        else:
            self.no_update_count += 1

        if self.current_iter % self.log_rate == 0:
            self.writer.add_scalar('best-seen', self.best_seen_performance, self.current_iter)
            self.writer.add_scalar('current-seen', current_score, self.current_iter)
            self.writer.flush()
        return current_score

    def close(self):
        self.writer.close()


class Policy(object):
    def __init__(self, _access=None, _rank=None, _timeout=None,
                 _expose=None, _wait_chop=None, _extra=None, _from=None,
                 load_file=None, encoded=None):
        self.score = -1
        self.learner = _from
        self.max_state = self.learner.max_state
        # the default txn_buf_size and backoff parameters.
        if load_file is not None:
            with open(load_file, "r") as f:
                self.read_from_file(f)
        elif encoded is not None:
            self.decode(encoded)
        else:
            # common policies (applicable to interactive mode)
            self.access = np.array(_access)
            self.rank = np.array(_rank)
            self.timeout_policy = _timeout
        self._hash = hash(self.__str__())

    # float_correction is used to correct the .
    def float_correction(self):
        pass

    def write_to_file(self, f_out):
        self.float_correction()
        f_out.write("conflict detection:\n")
        f_out.writelines([str(int(value)) for value in self.access])
        f_out.write("\n")
        f_out.write("wait priorities:\n")
        f_out.writelines(["%.3f " % float(value) for value in self.rank])
        f_out.write("\n")
        f_out.write("timeout:\n")
        expose_str = [str(value) + ' ' for value in self.timeout_policy]
        f_out.writelines(expose_str)
        f_out.write("\n")
        f_out.write("learner encoding = \n{encoded_params}\n".format(encoded_params=self.encode()))

    def read_from_file(self, file):
        n = self.learner.max_state
        res = [0.0 for _ in range(3 * n)]
        cur_line = file.readline()
        if "conflict detection" in cur_line:
            values = file.readline().strip()
            self.access = np.array([float(value) for value in values[:n]])
            # assert len(values) == n
        cur_line = file.readline()
        if "wait priorities" in cur_line:
            values = file.readline().strip().split()
            self.rank = np.array([float(value) for value in values[:n]])
            # assert len(values) == n
        cur_line = file.readline()
        if "timeout" in cur_line:
            values = file.readline().strip().split()
            self.timeout_policy = np.array([float(value) for value in values[:n]])
            # assert len(values) == n
        return res

    def save_to_path(self, path):
        with open(path, 'w') as file:
            self.write_to_file(file)

    def encode(self):
        """Encode learner's internal state into the parameter dictionary"""
        if self.learner.best_policy is None:
            self.learner.best_policy = self

        access_params = ()
        rank_params = ()
        timeout_params = ()

        # Encode the 'access' policy parameters
        if self.learner.setting["access"]:
            access_params = (tuple(self.access),)
            # print(access_params)

        # Encode the 'rank' policy parameters
        if self.learner.setting["rank"]:
            rank_params = tuple(self.rank)

        # Encode the 'timeout' policy parameters
        if self.learner.setting["timeout"]:
            timeout_params = tuple(self.timeout_policy)

        return {
            "access": access_params,
            "rank": rank_params,
            "timeout": timeout_params
        }

    def decode(self, param_dict):
        """Decode the parameter dictionary into the learner's internal state."""
        # print(param_dict)

        # Handle 'access' policy
        if self.learner.setting["access"]:
            access_values = param_dict.get("access", None)[0]
            assert access_values is not None, "Expected access_values to be provided, but got None."
            self.access = np.array(access_values, dtype=int)
        else:
            self.access = self.learner.best_policy.access

        # Handle 'rank' policy
        if self.learner.setting["rank"]:
            rank_values = param_dict.get("rank", None)
            assert rank_values is not None, "Expected rank_values to be provided, but got None."
            self.rank = np.array(rank_values)
        else:
            self.rank = self.learner.best_policy.rank

        # Handle 'timeout' policy
        if self.learner.setting["timeout"]:
            timeout_values = param_dict.get("timeout", None)
            assert timeout_values is not None, "Expected timeout_values to be provided, but got None."
            self.timeout_policy = np.array(timeout_values[-self.max_state:])
        else:
            self.timeout_policy = self.learner.best_policy.timeout_policy

    def hash(self):
        return hash(json.dumps(self.encode(), sort_keys=True, default=convert_np))


def convert_np(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
