import copy
import json
import os
import sys
import warnings
import nevergrad as ng
import numpy as np
import nevergrad.common.typing as tp

from chop_helper import *
from utils import run, parse
from torch.utils.tensorboard import SummaryWriter
from sc_graph_helper import calculate_wait_access

warnings.simplefilter(action='ignore', category=FutureWarning)

MUTATION_MAX_STEP = 2
nearly_linear = 1.1
max_try_from_a_chop = N_ACCESS * 5


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
        self.graphic_reduction = setting.get("graphic_reduction", self.graphic_reduction)
        self.pop_size = setting.get("pop_size", self.pop_size)
        self.mutate_rate = setting.get("mutate_rate", self.mutate_rate)
        self.branching_factor = setting.get("branching_factor", self.branching_factor)
        self.set_bounds()

    def set_bounds(self):
        check_length = 0

        # Expose policy parameters (N_CRITICAL_EXPOSE IC3 chopped) ~ 17 in TPCC.
        expose_parameters = []
        if self.setting["expose"]:
            expose_parameters.extend([ng.p.Choice([0, 1], repetitions=N_CRITICAL_EXPOSE)])
            check_length += N_CRITICAL_EXPOSE

        # Wait policy parameters (base_num * N_TX_TYPE * (N_TX_TYPE + 1)) ~ 12 in TPCC.
        wait_parameters = []
        if self.setting["wait"]:
            base_num = self.setting["wait_guard"]
            if base_num > 0:
                assert False    # This branch no longer supported.
                # # only loading guard points
                # wait_parameters.extend(
                #     [ng.p.Scalar(lower=0,
                #                  upper=TXN_CRITICAL_ACCESS_NUM[(i // base_num) % N_TXN_TYPE])
                #      .set_integer_casting().set_mutation(sigma=1, exponent=None)
                #      for i in range(base_num * N_TXN_TYPE * N_TXN_TYPE)]
                # )
                # wait_parameters.extend(
                #     [ng.p.Scalar(lower=0, upper=TXN_ACCESS_NUM[i // base_num])
                #      .set_integer_casting().set_mutation(sigma=1, exponent=None)
                #      for i in range(base_num * N_TXN_TYPE)]
                #     # [ng.p.Choice(CRITICAL_GUARD_POINTS[i // base_num]) for i in range(base_num * N_TXN_TYPE)]
                # )
                # check_length += base_num * N_TXN_TYPE * N_TXN_TYPE
                # check_length += base_num * N_TXN_TYPE
            elif base_num == 0:
                # the whole graph.
                wait_parameters.extend(
                    [ng.p.Scalar(lower=0,
                                 upper=TXN_ACCESS_NUM[(i // N_CRITICAL_WAIT) % N_TXN_TYPE])
                     .set_integer_casting().set_mutation(sigma=2, exponent=None)
                     for i in range(N_CRITICAL_WAIT * N_TXN_TYPE)]
                )

        # Access policy parameters (adaptive field)
        access_parameters = []
        if self.setting["access"]:
            access_parameters.extend([ng.p.Choice([0, 1], repetitions=self.max_state)])
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
            timeout_parameters.extend([ng.p.Scalar(lower=1, upper=32, init=32).set_integer_casting()])
            # to avoid np x ** y overflow, we start from 1.
            timeout_parameters.extend([ng.p.Scalar(lower=0, upper=8, init=2).set_mutation(exponent=None)
                                       for _ in range(2 * 3 * N_TXN_TYPE)])
            # timeout
            timeout_parameters.extend([ng.p.Scalar(lower=100,
                                                   upper=1000000, init=100000) for _ in range(self.max_state)])
            check_length += self.max_state

        # Combine the policies into the Instrumentation
        self.bounds = ng.p.Instrumentation(
            expose=ng.p.Tuple(*expose_parameters),
            wait=ng.p.Tuple(*wait_parameters),
            access=ng.p.Tuple(*access_parameters),
            rank=ng.p.Tuple(*rank_parameters),
            timeout=ng.p.Tuple(*timeout_parameters)
        )

        self.check_encoder_length = check_length

    def print_population(self):
        print("current population = ", [p.score for p in self.best_population])

    def __init__(self, base_command, name, log_dir, starting_points, max_state, seed, log_rate=1, _runtime=1,
                 setting=None):
        if setting is None:
            setting = {"expose": False,
                       "wait": False,
                       "wait_guard": -1,  # 0 for polyjuice like, > 1 for layered, -1 for calculated
                       "rank": False,
                       "access": False,
                       "timeout": False}
        self.max_state = max_state
        self.seed = seed
        self.current_iter = 0
        self.branching_factor = 1
        # # we first greedily cut the rendezvous
        # self.rendezvous_cap = TXN_ACCESS_NUM
        # self.num_full_access = np.sum(TXN_ACCESS_NUM)
        # self.num_learn_access = np.sum(TXN_ACCESS_NUM)
        #
        self.searched_points_hash = {}
        self.setting = setting
        self.best_seen_performance = 0
        self.log_rate = log_rate
        self.pop_size = 5
        self.mutate_rate = 0.1
        self.graphic_reduction = True
        self.best_population = []
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

    def tell(self, policy):
        self.best_population.append(policy)
        self.best_population.sort(key=lambda p: p.score, reverse=True)
        if len(self.best_population) > self.pop_size:
            self.best_population = self.best_population[:self.pop_size]

    def ask(self):
        res = []
        i = 0
        # Type 1: merge the two best points to speedup graph reduction.
        if self.graphic_reduction and len(self.best_population) > 1:
            policy = self.best_population[0]
            tmp = policy.merge(policy)
            if self.searched_points_hash.get(tmp.hash(), False) is False:
                self.searched_points_hash[tmp.hash()] = True
                res.append(tmp)

        # Type 2: mutation.
        while i < len(self.best_population) and len(self.best_population) > 0:
            policy = self.best_population[i]
            for _ in range(self.branching_factor):
                # during graph reduction, since we are shrinking on single direction, the policy may quickly get stuck.
                tmp = policy.try_mutate()
                if tmp is not None:
                    self.searched_points_hash[tmp.hash()] = True
                    res.append(tmp)
                else:
                    # we remove the point from population since we could have touched the optima.
                    self.best_population = self.best_population[:i] + self.best_population[i + 1:]
                    i -= 1
                    break
            i += 1


        return res

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

    def sampling(self):
        # greedy method: cutting the tail to tight rendezvous
        samples = []
        if not self.setting["expose"] and not self.setting["access"]:
            # skill sampling for non-chop learning, use nevergrad sampler.
            return samples

        for i in [3, 2, 1, 0]:
            r = TXN_ACCESS_START[i + 1]
            for l in reversed(range(TXN_ACCESS_START[i], TXN_ACCESS_START[i + 1])):
                p_t = self.best_policy.cutting_rendezvous(l, r)
                p_t.score = self.evaluate_policy(p_t)
                samples.append(p_t)
                # print("Sampled scores = ", p_t.score)
                if p_t.score < self.best_seen_performance * 0.7:
                    break
        print("Greedy sampling finish: scores = ", [p.score for p in samples])
        return samples


class Policy(object):
    def __init__(self, _access=None, _rank=None, _timeout=None,
                 _expose=None, _wait_chop=None, _extra=None, _from=None,
                 load_file=None, encoded=None):
        self.score = -1
        self.learner = _from
        self.mutate_factor = _from.mutate_rate
        self.max_state = self.learner.max_state
        # the default txn_buf_size and backoff parameters.
        self.extra_policies = [32] + [2 for _ in range(6 * N_TXN_TYPE)]
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
            self.extra_policies = _extra
            # stored procedure related policies.
            self.expose = np.array(_expose)
            self.wait_chop = np.array(_wait_chop)
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
        f_out.write("expose:\n")
        f_out.writelines([str(int(value)) for value in self.expose])
        f_out.write("\n")
        f_out.write("chop wait:\n")
        # wait chop correction.
        self.wait_chop = calculate_wait_access(self.expose, self.access)
        f_out.writelines([str(int(self.wait_chop[i])) + ' ' for i in range(len(self.wait_chop))])
        f_out.write("\n")
        f_out.write("extra:\n")
        f_out.writelines([str(int(value)) + ' ' for value in self.extra_policies])
        f_out.write("\n\n\n")
        f_out.write("learner encoding = \n{encoded_params}\n".format(encoded_params=self.encode()))

    def read_from_file(self, file):
        n = self.learner.max_state
        res = [0.0 for _ in range(3 * n)]
        cur_line = file.readline()
        if "conflict detection" in cur_line:
            values = file.readline().strip()
            self.access = np.array([float(value) for value in values])
            assert len(values) == n
        cur_line = file.readline()
        if "wait priorities" in cur_line:
            values = file.readline().strip().split()
            self.rank = np.array([float(value) for value in values])
            assert len(values) == n
        cur_line = file.readline()
        if "timeout" in cur_line:
            values = file.readline().strip().split()
            self.timeout_policy = np.array([float(value) for value in values])
            assert len(values) == n
        cur_line = file.readline()
        if "expose" in cur_line:
            values = file.readline().strip()
            self.expose = np.array([float(value) for value in values])
            assert len(values) == n
        cur_line = file.readline()
        if "chop wait" in cur_line:
            values = file.readline().strip().split()
            # print(len(values))
            assert len(values) == n * N_TXN_TYPE
            self.wait_chop = np.array([int(values[i]) for i in range(len(values))])
        cur_line = file.readline()
        if "extra" in cur_line:
            values = file.readline().strip().split()
            assert len(values) == 6 * N_TXN_TYPE + 1
            self.extra_policies = np.array([int(val) for val in values])
        return res

    def save_to_path(self, path):
        with open(path, 'w') as file:
            self.write_to_file(file)

    def encode(self):
        """Encode learner's internal state into the parameter dictionary"""
        if self.learner.best_policy is None:
            self.learner.best_policy = self

        wait, expose = chop_domain_filter_encode(self.max_state, self.wait_chop, self.expose)

        expose_params = ()
        wait_params = ()
        access_params = ()
        rank_params = ()
        timeout_params = ()

        # Encode the 'expose' policy parameters
        if self.learner.setting["expose"]:
            expose_params = (tuple(expose),)

        # Encode the 'wait' policy parameters
        if self.learner.setting["wait"]:
            base_num = self.learner.setting["wait_guard"]
            if base_num > 0:
                wait_segments, wait_guards = translate_wait_to_guard_points(wait, base_num)

                wait_segment_params = []
                wait_guard_params = []

                # Encode wait segments
                for i in range(N_TXN_TYPE):
                    for j in range(N_TXN_TYPE):
                        sorted_wait_segment = sorted(wait_segments[i, j])  # Sort wait segments
                        wait_segment_params.extend(sorted_wait_segment)

                # Encode wait guards
                for i in range(N_TXN_TYPE):
                    sorted_wait_guards = sorted(wait_guards[i])  # Sort wait guards
                    wait_guard_params.extend(sorted_wait_guards)

                wait_params = tuple(np.append(wait_segment_params, wait_guard_params))
            elif base_num == 0:
                wait_params = tuple(np.array(wait))
            else:
                wait_params = tuple()

        # Encode the 'access' policy parameters
        if self.learner.setting["access"]:
            access_params = (tuple(self.access),)
            # print(access_params)

        # Encode the 'rank' policy parameters
        if self.learner.setting["rank"]:
            rank_params = tuple(self.rank)

        # Encode the 'timeout' policy parameters
        if self.learner.setting["timeout"]:
            timeout_params = tuple(np.concatenate((self.extra_policies, self.timeout_policy)))

        return {
            "expose": expose_params,
            "wait": wait_params,
            "access": access_params,
            "rank": rank_params,
            "timeout": timeout_params
        }

    def decode(self, param_dict):
        """Decode the parameter dictionary into the learner's internal state."""
        # print(param_dict)

        # Handle 'expose' policy
        if self.learner.setting["expose"]:
            expose_values = param_dict.get("expose", None)[0]
            assert expose_values is not None, "Expected expose_values to be provided, but got None."
            self.expose = np.array(chop_domain_filter_decode(self.max_state, None, np.array(expose_values))[1],
                                   dtype=int)
        else:
            self.expose = self.learner.best_policy.expose

        # Handle 'access' policy
        if self.learner.setting["access"]:
            access_values = param_dict.get("access", None)[0]
            assert access_values is not None, "Expected access_values to be provided, but got None."
            self.access = np.array(access_values, dtype=int)
        else:
            self.access = self.learner.best_policy.access

        # Handle 'wait' policy
        if self.learner.setting["wait"]:
            wait_values = param_dict.get("wait", None)
            base_num = self.learner.setting["wait_guard"]
            if base_num > 0:
                assert wait_values is not None, "Expected wait_values to be provided, but got None."
                lm = base_num * N_TXN_TYPE * N_TXN_TYPE
                lm2 = lm + base_num * N_TXN_TYPE
                guard_points = np.array(wait_values[lm:lm2]).reshape((N_TXN_TYPE, base_num))
                wait_segments = np.array(wait_values[:lm]).reshape((N_TXN_TYPE, N_TXN_TYPE, base_num))
                for i in range(N_TXN_TYPE):
                    for j in range(N_TXN_TYPE):
                        np.sort(wait_segments[i, j])
                for i in range(N_TXN_TYPE):
                    np.sort(guard_points[i])
                filtered_wait = reverse_translate_wait_to_guard_points(wait_segments, guard_points, n_guard=base_num)
                self.wait_chop, _ = chop_domain_filter_decode(self.max_state, np.array(filtered_wait), None)
            elif base_num == 0:
                assert wait_values is not None, "Expected wait_values to be provided, but got None."
                filtered_wait = wait_values
                self.wait_chop, _ = chop_domain_filter_decode(self.max_state, np.array(filtered_wait), None)
            else:
                self.wait_chop = calculate_wait_access(self.expose, self.access)
        else:
            self.wait_chop = self.learner.best_policy.wait_chop

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
            self.extra_policies = np.array(timeout_values[:-self.max_state])
        else:
            self.timeout_policy = self.learner.best_policy.timeout_policy
            self.extra_policies = self.learner.best_policy.extra_policies

        assert len(self.extra_policies) > 0

    def hash(self):
        return hash(json.dumps(self.encode(), sort_keys=True, default=convert_np))

    def cutting_rendezvous(self, l, r):
        res = Policy(_access=self.access, _rank=self.rank, _timeout=self.timeout_policy,
                     _expose=self.expose, _wait_chop=self.wait_chop, _from=self.learner,
                     _extra=self.extra_policies, load_file=None, encoded=None)
        if self.learner.setting["expose"]:
            res.expose[l:r] = 0
        if self.learner.setting["access"]:
            res.access[l:r] = 0
        return res

    def merge(self, parent):
        # This is only used during wait chop learning.
        res = Policy(_access=self.access, _rank=self.rank, _timeout=self.timeout_policy,
                     _expose=self.expose, _wait_chop=self.wait_chop, _from=self.learner, _extra=self.extra_policies)
        assert self.learner.graphic_reduction  # only used to speedup graphic reduction.
        if self.learner.setting["expose"]:
            res.expose[parent.expose == 0] = 0

        if self.learner.setting["access"]:
            res.expose[parent.access == 0] = 0

        assert not self.learner.setting["timeout"]
        assert not self.learner.setting["rank"]
        return res

    def try_mutate(self):
        tmp = self.mutate_once()
        n_try = 0
        while self.learner.searched_points_hash.get(tmp.hash(), False) is not False \
                and n_try < max_try_from_a_chop:
            tmp = self.mutate_once()
            n_try += 1
        if n_try == max_try_from_a_chop:
            return None
        else:
            return tmp

    def mutate_once(self):
        # This is only used during wait chop learning.
        res = Policy(_access=self.access, _rank=self.rank, _timeout=self.timeout_policy,
                     _expose=self.expose, _wait_chop=self.wait_chop, _from=self.learner, _extra=self.extra_policies)
        if self.learner.graphic_reduction:
            if self.learner.setting["expose"]:
                # Type 1: we can mutate by merging adjacent pieces.
                mutation_mask = np.random.rand(len(self.expose)) < self.mutate_factor
                res.expose[mutation_mask] = 0

            # Handle 'access' policy, mark some access as "low conflict probability"
            if self.learner.setting["access"]:
                # Type 2: we can mutate by tagging some access as "low conflict probability".
                mutation_mask = np.random.rand(len(self.access)) < self.mutate_factor
                res.access[mutation_mask] = 0
        else:
            # To avoid stuck on local optima, we allow the addition of edges.
            if self.learner.setting["expose"]:
                mut = (np.random.rand(len(self.expose)) < self.mutate_factor).astype(int)
                res.expose = res.expose.astype(int) ^ mut

            if self.learner.setting["access"]:
                mut = (np.random.rand(len(self.access)) < self.mutate_factor).astype(int)
                res.access = res.access.astype(int) ^ mut

        res.expose[res.access == 0] = 0
        # Heuristic, if one operation does not conflict, it would always be better to expose at the last operation?

        assert not self.learner.setting["timeout"]
        assert not self.learner.setting["rank"]
        return res


def convert_np(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
