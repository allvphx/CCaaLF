import time
from cc_optimizer import Policy, CCLearner
import nevergrad as ng

TIME_LIMIT = 3 * 60 * 60

MAX_WINDOW = 1.0
MIN_WINDOW = 0.0
MAX_NO_UPDATE_COUNT_LIMIT = 100
MIN_NO_UPDATE_COUNT_LIMIT = 10
MAX_KAPPA = 2.5
MIN_KAPPA = 0

# If we did not find better solution in one round, a better solution tends to be found with more trials.
no_update_count_cap = MIN_NO_UPDATE_COUNT_LIMIT

# For exploration, we use larger kappa and window.
kappa = MAX_KAPPA
window = MAX_WINDOW


def learning_round(learner, rand=23):
    def evaluate(*args, **kwargs):
        params = list(args)
        policy = Policy(encoded=params, _from=learner)
        return -learner.evaluate_policy(policy)

    optimizer = ng.optimizers.TBPSA(parametrization=learner.bounds[learner.training_stage], budget=100)
    if learner.starting_points is not None:
        for p in learner.starting_points:
            candidate = optimizer.parametrization.spawn_child(new_value=(tuple(p.encode()), {}))
            optimizer.tell(candidate, evaluate(*candidate.args, **candidate.kwargs))

    optimizer.minimize(evaluate)
    return learner.best_seen_performance, time.time() - learner.start_time


def learn(command, fin_log_dir, state_size, start_policy=None, seed=23):
    learner = CCLearner(command, "ng learner", "./training/bo-all/{name}".format(name="BO-DR"),
                        None, state_size, 13)
    global kappa, no_update_count_cap, window
    window = MAX_WINDOW
    kappa = MAX_KAPPA
    no_update_count_cap = MIN_NO_UPDATE_COUNT_LIMIT
    if start_policy is not None:
        learner.start_policy_list = []
        for p_dir in start_policy:
            policy = Policy(load_file=p_dir)
            learner.start_policy_list.append(policy)

    while True:
        # Due to domain reduction, we will shrink to a local optima, thus we run multiple round to reset windows.
        if not learning_round(learner, rand=seed):
            break
        else:
            seed = (seed * 11) % 1000
        print("----------------- starting another round of exploration -------------------")
    training_time = time.time() - learner.start_time
    print("Training time for exploration: {:.2f} seconds".format(training_time))
    return learner.best_seen_performance, fin_log_dir

# ./out-perf.masstree/benchmarks/dbtest --bench tpcc --retry-aborted-transactions --parallel-loading
# --backoff-aborted-transactions --scale-factor 1 --num-threads 16 --runtime 1 --encoder
# ./encoder/default_encoder_tpcc.txt  --policy ./training/aim.txt
