from policy import reset, black_box_function, read_from_file
import policy as cp
import time
from bayes_opt import SequentialDomainReductionTransformer
from bayes_opt import BayesianOptimization, UtilityFunction
from torch.utils.tensorboard import SummaryWriter
import numpy as np
# from scipy.optimize import NonlinearConstraint

MAX_NO_UPDATE_COUNT = 20
training_time_limit = 30 * 60
MAX_RESTART_ROUNDS = 10
# def nonlinear_constraint_func(**params):
#     x = np.array([params[key] for key in sorted(params.keys())])
#     constraints = []
#     penalty = 0
#     if cp.training_stage < cp.NUM_OF_ORDER_LEARNING_STAGE:
#         for i in range(cp.MAX_STATE):
#             if i % cp.STEPS != 0 and x[i] > x[i-1]:
#                 penalty += x[i] - x[i-1]
#     else:
#         for i in range(cp.MAX_STATE, 2*cp.MAX_STATE):
#             if i % cp.STEPS != 0 and x[i] > x[i-1]:
#                 penalty += x[i] - x[i-1]
#     return penalty


def learning_round(command, fin_log_dir, writer, rand_s, start_policy=None):
    domain_reduction = SequentialDomainReductionTransformer()

    if cp.training_stage < cp.NUM_OF_ORDER_LEARNING_STAGE:
        var_bounds = {'x{:04d}'.format(i): (0, cp.cap[i + cp.MAX_STATE]) for i in range(cp.MAX_STATE)}
    elif cp.training_stage < cp.NUM_OF_ACCESS_LEARNING_STAGE + cp.NUM_OF_ORDER_LEARNING_STAGE:
        var_bounds = {'x{:04d}'.format(i): (0, cp.cap[i]) for i in range(2 * cp.MAX_STATE)}
    else:
        var_bounds = {'x{:04d}'.format(i): (0, cp.cap[i]) for i in range(4 * cp.MAX_STATE)}

    def bbf(**params):
        x = [params[key] for key in sorted(params.keys())]
        return black_box_function(command=command, writer=writer, fin_log_dir=fin_log_dir, x=x)

    # nonlinear_constraint = NonlinearConstraint(nonlinear_constraint_func, lb=-np.inf, ub=0)
    optimizer = BayesianOptimization(
        f=bbf,
        pbounds=var_bounds,
        random_state=rand_s,
        verbose=0,
        allow_duplicate_points=False,
        bounds_transformer=domain_reduction  # Apply domain reduction
    )

    if start_policy is not None and cp.training_stage == 0:
        for p in start_policy:
            policy = cp.State(p)
            # policy.re_transform()
            # start_policy = policy.policy
            cp.best_policy = start_policy
            if cp.training_stage < cp.NUM_OF_ORDER_LEARNING_STAGE:
                optimizer.probe(params=start_policy[cp.MAX_STATE:2 * cp.MAX_STATE], lazy=True)
            elif cp.training_stage < cp.NUM_OF_ORDER_LEARNING_STAGE + cp.NUM_OF_ACCESS_LEARNING_STAGE:
                optimizer.probe(params=start_policy[:2 * cp.MAX_STATE], lazy=True)
            else:
                optimizer.probe(params=start_policy, lazy=True)

    if cp.training_stage > 0:
        print("loading training dataset, with scores = ", [p.score for p in cp.best_seen_policies])
        for p in cp.best_seen_policies:
            # print("loading point with score = ", score, "policy = ", p)
            if cp.training_stage < cp.NUM_OF_ORDER_LEARNING_STAGE:
                optimizer.register(params=p.policy[cp.MAX_STATE:2 * cp.MAX_STATE], target=p.score)
            elif cp.training_stage < cp.NUM_OF_ORDER_LEARNING_STAGE + cp.NUM_OF_ACCESS_LEARNING_STAGE:
                optimizer.register(params=p.policy[:2 * cp.MAX_STATE], target=p.score)
            else:
                optimizer.register(params=p.policy, target=p.score)

    if cp.training_stage == cp.NUM_OF_ACCESS_LEARNING_STAGE + cp.NUM_OF_ORDER_LEARNING_STAGE:
        occ = [0.0 for _ in range(cp.MAX_STATE)] + [0 for _ in range(cp.MAX_STATE)] + \
              [0 for _ in range(cp.MAX_STATE)] + [0 for _ in range(cp.MAX_STATE)]
        s2pl_w = [2.0 for _ in range(cp.MAX_STATE)] + [10.0 for _ in range(cp.MAX_STATE)] + \
                 [0.0 for _ in range(cp.MAX_STATE)] + [5.0 for _ in range(cp.MAX_STATE)]
        s2pl_nw = [2.0 for _ in range(cp.MAX_STATE)] + [10.0 for _ in range(cp.MAX_STATE)] + \
                  [0.0 for _ in range(cp.MAX_STATE)] + [0 for _ in range(cp.MAX_STATE)]
        all_expose = [1.0 for _ in range(cp.MAX_STATE)] + [10.0 for _ in range(cp.MAX_STATE)] + \
                     [1.0 for _ in range(cp.MAX_STATE)] + [5.0 for _ in range(cp.MAX_STATE)]
        optimizer.probe(params=occ, lazy=True)
        optimizer.probe(params=s2pl_w, lazy=True)
        optimizer.probe(params=s2pl_nw, lazy=True)
        optimizer.probe(params=all_expose, lazy=True)

    optimizer.set_gp_params(alpha=1e-3)
    # as suggested in http://bayesian-optimization.github.io/BayesianOptimization/advanced-tour.html
    # change alpha to accommodate noise introduced by discrete value.

    cp.no_update_count = 0

    kappa = 2.5

    while True:
        elapsed_time = time.time() - cp.start_time
        if elapsed_time > training_time_limit:
            return False
        optimizer.maximize(init_points=0, n_iter=1,
                           acquisition_function=UtilityFunction(kind="ucb", kappa=kappa))
        if cp.no_update_count == MAX_NO_UPDATE_COUNT:
            return True


def learn(command, fin_log_dir, state_size, start_policy=None):
    writer = SummaryWriter(log_dir=fin_log_dir)
    reset(state_size)
    current_start_policy = None
    if start_policy is not None:
        with open(start_policy, 'r') as file:
            current_start_policy = read_from_file(file)

    seed = 13
    n_round = 0
    cp.training_stage = 0
    while n_round < MAX_RESTART_ROUNDS:
        # Due to domain reduction, we will shrink to a local optima, thus we run multiple round to reset windows.
        if not learning_round(command, fin_log_dir, writer, seed, current_start_policy):
            break
        else:
            current_start_policy = cp.best_policy
            seed = (seed * 17) % 1000
        print("----------------- starting another round of exploration -------------------")
        n_round += 1
        cp.training_stage += 1
        if cp.training_stage == cp.NUM_OF_ORDER_LEARNING_STAGE:
            print("switching stage, starting more aggressive exploration (now can learn access).")
        elif cp.training_stage == cp.NUM_OF_ORDER_LEARNING_STAGE + cp.NUM_OF_ACCESS_LEARNING_STAGE:
            print("switching stage, starting more aggressive exploration (now can learn all).")
    training_time = time.time() - cp.start_time
    print("Training time for exploration: {:.2f} seconds".format(training_time))
    writer.close()
    return cp.best_seen, fin_log_dir
