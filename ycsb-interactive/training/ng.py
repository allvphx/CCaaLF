import nevergrad as ng
from cc_optimizer import CCLearner, Policy
import time
import json

TIME_LIMIT = 30 * 60

MAX_WINDOW = 1.0
MIN_WINDOW = 0.0
MAX_NO_UPDATE_COUNT_LIMIT = 100
MIN_NO_UPDATE_COUNT_LIMIT = 10
BATCH_SIZE = 5
DO_GREEDY_SAMPLING = False

# If we did not find better solution in one round, a better solution tends to be found with more trials.
no_update_count_cap = MIN_NO_UPDATE_COUNT_LIMIT

optimizers = [
    None,
    # ng.optimizers.ParametrizedBO(gp_parameters={'alpha': 1e-2}).set_name("BO"),
    # ng.optimizers.DiscreteOnePlusOne,
    # ng.optimizers.RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne,
    # ng.optimizers.TwoPointsDE,
    # ng.optimizers.LhsDE,
    # ng.optimizers.ParametrizedBO(gp_parameters={'alpha': 1e-3}).set_name("BO"),
    # ng.optimizers.DiscreteOnePlusOne,
    # ng.optimizers.PortfolioDiscreteOnePlusOne,
    # ng.optimizers.Shiwa,
    # ng.optimizers.LhsDE,
    # ng.optimizers.RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne.set_name("RPONDOPO"),
    # ng.optimizers.PSO,
    # ng.optimizers.ConfiguredPSO(transform="arctan", popsize=8).set_name("PSO"),
    # ng.optimizers.RandomSearch,
    # ng.optimizers.TBPSA,
]


def benchmark_optimizer(learner, optimizer_class, neval=100):
    def evaluate(*args, **kwargs):
        policy = Policy(encoded=kwargs, _from=learner)
        return -learner.evaluate_policy(policy)

    optimizer = optimizer_class(parametrization=learner.bounds, budget=neval)
    duplicate_policy = {}

    if learner.training_stage > 0:
        for p in learner.evaluated_history:
            if duplicate_policy.get(p.hash(), False) is not False:
                continue
            duplicate_policy[p.hash()] = True
            candidate = optimizer.parametrization.spawn_child(new_value=((), p.encode()))
            optimizer.tell(candidate, p.score)
    else:
        if learner.starting_points is not None:
            for p in learner.starting_points:
                candidate = optimizer.parametrization.spawn_child(new_value=((), p.encode()))
                optimizer.tell(candidate, evaluate(*candidate.args, **candidate.kwargs))
            learner.starting_points = None

    for _ in range(neval):
        candidate = optimizer.ask()
        score = evaluate(*candidate.args, **candidate.kwargs)
        optimizer.tell(candidate, score)
        if learner.no_update_count > learner.patience or time.time() - learner.start_time > TIME_LIMIT:
            # if we did not improve in several steps, we switch to next stage.
            break

    return learner.best_seen_performance, time.time() - learner.start_time


# We are building a training pipeline to decide which feature to train first, and which algorithm we use.
# Some of them are both non-differentiable and non-continuous, invalidating most model-based algorithms!
training_pipeline = [
    # Store_procedure learning pipeline.
    # Exploitation, shrinks to local optimums.
    {"rank": True, "access": True, "timeout": False, "patient": 100,
     "learner": ng.optimizers.ParametrizedBO(gp_parameters={'alpha': 1e-2}).set_name("BO")},
    {"rank": False, "access": False, "timeout": True, "patient": 100,
     "learner": ng.optimizers.ParametrizedBO(gp_parameters={'alpha': 1e-2}).set_name("BO")},
    {"rank": True, "access": True, "timeout": True, "patient": 10000,
     "learner": ng.optimizers.ParametrizedBO(gp_parameters={'alpha': 1e-2}).set_name("BO")},
]


def benchmark(command, fin_log_dir, state_size, start_policy=None, neval=1000):
    results = []
    for optimizer_class in optimizers:
        name = "Mine" if optimizer_class is None else optimizer_class.name
        learner = CCLearner(command, "ng learner", "./training/bo-all/{name}".
                            format(name=name),
                            None, state_size, 13)
        if start_policy is not None:
            learner.load_initial_policy_from_file(start_policy)
        learner.training_stage = 0
        print(f"Running {name}...")
        while learner.training_stage < len(training_pipeline):
            setup = training_pipeline[learner.training_stage]
            learner.patience = setup["patient"]
            learner.update_setting(setup)
            best_score, duration = benchmark_optimizer(learner, optimizer_class, neval)
            results.append({
                "stage": learner.training_stage,
                "optimizer": name,
                "best_score": best_score,
                "duration": duration,
            })
            learner.training_stage += 1
            learner.no_update_count = 0
            print("Starting a new training stage {round}, at iteration {iter} got value {best}!".
                  format(round=learner.training_stage, iter=learner.current_iter, best=learner.best_policy.score))
            print(f"Completed {name} in {duration:.2f} seconds, best score: {best_score:.4f}")
        learner.close()
    with open('%s/data.json' % fin_log_dir, 'w') as json_file:
        json.dump(results, json_file, indent=4)
    return results


def training(command, fin_log_dir, state_size, start_policy=None, neval=1000):
    results = []
    learner = CCLearner(command, "FlexiCC learner", fin_log_dir, None, state_size, 13)
    if start_policy is not None:
        learner.load_initial_policy_from_file(start_policy)
    learner.training_stage = 0
    len_pipe = len(training_pipeline)
    duration = 0
    while duration < TIME_LIMIT and learner.training_stage < len(training_pipeline):
        setup = training_pipeline[learner.training_stage % len_pipe]
        learner.update_setting(setup)
        optimizer_class = setup["learner"]
        best_score, duration = benchmark_optimizer(learner, optimizer_class, neval)
        name = "FlexiCC" if optimizer_class is None else optimizer_class.name
        results.append({
            "stage": learner.training_stage,
            "optimizer": name,
            "best_score": best_score,
            "duration": duration,
        })
        if setup.get("pop_size", None) is not None:
            training_pipeline[learner.training_stage % len_pipe]["pop_size"] *= 2
            training_pipeline[learner.training_stage % len_pipe]["patient"] = \
                10 * training_pipeline[learner.training_stage % len_pipe]["pop_size"]
        learner.training_stage += 1
        learner.no_update_count = 0
        print("Starting a new training stage {round}, at iteration {iter} got value {best}!".
              format(round=learner.training_stage, iter=learner.current_iter, best=learner.best_policy.score))
        print(f"Completed {name} in {duration:.2f} seconds, best score: {best_score:.4f}")
        learner.close()
    return results

