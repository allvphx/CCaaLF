import os.path

from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer, UtilityFunction
from flexi_policy_train import evaluate_encoder
from torch.utils.tensorboard import SummaryWriter
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

writer = SummaryWriter(log_dir="./encoder/")

ENCODER_TX_TYPE = 0
ENCODER_TX_OP_TYPE = 1
ENCODER_TX_N_OP = 2
ENCODER_TX_BLOCKED_ON = 3
ENCODER_TX_BLOCKING = 4
ENCODER_TX_N_BLOCKED = 5
ENCODER_N_FEATURES = 6
eps = 1e-5

log2_values = [0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5]
encoder_feature_cap = [3, 5, 12, 16, 16, 16]
os.makedirs("./encoder/steps", exist_ok=True)
os.makedirs("./encoder/critical", exist_ok=True)


def var_range(encode_type, i):
    if encode_type[i] == 0:  # EncodeIgnore
        return 1
    elif encode_type[i] == 1:  # EncodeIfNot
        return 2
    elif encode_type[i] == 2:  # EncodeLog
        return log2_values[encoder_feature_cap[i]] + 1
    elif encode_type[i] == 3:  # EncodeLinear
        return encoder_feature_cap[i]
    else:
        raise ValueError(f"Unknown encoding type: {encode_type[i]}")


def get_max_state(x):
    max_state = 1
    for i in range(ENCODER_N_FEATURES):
        max_state *= var_range(x, i)
    return max_state


def save_encoder_to_file(x, filename="encoder.txt"):
    with open(filename, "w") as f:
        f.write(" ".join([str(int(x[i])) for i in range(len(x))]) + "\n")


encoder_iter = 0
best_seen_score = 0


def evaluate(**args):
    global encoder_iter, best_seen_score, writer
    # policy value correction.
    x = [int(args[key] * 4 - eps) for key in sorted(args.keys())]
    x[ENCODER_TX_TYPE] = 0 if x[ENCODER_TX_TYPE] <= 2 else 3
    x[ENCODER_TX_OP_TYPE] = 0 if x[ENCODER_TX_OP_TYPE] <= 2 else 3
    for i in range(len(x)):
        x[i] = min(x[i], 3)
        x[i] = max(x[i], 0)

    encoder_iter += 1
    file_name = "./encoder/steps/encoder_%d.txt" % encoder_iter
    save_encoder_to_file(x, file_name)
    score = evaluate_encoder(file_name, get_max_state(x))
    if score[0] > best_seen_score:
        best_seen_score = score[0]
        save_encoder_to_file(x, "./encoder/critical/encoder_%d.txt" % encoder_iter)
    writer.add_scalar('best-seen', best_seen_score, encoder_iter)
    writer.add_scalar('current', score[0], encoder_iter)
    writer.flush()
    return score[0]


txn_access_only = [1, 0, 1, 0, 0, 0]


def main():
    var_bounds = {}
    for i in range(ENCODER_N_FEATURES):  # For A and b
        var_bounds[f'x{i}'] = (0, 1)

    domain_reduction = SequentialDomainReductionTransformer(minimum_window=0.5)

    # Initialize the optimizer
    optimizer = BayesianOptimization(
        f=evaluate,
        pbounds=var_bounds,
        random_state=46,
        verbose=1,
        bounds_transformer=domain_reduction
    )

    if os.path.exists("./encoder/encoder_logs.json"):
        load_logs(optimizer, logs=["./encoder/encoder_logs.json"])
        print("New optimizer is now aware of {} points.".format(len(optimizer.space)))
        logger = JSONLogger(path="./encoder/encoder_logs")
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
        optimizer.set_gp_params(alpha=1e-3)
    else:
        logger = JSONLogger(path="./encoder/encoder_logs")
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
        optimizer.set_gp_params(alpha=1e-3)
        optimizer.probe(params=txn_access_only, lazy=True)

    acquisition_function = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)  # balanced exploration

    # Perform the optimization
    optimizer.maximize(
        init_points=0,  # Number of random initialization points
        n_iter=1000,  # Number of optimization iterations
        acquisition_function=acquisition_function
    )

    # Retrieve the best result
    best_result = optimizer.max
    print("Best result:", best_result)

    writer.close()


if __name__ == '__main__':
    main()
