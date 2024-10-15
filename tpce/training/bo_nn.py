from datetime import datetime
from policy import *
from ibnn import SingleTaskIBNN
import torch
from botorch.acquisition import qExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize
from botorch.sampling.stochastic_samplers import StochasticSampler
from cc_func import CCControl
from torch.utils.tensorboard import SummaryWriter

start_time = time.time()

# For I-BNN model args.
model_args = {
    "acq": "ei",
    "dim_reduction": 20,
    "depth": 3,
    "var_b": 1.6,
    "var_w": 10.0,
    "kernel": "erf"
}


def round_input(x):
    # make value discrete: p_d.
    x[..., 0:MAX_STATE] = torch.floor(x[..., 0:MAX_STATE])
    x[..., 2*MAX_STATE:] = torch.floor(x[..., 2*MAX_STATE:])
    return x


def construct_acq_by_model(model, train_y=None, kappa=2.5):
    if model_args["acq"] == "ei":
        sampler = StochasticSampler(sample_shape=torch.Size([128]))
        return qExpectedImprovement(
            model=model,
            best_f=train_y.max(),
            sampler=sampler
        )
    elif model_args["acq"] == "ucb":
        sampler = StochasticSampler(sample_shape=torch.Size([128]))
        return UpperConfidenceBound(
            model=model,
            beta=kappa ** 2,
            sampler=sampler
        )


def bayes_opt(model, bbf, q, init_x, init_y, model_save_dir, device):
    bounds = bbf.bounds.to(init_x)

    standard_bounds = torch.zeros(2, bbf.dim).to(init_x)
    standard_bounds[1] = 1

    train_x = init_x
    train_y = init_y

    for i in range(500):
        sys.stdout.flush()
        sys.stderr.flush()
        print("\n iteration %d" % i)

        # fit surrogate model on normalized train x
        # todo: add some kind of domain reduction.
        if i % 50 == 0 and i > 0:
            mean_x = train_x.mean(dim=0)
            bounds = torch.stack([mean_x - 0.1, mean_x + 0.1]).clamp(min=0.0, max=1.0)
            print("Updated bounds:", bounds)

        model_start = time.time()
        normalized_x = normalize(train_x, bounds).to(train_x)
        model.fit_and_save(train_x, train_y, model_save_dir)
        model_end = time.time()
        print("fit time", model_end - model_start)

        acq_start = time.time()
        acquisition = construct_acq_by_model(model, train_y=train_y)
        normalized_candidates, acq_values = optimize_acqf(
            acquisition, standard_bounds, q=q, num_restarts=5,
            raw_samples=20, return_best_only=False,
            options={"batch_limit": 5, "maxiter": 200})
        candidates = unnormalize(normalized_candidates.detach(), bounds=bounds)

        # round candidates to correct policy values.
        candidates = round_input(candidates)
        # calculate acquisition values after rounding
        normalized_rounded_candidates = normalize(candidates, bounds)
        acq_values = acquisition(normalized_rounded_candidates)
        acq_end = time.time()
        print("acquisition time", acq_end - acq_start)

        best_index = acq_values.max(dim=0).indices.item()
        # best x is best acquisition value after rounding
        new_x = candidates[best_index].to(train_x)

        del acquisition
        del acq_values
        del normalized_candidates
        del normalized_rounded_candidates
        torch.cuda.empty_cache()

        # evaluate new y values and save
        new_y = bbf(new_x)
        # add explicit output dimension
        new_y = new_y.unsqueeze(-1)
        train_x = torch.cat([train_x, new_x])
        train_y = torch.cat([train_y, new_y])
        print("Max value", train_y.max().item())

    if model_save_dir is not None:
        torch.save(train_x.cpu(), "%s/train_x.pt" % model_save_dir)
        torch.save(train_y.cpu(), "%s/train_y.pt" % model_save_dir)

    max_index = torch.argmax(train_y)
    return train_x[max_index], train_y[max_index]


def initialize_points(n_points, do_func, start_policy=None, device=None):
    x_train = [occ, s2pl_w, all_expose, s2pl_nw]
    # Initial policies and classical algorithms.
    # if start_policy is not None:
    #     with open(start_policy, 'r') as file:
    #         x_train.append(read_from_file(file))
    # x_train.append(s2pl_w)
    # x_train.append(s2pl_nw)
    # x_train.append(all_expose)
    for i in range(n_points):
        x_train.append(np.random.uniform(low=eps, high=cap))

    x_train = torch.tensor(x_train).to(device, dtype=torch.float64)
    x_train = round_input(x_train)
    y_train = do_func(x_train)
    return x_train, y_train.unsqueeze(-1)


def learn_optimized(command, fin_log_dir, start_policy='./training/ic3.txt'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(46)
    writer = SummaryWriter(log_dir=fin_log_dir)

    current_time = datetime.now()
    save_dir = current_time.strftime("experiment_results/%y_%m_%d-%H_%M_%S")
    os.makedirs(save_dir)

    input_dim = len(cap)
    do_func = CCControl(command=command, seed=46, writer=writer, log_dir=fin_log_dir, negate=False)
    init_x, init_y = initialize_points(0, do_func, start_policy=start_policy, device=device)
    print("init x = ", init_x)
    print("init y = ", init_y)
    model_save_dir = save_dir + "_model"
    os.makedirs(model_save_dir)

    st = time.time()
    model = SingleTaskIBNN(model_args, input_dim, 1, device)
    best_x, best_y = bayes_opt(model, do_func, 10, init_x, init_y, model_save_dir, device)
    del model

    training_time = time.time() - st
    print("Training time for exploration: {:.2f} seconds".format(training_time))
    torch.cuda.empty_cache()
    writer.close()
