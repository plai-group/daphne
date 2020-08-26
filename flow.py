import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import Normal, Uniform, Bernoulli, Laplace
import torch.nn as nn
import torch.optim as optim
from math import sqrt
from tqdm import tqdm

from toolz import last

import lib.layers as layers
import math

from collections import namedtuple


def set_cnf_options(model):
    tol = 1e-5

    def _set(module):
        if isinstance(module, layers.CNF):
            # Set training settings
            module.solver = "dopri5"
            module.atol = tol # args.atol
            module.rtol = tol # args.rtol


            # Set the test settings
            module.test_solver = "dopri5"
            module.test_atol = tol
            module.test_rtol = tol

        if isinstance(module, layers.ODEfunc):
            module.rademacher = False
            module.residual = False

    model.apply(_set)


def create_cnf(diffeq, regularization_fns=None):

    # inlined args default values
    solver = "dopri5"
    divergence_function = "approximate" #"brute_force" # TODO?
    residual = False
    rademacher = False
    time_length = 1.0
    train_T = True # TODO

    odefunc = layers.ODEfunc(
        diffeq=diffeq,
        divergence_fn=divergence_function,
        residual=residual,
        rademacher=rademacher,
    )
    cnf = layers.CNF(
        odefunc=odefunc,
        T=time_length,
        train_T=train_T,
        regularization_fns=regularization_fns,
        solver=solver,
    )

    set_cnf_options(cnf)

    return cnf


def get_transforms(model):

    def sample_fn(z, logpz=None):
        if logpz is not None:
            return model(z, logpz, reverse=True)
        else:
            return model(z, reverse=True)

    def density_fn(x, logpx=None):
        if logpx is not None:
            return model(x, logpx, reverse=False)
        else:
            return model(x, reverse=False)

    return sample_fn, density_fn


def compute_loss(args, log_scalar):
    # load data
    x, y = create_batch(args.gmodel.sample, args.batch_size)
    x = x.to(args.device)
    y = y.to(args.device)
    x_ = (x - args.xshift)/args.xscale
    logp_xscale = -torch.sum(torch.log(args.xscale)) # change of variable
    y_ = (y - args.yshift)/args.yscale
    args.diffeq.conditioned[:, :] = y_

    p_ = Normal(0.0, 1.0)  # rescale with marginal statistics
    x0 = p_.sample(x.shape).to(x)
    P_ = p_.log_prob(x0).sum(dim=1) # P underscore

    # reverse KL terms
    print("== reverse pass")
    args.direction = 1.0

    # transform to z
    z, delta_log_q = map(last, args.cnf(x0, P_.unsqueeze(1), 
                                        integration_times=args.integration_times))
    reverse_num_evals = args.cnf.num_evals()

    reverse_reg = args.cnf.get_regularization_states()
    if len(reverse_reg) == 2:
        log_scalar("reverse_angle", reverse_reg[0][1].item())
        log_scalar("reverse_arc_len", reverse_reg[1][1].item())



    PI = args.gmodel.log_prior(z*args.xscale + args.xshift)
    LAMBDA = args.gmodel.log_likelihood(z*args.xscale + args.xshift, y)
    Q = ((delta_log_q.squeeze(1)) + logp_xscale)

    shifted_reverse_kl = Q - (PI + LAMBDA)


    # forward KL terms
    print("== forward pass")
    args.direction = -1.0

    zero = torch.zeros(x.shape[0], 1).to(x)
    z_, delta_log_p_ = map(last, args.cnf(x_, zero, reverse=True))
    forward_num_evals = args.cnf.num_evals()

    forward_reg = args.cnf.get_regularization_states()
    if len(forward_reg) == 2:
        log_scalar("forward_angle", forward_reg[0][1].item())
        log_scalar("forward_arc_len", forward_reg[1][1].item())

    P__ = p_.log_prob(z_).sum(dim=1)
    PI_ = args.gmodel.log_prior(x)
    LAMBDA_ = args.gmodel.log_likelihood(x, y)
    Q_ = P__ - delta_log_p_.squeeze(1) + logp_xscale

    assert PI_.shape == torch.Size([args.batch_size])
    assert LAMBDA_.shape == torch.Size([args.batch_size])

    shifted_forward_kl = (PI_ + LAMBDA_) - Q_


    result = namedtuple("FlowResult",
                        ("forward_kl", "reverse_kl",
                         "forward_reg", "reverse_reg",
                         "forward_num_evals", "reverse_num_evals"))

    result.forward_kl = torch.mean(shifted_forward_kl)
    result.reverse_kl = torch.mean(shifted_reverse_kl)
    result.forward_reg = forward_reg
    result.reverse_reg = reverse_reg
    result.forward_num_evals = forward_num_evals
    result.reverse_num_evals = reverse_num_evals

    return result


def create_batch(sample_fn, minibatch_size=10, repeat_samples=1):
    xs = []
    ys = []
    for _ in range(minibatch_size):
        x, y = sample_fn()
        for i in range(repeat_samples):
            xs.append(x)
            ys.append(y)
    xs = torch.stack(xs)
    ys = torch.stack(ys)

    assert(xs.shape == (minibatch_size*repeat_samples, xs[0].shape[0]))
    assert(ys.shape == (minibatch_size*repeat_samples, ys[0].shape[0]))
    return xs, ys

