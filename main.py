import numpy as np
import torch
from torch.distributions import Normal, Uniform, Bernoulli, Laplace, Binomial
import torch.nn as nn
import torch.optim as optim
from math import sqrt

from itertools import chain
from flow import create_cnf, compute_loss, create_batch, get_transforms

from nets import AdaptedODENet, SparseODENet, ODENet

import models

from types import SimpleNamespace

from sacred import Experiment, SETTINGS
from sacred.observers import MongoObserver

import random

import os

from sacred import Experiment
from sacred.observers import TinyDbObserver
from sacred.observers import FileStorageObserver
ex = Experiment()

def seed_all(seed):
    """Seed all devices deterministically off of seed and somewhat
    independently."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@ex.config
def my_config():
    lr = 1e-2
    weight_decay = 1e-5
    to_augment = []
    train_steps = 10
    batch_size = 100
    gmodel_name = 'gaussian_bn'
    flow_connectivity = "fully_connected"
    loss_choice = "forward"
    repeat_samples = 1
    integration_times = [0.0, 1.0]

    device = "cpu"


@ex.capture
def log_scalar(name, scalar, step=None, _run=None):
    if isinstance(scalar, torch.Tensor):
        scalar = scalar.item()

    if step is not None:
        print("Step: {} - {}: {}".format(step, name, scalar))
        _run.log_scalar(name, scalar, step)
    else:
        print("{}: {}".format(name, scalar))
        _run.log_scalar(name, scalar)


def init(seed, config, _run):
    seed_all(seed)
    args = SimpleNamespace(**config)
    args._run = _run

    args.device = torch.device("cpu") if args.device == "cpu" else getFreeGPU()
    print("Using device:", args.device)


    args.integration_times = torch.tensor(args.integration_times).to(args.device)


    if args.gmodel_name.startswith("autogen"):
        autogen_models = __import__(args.gmodel_name, fromlist="*")
        args.gmodel = autogen_models.FlowModel()
    elif args.gmodel_name == "gaussian_bn":
        args.gmodel = models.GaussianBayesianNetwork()
    elif args.gmodel_name == "crazy1d":
        args.gmodel = models.Crazy1D()
    elif args.gmodel_name == "circle":
        args.gmodel = models.CircleModel()
    elif args.gmodel_name == "state_space":
        args.gmodel = models.StateSpaceModel()
    elif args.gmodel_name == "state_space_larger":
        args.gmodel = models.StateSpaceModelLarger()
    elif args.gmodel_name == "8gaussians":
        args.gmodel = models.ToyData("8gaussians")
    elif args.gmodel_name == "rings":
        args.gmodel = models.ToyData("rings")
    elif args.gmodel_name == "swissroll":
        args.gmodel = models.ToyData("swissroll")
    elif args.gmodel_name == "bigger_graph1":
        args.gmodel = models.BiggerGraph1()
    elif args.gmodel_name == "bigger_graph2":
        args.gmodel = models.BiggerGraph2()
    elif args.gmodel_name == "bigger_graph3":
        args.gmodel = models.BiggerGraph3()
    elif args.gmodel_name == "bigger_graph4":
        args.gmodel = models.BiggerGraph4()
    elif args.gmodel_name == "bigger_graph5":
        args.gmodel = models.BiggerGraph5()
    elif args.gmodel_name == "bigger_graph6":
        args.gmodel = models.BiggerGraph6()
    elif args.gmodel_name == "simple_arith_circuit":
        args.gmodel = models.SimpleArithmeticCircuit()
    elif args.gmodel_name == "ber_gmm":
        args.gmodel = models.BernoulliGMM()

    else:
        raise Exception("Model unknown: {}".format(args.gmodel_name))

    if len(args.to_augment) > 0:
        print("Augmenting rvars: ", args.to_augment)
        args.gmodel = models.AugmentedModel(args.gmodel, args.to_augment)
        #print("Faithful inversion: ", args.gmodel.faithful_adjacency)

    args.dim_latent = args.gmodel.dim_latent
    args.dim_condition = args.gmodel.dim_condition

    min_std = 1e-5

    xs, ys = create_batch(args.gmodel.sample, 10000)
    xstd = torch.std(xs, dim=0).to(args.device)
    xscale = torch.max(xstd, torch.ones_like(xstd)*min_std)
    args.xscale = xscale
    args.xshift = torch.mean(xs, dim=0).to(args.device)
    ystd = torch.std(ys, dim=0).to(args.device)
    yscale = torch.max(ystd, torch.ones_like(ystd)*min_std)
    args.yscale = yscale
    args.yshift = torch.mean(ys, dim=0).to(args.device)


    if args.flow_connectivity == "fully_connected":
        args.diffeq = AdaptedODENet(args.dim_latent, args.dim_condition)

    elif args.flow_connectivity == "faithful":
        args.diffeq = SparseODENet(args.dim_latent, args.dim_condition, args.gmodel.faithful_adjacency, args.device)
    elif args.flow_connectivity == "faithful_large":
        args.diffeq = SparseODENet(args.dim_latent, args.dim_condition, args.gmodel.faithful_adjacency, args.device, num_layers=8)
    elif args.flow_connectivity == "faithful_small":
        args.diffeq = SparseODENet(args.dim_latent, args.dim_condition, args.gmodel.faithful_adjacency, args.device, num_layers=2)

    elif args.flow_connectivity == "random_sparse":
        args.diffeq = SparseODENet(args.dim_latent, args.dim_condition, args.gmodel.rand_adjacency, args.device)
    elif args.flow_connectivity == "ffjord_baseline":
        # default arguments from train_misc.py and train_toy.py
        args.diffeq = ODENet(hidden_dims=(64, 64, 64),
                             input_shape=(args.dim_latent,), strides=None, conv=False,
                             conditional_dims=args.dim_condition,
                             layer_type="concatsquash", nonlinearity="tanh")
    else:
        raise Exception("Connectivity type unknown: {}".format(args.flow_connectivity))


    args.diffeq.to(args.device)
    args.diffeq.conditioned = ys[0:args.batch_size]

    args.cnf = create_cnf(
        args.diffeq, regularization_fns=None).to(args.device)

    return args

@ex.automain
def experiment(_seed, _config, _run):
    args = init(_seed, _config, _run)

    cnf = args.cnf

    params = chain(cnf.parameters())
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    cnf.train()

    moving_sym_kl = None
    for i in range(args.train_steps):
        if i == int(0.5*args.train_steps):
            for g in optimizer.param_groups:
                g['lr'] = g['lr']*0.1
                log_scalar("learning_rate", g['lr'], i)

        if i == int(0.8*args.train_steps):
            for g in optimizer.param_groups:
                g['lr'] = g['lr']*0.1
                log_scalar("learning_rate", g['lr'], i)

        if i == int(0.95*args.train_steps):
            for g in optimizer.param_groups:
                g['lr'] = g['lr']*0.5
                log_scalar("learning_rate", g['lr'], i)


        optimizer.zero_grad()

        res = compute_loss(args, log_scalar)
        if args.loss_choice == "forward":
            res.forward_kl.backward()
        if args.loss_choice == "backward":
            res.reverse_kl.backward()
        if args.loss_choice == "sym":
            #rescale_reverse = forward_kl.item()/reverse_kl.item()
            (res.forward_kl + res.reverse_kl).backward()
        if args.loss_choice == "sym_reg":
            # Only reverse KL makes sense for angles because logp is only known in this case
            (res.forward_kl + res.reverse_kl + (res.reverse_reg[0][1])).backward()
        if args.loss_choice == "forw_reg":
            # Only reverse KL makes sense for angles because logp is only known in this case
            (res.forward_kl + res.reverse_reg[0][1]).backward()
        if args.loss_choice == "reg_only":
            forward_vel = res.forward_reg[1][1]
            forward_angle = res.forward_reg[0][1]
            reverse_vel = res.reverse_reg[1][1]
            reverse_angle = res.reverse_reg[0][1]
            (reverse_angle).backward()

        log_scalar("forward_kl", res.forward_kl.item())
        log_scalar("reverse_kl", res.reverse_kl.item())
        log_scalar("backprop_solver_evals", cnf.num_evals())
        log_scalar("forward_solver_evals", res.forward_num_evals)
        log_scalar("reverse_solver_evals", res.reverse_num_evals)
        if moving_sym_kl is None:
            moving_sym_kl = 0.5*(res.forward_kl.item() + res.reverse_kl.item())
        else:
            moving_sym_kl = max(0.95*moving_sym_kl + 0.05*(0.5*(res.forward_kl.item() + res.reverse_kl.item())), 1e-5)
        log_scalar("moving_sym_kl", moving_sym_kl)

        if i % 500 == 0:
            torch.save(cnf.state_dict(), f"./{i}_flow.th")
            ex.add_artifact(f"./{i}_flow.th")
        #log_scalar("reg_states", cnf.get_regularization_states())

        optimizer.step()

    filename = "./final_flow.th"
    torch.save(cnf.state_dict(), filename)
    ex.add_artifact(filename)

    return res.forward_kl.item(), res.reverse_kl.item()

