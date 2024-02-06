from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import pandas as pd
import numpy as np 
seed=8
torch.manual_seed(seed)
use_cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda' if use_cuda else 'cpu')
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


def get_mnist(N = 100):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
        
    mnist = datasets.MNIST('./', train=True, download=True, transform = transform)

    x, y = [], []
    for i, (x_i,y_i) in enumerate(mnist):
        if i == N:
            break
        x.append(x_i)
        y.append(y_i)
    x = torch.stack(x, dim=0)
    y = torch.tensor(y)
    return x, y

def get_risks(y, n_groups = 4, seed = 0):

    y_np = y.numpy()
    rnd = np.random.RandomState(seed)

    # assign class labels to one of `n_groups` risk groups
    classes = np.unique(y_np)
    group_assignment = {}
    group_members = {}
    groups = rnd.randint(n_groups, size=classes.shape)
    for label, group in zip(classes, groups):
        group_assignment[label] = group
        group_members.setdefault(group, []).append(label)

    # assign risk score to each class label (to each group + randomness within each group for each label)
    risk_per_class = {}
    for label in classes:
        group_idx = group_assignment[label]
        group = group_members[group_idx]
        label_idx = group.index(label)
        group_size = len(group)

        # allow risk scores in each group to vary slightly
        group_score = 3.0 * group_idx + 2.0
        class_score = group_score - (label_idx - (group_size // 2)) / 4.0
        risk_per_class[label] = class_score
        print("Label:{}, Group:{}, Group Risk:{}, Label Risk:{}".format(label,group_idx,group_score,class_score))

    risk = torch.tensor([risk_per_class[y_i] for y_i in y_np])
    assert torch.all(risk > 0)
    return risk

def make_gamma_dist(risk):
    t_mean = risk * 10.0
    var = 0.01
    # alpha is shape
    # beta is rate, which is 1 / scale 
    alpha = t_mean.pow(2) / var
    beta = t_mean / var
    return torch.distributions.Gamma(alpha, beta)

def make_exp_dist(risk):
    return Exponential(rate = risk)

def get_censoring_times(t, prob_censored_approx = .10):
    t_np = t.numpy()
    # generate time of censoring
    # by computing the qt = (1-prob_censored)^th quantile and
    # uniformly sampling between the smallest t in the data and qt
    qt = np.quantile(t_np, 1.0 - prob_censored_approx)
    c = torch.distributions.Uniform(t.min(), qt).sample(sample_shape=(t.shape[0],))
    return c

if __name__ == '__main__':

    N = 100
    x, y = get_mnist(N = N)
    risk = get_risks(y, n_groups = 4, seed = 0)
    t = make_gamma_dist(risk).sample()
    c = get_censoring_times(t, prob_censored_approx = 0.10)
    print("t min mean max", t.min(), t.mean(), t.max())
    print("c min mean max", c.min(), c.mean(), c.max())
    print("proportion censored", (c<t).sum().float() / N)

