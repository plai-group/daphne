import torch
import numpy as np
from torch.distributions import Normal, Uniform, Bernoulli, Laplace, Binomial
from lib.toy_data import inf_train_gen
import math
from copy import deepcopy
from flow import create_batch




class GaussianBayesianNetwork:
    def __init__(self):
        self.dim_latent = 7
        self.dim_condition = 8

        self.faithful_adjacency = [[0, 1], [0, 2],
                                   [1, 3], [1, 4], [1, 6], [1, 11], [1, 12],
                                   [2, 1], [2, 5], [2, 6],
                                   [3, 7], [3, 8], [3, 9], [3, 10], [3, 11], [3, 12], [3, 13], [3, 14],
                                   [4, 3], [4, 6], [4, 9], [4, 10], [4, 11], [4, 12],
                                   [5, 1], [5, 6], [5, 11], [5, 12],
                                   [6, 3], [6, 9], [6, 10], [6, 11], [6, 12], [6, 13], [6, 14]] + [[i, i] for i in range(self.dim_latent)]
        self.rand_adjacency = []
        while len(self.rand_adjacency) != len(self.faithful_adjacency):
            self.rand_adjacency = list(set([(i, j)
                                            for i in range(self.dim_latent)
                                            for j in range(self.dim_latent + self.dim_condition) if np.random.rand() > 0.6] + [(i, i) for i in range(self.dim_latent)]))


    def log_prior(self, x):
        PI  = Normal(20, 10).log_prob(x[:, 0])


        PI += Normal(x[:, 0], 5).log_prob(x[:, 1])
        PI += Normal(x[:, 0], 5).log_prob(x[:, 2])


        PI += Normal(x[:, 1], 1).log_prob(x[:, 3])
        PI += Normal(x[:, 1], 1).log_prob(x[:, 4])

        PI += Normal(x[:, 2], 1).log_prob(x[:, 5])
        PI += Normal(x[:, 2], 1).log_prob(x[:, 6])

        return PI


    def log_likelihood(self, x, y):
        LAMBDA  = Normal(x[:, 3], 1).log_prob(y[:, 0])
        LAMBDA += Normal(x[:, 3], 1).log_prob(y[:, 1])

        LAMBDA += Normal(x[:, 4], 1).log_prob(y[:, 2])
        LAMBDA += Normal(x[:, 4], 1).log_prob(y[:, 3])

        LAMBDA += Normal(x[:, 5], 1).log_prob(y[:, 4])
        LAMBDA += Normal(x[:, 5], 1).log_prob(y[:, 5])

        LAMBDA += Normal(x[:, 6], 1).log_prob(y[:, 6])
        LAMBDA += Normal(x[:, 6], 1).log_prob(y[:, 7])

        return LAMBDA


    def sample(self):
        x0 = Normal(20, 10).sample()

        x1 = Normal(x0, 5).sample()
        x2 = Normal(x0, 5).sample()


        x3 = Normal(x1, 1).sample()
        x4 = Normal(x1, 1).sample()

        x5 = Normal(x2, 1).sample()
        x6 = Normal(x2, 1).sample()


        x7 = Normal(x3, 1).sample()
        x8 = Normal(x3, 1).sample()

        x9 = Normal(x4, 1).sample()
        x10 = Normal(x4, 1).sample()

        x11 = Normal(x5, 1).sample()
        x12 = Normal(x5, 1).sample()

        x13 = Normal(x6, 1).sample()
        x14 = Normal(x6, 1).sample()

        xs = torch.tensor([x0, x1, x2, x3, x4, x5, x6])

        return xs, torch.tensor([x7, x8, x9, x10, x11, x12, x13, x14])



class CircleModel:
    def __init__(self):
        self.dim_latent = 2
        self.dim_condition = 1

    def sample(self):
        x0 = Normal(0, 1).sample()
        x1 = Normal(0, 1).sample()

        y = Normal(x0**2 + x1**2, 0.01).sample()
        #y = Normal(1, 0.01).sample()

        return torch.tensor([x0, x1]), torch.tensor([y])

    def log_likelihood(self, x, y):
        LAMBDA  = Normal(x[:, 0]**2 + x[:, 1]**2, 0.01).log_prob(y[:, 0])

        return LAMBDA

    def log_prior(self, x):
        PI  = Normal(0, 1).log_prob(x[:, 0])
        PI += Normal(0, 1).log_prob(x[:, 1])

        return PI


class StateSpaceModel:
    def __init__(self):
        self.dim_latent = 4
        self.dim_condition = 4

        self.faithful_adjacency = [[0,4],[0,5],[0,6],[0,7],
                                   [1,0],[1,5],[1,6],[1,7],
                                   [2,0],[2,1],[2,6],[2,7],
                                   [3,0],[3,1],[3,2],[3,7]] + [[i, i] for i in range(self.dim_latent)]
        self.rand_adjacency = []
        while len(self.rand_adjacency) != len(self.faithful_adjacency):
            self.rand_adjacency = list(set([(i, j)
                                            for i in range(self.dim_latent)
                                            for j in range(self.dim_latent + self.dim_condition)
                                            if np.random.rand() > 0.3] + [(i, i) for i in range(self.dim_latent)]))

    def sample(self):
        x0 = Normal(0, 5).sample()
        x1 = x0 + Normal(0, 0.1).sample()
        x2 = x1 + Normal(0, 0.1).sample()
        x3 = x2 + Normal(0, 0.1).sample()

        y0 = Normal(x0, 0.1).sample()
        y1 = Normal(x1, 0.1).sample()
        y2 = Normal(x2, 0.1).sample()
        y3 = Normal(x3, 0.1).sample()

        return torch.tensor([x0, x1, x2, x3]), torch.tensor([y0, y1, y2, y3])

    def log_prior(self, x):
        PI =  Normal(0, 5).log_prob(x[:, 0])
        PI += Normal(x[:, 0], 0.1).log_prob(x[:, 1])
        PI += Normal(x[:, 1], 0.1).log_prob(x[:, 2])
        PI += Normal(x[:, 2], 0.1).log_prob(x[:, 3])

        return PI

    def log_likelihood(self, x, y):
        LAMBDA =  Normal(x[:, 0], 0.1).log_prob(y[:, 0])
        LAMBDA += Normal(x[:, 1], 0.1).log_prob(y[:, 1])
        LAMBDA += Normal(x[:, 2], 0.1).log_prob(y[:, 2])
        LAMBDA += Normal(x[:, 3], 0.1).log_prob(y[:, 3])

        return LAMBDA


class ToyData:
    def __init__(self, data):
        self.dim_latent = 2
        self.dim_condition = 0
        self.data = data

    def sample(self):
        return torch.tensor(inf_train_gen(self.data, batch_size=1)), torch.tensor([])

    def log_likelihood(self, x, y):
        return torch.zeros(x.shape[0])

    def log_prior(self, x):
        return torch.zeros(x.shape[0])


class StateSpaceModelLarger:
    def __init__(self):
        self.dim_latent = 10
        self.dim_condition = 10

        self.faithful_adjacency = [[0,10],[0,11],[0,12],[0,13],[0,14],[0,15],[0,16],[0,17],[0,18],[0,19],
                                   [1,0],[1,11],[1,12],[1,13],[1,14],[1,15],[1,16],[1,17],[1,18],[1,19],
                                   [2,0],[2,1],[2,12],[2,13],[2,14],[2,15],[2,16],[2,17],[2,18],[2,19],
                                   [3,0],[3,1],[3,13],[3,14],[3,15],[3,16],[3,17],[3,18],[3,19],[3,2],
                                   [4,0],[4,1],[4,14],[4,15],[4,16],[4,17],[4,18],[4,19],[4,2],[4,3],
                                   [5,0],[5,1],[5,15],[5,16],[5,17],[5,18],[5,19],[5,2],[5,3],[5,4],
                                   [6,0],[6,1],[6,16],[6,17],[6,18],[6,19],[6,2],[6,3],[6,4],[6,5],
                                   [7,0],[7,1],[7,17],[7,18],[7,19],[7,2],[7,3],[7,4],[7,5],[7,6],
                                   [8,0],[8,1],[8,18],[8,19],[8,2],[8,3],[8,4],[8,5],[8,6],[8,7],
                                   [9,0],[9,1],[9,19],[9,2],[9,3],[9,4],[9,5],[9,6],[9,7],[9,8]] + [[i, i] for i in range(self.dim_latent)]

        self.rand_adjacency = []
        while len(self.rand_adjacency) != len(self.faithful_adjacency):
            self.rand_adjacency = list(set([(i, j)
                                            for i in range(self.dim_latent)
                                            for j in range(self.dim_latent + self.dim_condition) if np.random.rand() > 0.47] + [(i, i) for i in range(self.dim_latent)]))

    def sample(self):
        x0 = Normal(0, 5).sample()
        x1 = x0 + Normal(0, 0.1).sample()
        x2 = x1 + Normal(0, 0.1).sample()
        x3 = x2 + Normal(0, 0.1).sample()
        x4 = x3 + Normal(0, 0.1).sample()
        x5 = x4 + Normal(0, 0.1).sample()
        x6 = x5 + Normal(0, 0.1).sample()
        x7 = x6 + Normal(0, 0.1).sample()
        x8 = x7 + Normal(0, 0.1).sample()
        x9 = x8 + Normal(0, 0.1).sample()

        y0 = Normal(x0, 0.1).sample()
        y1 = Normal(x1, 0.1).sample()
        y2 = Normal(x2, 0.1).sample()
        y3 = Normal(x3, 0.1).sample()
        y4 = Normal(x4, 0.1).sample()
        y5 = Normal(x5, 0.1).sample()
        y6 = Normal(x6, 0.1).sample()
        y7 = Normal(x7, 0.1).sample()
        y8 = Normal(x8, 0.1).sample()
        y9 = Normal(x9, 0.1).sample()

        return (torch.tensor([x0, x1, x2, x3, x4, x5, x6, x7, x8, x9]),
                torch.tensor([y0, y1, y2, y3, y4, y5, y6, y7, y8, y9]))

    def log_prior(self, x):
        PI =  Normal(0, 5).log_prob(x[:, 0])
        PI += Normal(x[:, 0], 0.1).log_prob(x[:, 1])
        PI += Normal(x[:, 1], 0.1).log_prob(x[:, 2])
        PI += Normal(x[:, 2], 0.1).log_prob(x[:, 3])
        PI += Normal(x[:, 3], 0.1).log_prob(x[:, 4])
        PI += Normal(x[:, 4], 0.1).log_prob(x[:, 5])
        PI += Normal(x[:, 5], 0.1).log_prob(x[:, 6])
        PI += Normal(x[:, 6], 0.1).log_prob(x[:, 7])
        PI += Normal(x[:, 7], 0.1).log_prob(x[:, 8])
        PI += Normal(x[:, 8], 0.1).log_prob(x[:, 9])

        return PI

    def log_likelihood(self, x, y):
        LAMBDA =  Normal(x[:, 0], 0.1).log_prob(y[:, 0])
        LAMBDA += Normal(x[:, 1], 0.1).log_prob(y[:, 1])
        LAMBDA += Normal(x[:, 2], 0.1).log_prob(y[:, 2])
        LAMBDA += Normal(x[:, 3], 0.1).log_prob(y[:, 3])
        LAMBDA += Normal(x[:, 4], 0.1).log_prob(y[:, 4])
        LAMBDA += Normal(x[:, 5], 0.1).log_prob(y[:, 5])
        LAMBDA += Normal(x[:, 6], 0.1).log_prob(y[:, 6])
        LAMBDA += Normal(x[:, 7], 0.1).log_prob(y[:, 7])
        LAMBDA += Normal(x[:, 8], 0.1).log_prob(y[:, 8])
        LAMBDA += Normal(x[:, 9], 0.1).log_prob(y[:, 9])

        return LAMBDA


class Crazy1D:
    def __init__(self):
        self.dim_latent = 1
        self.dim_condition = 1

        self.faithful_adjacency = [[0, 0], [0, 1]]
        self.rand_adjacency = [[0, 0], [0, 1]]


    def sample(self):
        x = Normal(0, math.pi).sample([1])
        y = Normal(torch.sin(x), 0.01).sample()
        return x, y

    def log_prior(self, x):
        PI = Normal(0, 1).log_prob(x[:, 0])
        return PI

    def log_likelihood(self, x, y):
        LAMBDA = Normal(torch.sin(x[:, 0]), 0.01).log_prob(y[:, 0])
        return LAMBDA




class AugmentedModel:
    def __init__(self, model, to_augment):
        self.model = model
        self.num_augment = len(to_augment)
        self.dim_latent = model.dim_latent + len(to_augment)
        self.dim_condition = model.dim_condition

        # add connectivity between vars and new dimensions
        self.faithful_adjacency = []
        # shift conditioning vars
        for [a, b] in model.faithful_adjacency:
            assert a < model.dim_latent
            shifted = [a, b if b < model.dim_latent else b + len(to_augment)]
            self.faithful_adjacency.append(shifted)

        faithful_matrix = torch.zeros([self.dim_latent,
                                       model.dim_latent + model.dim_condition])
        for [a, b] in model.faithful_adjacency:
            faithful_matrix[a, b] = 1.0

        in_offset = model.dim_latent  # + model.dim_condition
        out_offset = model.dim_latent
        # add conditioning from respective var
        augmented_conditioning = []
        # count up on index of conditioning variables
        # TODO too complicated to achieve this.
        cond_offset = in_offset - 1
        for i in to_augment:
            cond_offset = cond_offset + 1
            for [a, b] in self.faithful_adjacency:
                if a == i:
                    augmented_conditioning.append([cond_offset, b])
        self.faithful_adjacency += augmented_conditioning

        # add in edges from augmented to vars
        self.faithful_adjacency += [[a, i + in_offset]
                                    for (i, a) in enumerate(to_augment)]
        # add in edges from vars to augmented
        self.faithful_adjacency += [[i + out_offset, a]
                                    for (i, a) in enumerate(to_augment)]
        # add self-edges between all augmenting variables
        self.faithful_adjacency += [[i + in_offset, j + in_offset]
                                    for i in range(len(to_augment))
                                    for j in range(len(to_augment))
                                    if faithful_matrix[to_augment[i], to_augment[j]] == 1.0]


        self.rand_adjacency = []
        for [a, b] in model.rand_adjacency:
            shifted = [a if a < model.dim_latent else a + len(to_augment),
                       b if b < model.dim_latent else b + len(to_augment)]
            self.rand_adjacency.append(shifted)

        rand_matrix = torch.zeros([model.dim_latent, model.dim_latent + model.dim_condition])
        for [a, b] in model.rand_adjacency:
            rand_matrix[a, b] = 1.0

        self.rand_adjacency += [[a, i + in_offset]
                                for (i, a) in enumerate(to_augment)]
        self.rand_adjacency += [[i + out_offset, a]
                                for (i, a) in enumerate(to_augment)]

        self.rand_adjacency += [[i + in_offset, j + in_offset]
                                for i in range(len(to_augment))
                                for j in range(len(to_augment))
                                if rand_matrix[to_augment[i], to_augment[j]] == 1.0]



    def sample(self):
        x, y = self.model.sample()
        x_aug = torch.cat([x, Normal(0, 1).sample([self.num_augment])], dim=0)
        assert x_aug.shape == torch.Size([self.dim_latent])
        return x_aug, y

    def log_prior(self, x):
        PI = self.model.log_prior(x[:, :self.model.dim_latent])
        PI += Normal(0, 1).log_prob(x[:, self.model.dim_latent:]).sum(dim=1)
        assert PI.shape == torch.Size([x.shape[0]])
        return PI

    def log_likelihood(self, x, y):
        # TODO should be, but doesn't matter in current models
        #return self.model.log_likelihood(x[:, :self.model.dim_latent], y)
        return self.model.log_likelihood(x, y)


class BiggerGraph1:
    def __init__(self):
        self.dim_latent = 17
        self.dim_condition = 5

        self.faithful_adjacency = [[0,1],[0,6],[0,9],
                                   [1,12],[1,2],[1,3],[1,6],[1,20],[1,21],
                                   [10,14],[10,15],[10,20],[10,21],
                                   [11,10],[11,14],[11,15],[11,20],[11,21],
                                   [12,11],[12,15],[12,2],[12,5],[12,8],[12,20],[12,21],
                                   [13,1],[13,12],[13,16],[13,6],[13,21],
                                   [14,15],[14,17],[14,20],[14,21],
                                   [15,17],[15,18],[15,19],[15,20],[15,21],
                                   [16,1],[16,12],[16,6],[16,20],[16,21],
                                   [2,11],[2,15],[2,4],[2,5],[2,8],[2,20],[2,21],
                                   [3,12],[3,2],[3,6],[3,20],[3,21],
                                   [4,11],[4,15],[4,5],[4,7],[4,8],[4,20],[4,21],
                                   [5,11],[5,15],[5,7],[5,8],[5,20],[5,21],
                                   [6,12],[6,2],[6,5],[6,8],[6,20],[6,21],
                                   [7,10],[7,11],[7,15],[7,8],[7,20],[7,21],
                                   [8,10],[8,11],[8,15],[8,20],[8,21],
                                   [9,1],[9,13],[9,6],[9,21]] + [[i, i] for i in range(self.dim_latent)]

        self.rand_adjacency = []
        while len(self.rand_adjacency) != len(self.faithful_adjacency):
            self.rand_adjacency = list(set([(i, j)
                                            for i in range(self.dim_latent)
                                            for j in range(self.dim_latent + self.dim_condition) if np.random.rand() > 0.75] + [(i, i) for i in range(self.dim_latent)]))

    def sample(self):
        x0 = Normal(0, 5).sample()
        x1 = Normal(x0, 0.1).sample()
        x2 = Normal(x1, 0.1).sample()
        x3 = Normal(x1, 0.1).sample()
        x4 = Normal(x2, 0.1).sample()
        x5 = Normal(x2, 0.1).sample()
        x6 = Normal(x3, 0.1).sample()
        x7 = Normal(x4 + x5, 0.1).sample()
        x8 = Normal(x6 + x5, 0.1).sample()
        x9 = Normal(x0 + x6, 0.1).sample()
        x10 = Normal(x7 + x8, 0.1).sample()
        x11 = Normal(x7 + x8 + x5, 0.1).sample()
        x12 = Normal(x6 + x8 + x5, 0.1).sample()
        x13 = Normal(x9 + x6, 0.1).sample()
        x14 = Normal(x11 + x10, 0.1).sample()
        x15 = Normal(x2 + x12 + x11, 0.1).sample()
        x16 = Normal(x13 + x12, 0.1).sample()

        y17 = Normal(x14, 0.1).sample()
        y18 = Normal(x15, 0.1).sample()
        y19 = Normal(x15, 0.1).sample()
        y20 = Normal(x16, 0.1).sample()
        y21 = Normal(x9, 0.1).sample()

        return (torch.tensor([x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16]),
                torch.tensor([y17, y18, y19, y20, y21]))

    def log_prior(self, x):
        PI =  Normal(0, 5).log_prob(x[:, 0])
        PI += Normal(x[:, 0], 0.1).log_prob(x[:, 1])
        PI += Normal(x[:, 1], 0.1).log_prob(x[:, 2])
        PI += Normal(x[:, 1], 0.1).log_prob(x[:, 3])
        PI += Normal(x[:, 2], 0.1).log_prob(x[:, 4])
        PI += Normal(x[:, 2], 0.1).log_prob(x[:, 5])
        PI += Normal(x[:, 3], 0.1).log_prob(x[:, 6])
        PI += Normal(x[:, 4] + x[:, 5], 0.1).log_prob(x[:, 7])
        PI += Normal(x[:, 6] + x[:, 5], 0.1).log_prob(x[:, 8])
        PI += Normal(x[:, 0] + x[:, 6], 0.1).log_prob(x[:, 9])
        PI += Normal(x[:, 7] + x[:, 8], 0.1).log_prob(x[:, 10])
        PI += Normal(x[:, 7] + x[:, 8] + x[:, 5], 0.1).log_prob(x[:, 11])
        PI += Normal(x[:, 6] + x[:, 8] + x[:, 5], 0.1).log_prob(x[:, 12])
        PI += Normal(x[:, 9] + x[:, 6], 0.1).log_prob(x[:, 13])
        PI += Normal(x[:, 11] + x[:, 10], 0.1).log_prob(x[:, 14])
        PI += Normal(x[:, 2] + x[:, 12] + x[:, 11], 0.1).log_prob(x[:, 15])
        PI += Normal(x[:, 13] + x[:, 12], 0.1).log_prob(x[:, 16])

        return PI

    def log_likelihood(self, x, y):
        LAMBDA =  Normal(x[:, 14], 0.1).log_prob(y[:, 0])
        LAMBDA +=  Normal(x[:, 15], 0.1).log_prob(y[:, 1])
        LAMBDA +=  Normal(x[:, 15], 0.1).log_prob(y[:, 2])
        LAMBDA +=  Normal(x[:, 16], 0.1).log_prob(y[:, 3])
        LAMBDA +=  Normal(x[:, 9], 0.1).log_prob(y[:, 4])

        return LAMBDA



import torch.nn
def softplus(x):
    return torch.nn.Softplus()(x)


class BiggerGraph2:
    def __init__(self):
        self.dim_latent = 17
        self.dim_condition = 5

        self.faithful_adjacency = [[0,1],[0,6],[0,9],
                                   [1,12],[1,2],[1,3],[1,6],[1,20],[1,21],
                                   [10,14],[10,15],[10,20],[10,21],
                                   [11,10],[11,14],[11,15],[11,20],[11,21],
                                   [12,11],[12,15],[12,2],[12,5],[12,8],[12,20],[12,21],
                                   [13,1],[13,12],[13,16],[13,6],[13,21],
                                   [14,15],[14,17],[14,20],[14,21],
                                   [15,17],[15,18],[15,19],[15,20],[15,21],
                                   [16,1],[16,12],[16,6],[16,20],[16,21],
                                   [2,11],[2,15],[2,4],[2,5],[2,8],[2,20],[2,21],
                                   [3,12],[3,2],[3,6],[3,20],[3,21],
                                   [4,11],[4,15],[4,5],[4,7],[4,8],[4,20],[4,21],
                                   [5,11],[5,15],[5,7],[5,8],[5,20],[5,21],
                                   [6,12],[6,2],[6,5],[6,8],[6,20],[6,21],
                                   [7,10],[7,11],[7,15],[7,8],[7,20],[7,21],
                                   [8,10],[8,11],[8,15],[8,20],[8,21],
                                   [9,1],[9,13],[9,6],[9,21]] + [[i, i] for i in range(self.dim_latent)]

        self.rand_adjacency = []
        while len(self.rand_adjacency) != len(self.faithful_adjacency):
            self.rand_adjacency = list(set([(i, j)
                                            for i in range(self.dim_latent)
                                            for j in range(self.dim_latent + self.dim_condition) if np.random.rand() > 0.75] + [(i, i) for i in range(self.dim_latent)]))

    def sample(self):
        x0 = Normal(0, 5).sample()
        x1 = Normal(x0, 0.1).sample()
        x2 = Normal(x1, 0.1).sample()
        x3 = Normal(x1, 0.1).sample()
        x4 = Normal(x2, 0.1).sample()
        x5 = Normal(x2, 0.1).sample()
        x6 = Normal(x3, 0.1).sample()
        x7 = Normal(torch.tanh(x4 + x5), 0.1).sample()
        x8 = Normal((x6 + x5)**2, 0.1).sample()
        x9 = Normal(softplus(x0 + x6), 0.1).sample()
        x10 = Normal(x7 * x8 + 2, 0.1).sample()
        x11 = Normal(x7 + (x8 * x5), 0.1).sample()
        x12 = Normal((x6 * x8) - x5, 0.1).sample()
        x13 = Normal(softplus(x9 - 2*x6), 0.1).sample()
        x14 = Normal(x11 * x10, 0.1).sample()
        x15 = Normal(x2 + softplus(x12 * x11**2), 0.1).sample()
        x16 = Normal(x13 - x12, 0.1).sample()

        y17 = Normal(x14, 0.1).sample()
        y18 = Normal(x15, 0.1).sample()
        y19 = Normal(x15, 0.1).sample()
        y20 = Normal(x16, 0.1).sample()
        y21 = Normal(x9, 0.1).sample()

        return (torch.tensor([x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16]),
                torch.tensor([y17, y18, y19, y20, y21]))

    def log_prior(self, x):
        PI =  Normal(0, 5).log_prob(x[:, 0])
        PI += Normal(x[:, 0], 0.1).log_prob(x[:, 1])
        PI += Normal(x[:, 1], 0.1).log_prob(x[:, 2])
        PI += Normal(x[:, 1], 0.1).log_prob(x[:, 3])
        PI += Normal(x[:, 2], 0.1).log_prob(x[:, 4])
        PI += Normal(x[:, 2], 0.1).log_prob(x[:, 5])
        PI += Normal(x[:, 3], 0.1).log_prob(x[:, 6])
        PI += Normal(torch.tanh(x[:, 4] + x[:, 5]), 0.1).log_prob(x[:, 7])
        PI += Normal((x[:, 6] + x[:, 5])**2, 0.1).log_prob(x[:, 8])
        PI += Normal(softplus(x[:, 0] + x[:, 6]), 0.1).log_prob(x[:, 9])
        PI += Normal(x[:, 7] * x[:, 8] + 2, 0.1).log_prob(x[:, 10])
        PI += Normal(x[:, 7] + (x[:, 8] * x[:, 5]), 0.1).log_prob(x[:, 11])
        PI += Normal((x[:, 6] * x[:, 8]) - x[:, 5], 0.1).log_prob(x[:, 12])
        PI += Normal(softplus(x[:, 9] - 2*x[:, 6]), 0.1).log_prob(x[:, 13])
        PI += Normal(x[:, 11] * x[:, 10], 0.1).log_prob(x[:, 14])
        PI += Normal(x[:, 2] + softplus(x[:, 12] * x[:, 11]**2), 0.1).log_prob(x[:, 15])
        PI += Normal(x[:, 13] - x[:, 12], 0.1).log_prob(x[:, 16])

        return PI

    def log_likelihood(self, x, y):
        LAMBDA =   Normal(x[:, 14], 0.1).log_prob(y[:, 0])
        LAMBDA +=  Normal(x[:, 15], 0.1).log_prob(y[:, 1])
        LAMBDA +=  Normal(x[:, 15], 0.1).log_prob(y[:, 2])
        LAMBDA +=  Normal(x[:, 16], 0.1).log_prob(y[:, 3])
        LAMBDA +=  Normal(x[:, 9], 0.1).log_prob(y[:, 4])

        return LAMBDA



class BiggerGraph3:
    def __init__(self):
        self.dim_latent = 17
        self.dim_condition = 5

        self.faithful_adjacency = [[0,1],[0,6],[0,9],
                                   [1,12],[1,2],[1,3],[1,6],[1,20],[1,21],
                                   [10,14],[10,15],[10,20],[10,21],
                                   [11,10],[11,14],[11,15],[11,20],[11,21],
                                   [12,11],[12,15],[12,2],[12,5],[12,8],[12,20],[12,21],
                                   [13,1],[13,12],[13,16],[13,6],[13,21],
                                   [14,15],[14,17],[14,20],[14,21],
                                   [15,17],[15,18],[15,19],[15,20],[15,21],
                                   [16,1],[16,12],[16,6],[16,20],[16,21],
                                   [2,11],[2,15],[2,4],[2,5],[2,8],[2,20],[2,21],
                                   [3,12],[3,2],[3,6],[3,20],[3,21],
                                   [4,11],[4,15],[4,5],[4,7],[4,8],[4,20],[4,21],
                                   [5,11],[5,15],[5,7],[5,8],[5,20],[5,21],
                                   [6,12],[6,2],[6,5],[6,8],[6,20],[6,21],
                                   [7,10],[7,11],[7,15],[7,8],[7,20],[7,21],
                                   [8,10],[8,11],[8,15],[8,20],[8,21],
                                   [9,1],[9,13],[9,6],[9,21]] + [[i, i] for i in range(self.dim_latent)]

        self.rand_adjacency = []
        while len(self.rand_adjacency) != len(self.faithful_adjacency):
            self.rand_adjacency = list(set([(i, j)
                                            for i in range(self.dim_latent)
                                            for j in range(self.dim_latent + self.dim_condition) if np.random.rand() > 0.75] + [(i, i) for i in range(self.dim_latent)]))

    def sample(self):
        x0 = Normal(0, 5).sample()
        x1 = Normal(x0, 0.1).sample()
        x2 = Normal(x1, 0.1).sample()
        x3 = Normal(x1, 0.1).sample()
        x4 = Normal(x2, 0.1).sample()
        x5 = Normal(x2, 0.1).sample()
        x6 = Normal(x3, 0.1).sample()
        x7 = Normal((x4 + x5), 0.1).sample()
        x8 = Normal((x6 + x5), 0.1).sample()
        x9 = Normal((x0 + x6), 0.1).sample()
        x10 = Normal(x7 * x8 + 2, 0.1).sample()
        x11 = Normal(x7 + (x8 * x5), 0.1).sample()
        x12 = Normal((x6 * x8) - x5, 0.1).sample()
        x13 = Normal((x9 - 2*x6), 0.1).sample()
        x14 = Normal(x11 * x10, 0.1).sample()
        x15 = Normal(x2 + (x12 * x11), 0.1).sample()
        x16 = Normal(x13 - x12, 0.1).sample()

        y17 = Normal(x14, 0.1).sample()
        y18 = Normal(x15, 0.1).sample()
        y19 = Normal(x15, 0.1).sample()
        y20 = Normal(x16, 0.1).sample()
        y21 = Normal(x9, 0.1).sample()

        return (torch.tensor([x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16]),
                torch.tensor([y17, y18, y19, y20, y21]))

    def log_prior(self, x):
        PI =  Normal(0, 5).log_prob(x[:, 0])
        PI += Normal(x[:, 0], 0.1).log_prob(x[:, 1])
        PI += Normal(x[:, 1], 0.1).log_prob(x[:, 2])
        PI += Normal(x[:, 1], 0.1).log_prob(x[:, 3])
        PI += Normal(x[:, 2], 0.1).log_prob(x[:, 4])
        PI += Normal(x[:, 2], 0.1).log_prob(x[:, 5])
        PI += Normal(x[:, 3], 0.1).log_prob(x[:, 6])
        PI += Normal(x[:, 4] + x[:, 5], 0.1).log_prob(x[:, 7])
        PI += Normal(x[:, 6] + x[:, 5], 0.1).log_prob(x[:, 8])
        PI += Normal(x[:, 0] + x[:, 6], 0.1).log_prob(x[:, 9])
        PI += Normal(x[:, 7] * x[:, 8] + 2, 0.1).log_prob(x[:, 10])
        PI += Normal(x[:, 7] + (x[:, 8] + x[:, 5]), 0.1).log_prob(x[:, 11])
        PI += Normal((x[:, 6] * x[:, 8]) - x[:, 5], 0.1).log_prob(x[:, 12])
        PI += Normal((x[:, 9] - 2*x[:, 6]), 0.1).log_prob(x[:, 13])
        PI += Normal(x[:, 11] * x[:, 10], 0.1).log_prob(x[:, 14])
        PI += Normal(x[:, 2] + x[:, 12] * x[:, 11], 0.1).log_prob(x[:, 15])
        PI += Normal(x[:, 13] - x[:, 12], 0.1).log_prob(x[:, 16])

        return PI

    def log_likelihood(self, x, y):
        LAMBDA =   Normal(x[:, 14], 0.1).log_prob(y[:, 0])
        LAMBDA +=  Normal(x[:, 15], 0.1).log_prob(y[:, 1])
        LAMBDA +=  Normal(x[:, 15], 0.1).log_prob(y[:, 2])
        LAMBDA +=  Normal(x[:, 16], 0.1).log_prob(y[:, 3])
        LAMBDA +=  Normal(x[:, 9], 0.1).log_prob(y[:, 4])

        return LAMBDA




class BiggerGraph4:
    def __init__(self):
        self.dim_latent = 17
        self.dim_condition = 5

        self.faithful_adjacency = [[0,1],[0,6],[0,9],
                                   [1,12],[1,2],[1,3],[1,6],[1,20],[1,21],
                                   [10,14],[10,15],[10,20],[10,21],
                                   [11,10],[11,14],[11,15],[11,20],[11,21],
                                   [12,11],[12,15],[12,2],[12,5],[12,8],[12,20],[12,21],
                                   [13,1],[13,12],[13,16],[13,6],[13,21],
                                   [14,15],[14,17],[14,20],[14,21],
                                   [15,17],[15,18],[15,19],[15,20],[15,21],
                                   [16,1],[16,12],[16,6],[16,20],[16,21],
                                   [2,11],[2,15],[2,4],[2,5],[2,8],[2,20],[2,21],
                                   [3,12],[3,2],[3,6],[3,20],[3,21],
                                   [4,11],[4,15],[4,5],[4,7],[4,8],[4,20],[4,21],
                                   [5,11],[5,15],[5,7],[5,8],[5,20],[5,21],
                                   [6,12],[6,2],[6,5],[6,8],[6,20],[6,21],
                                   [7,10],[7,11],[7,15],[7,8],[7,20],[7,21],
                                   [8,10],[8,11],[8,15],[8,20],[8,21],
                                   [9,1],[9,13],[9,6],[9,21]] + [[i, i] for i in range(self.dim_latent)]

        self.rand_adjacency = []
        while len(self.rand_adjacency) != len(self.faithful_adjacency):
            self.rand_adjacency = list(set([(i, j)
                                            for i in range(self.dim_latent)
                                            for j in range(self.dim_latent + self.dim_condition) if np.random.rand() > 0.75] + [(i, i) for i in range(self.dim_latent)]))

    def sample(self):
        x0 = Normal(0, 1.0).sample()
        x1 = Normal(x0, 1.0).sample()
        x2 = Normal(x1, 1.0).sample()
        x3 = Normal(x1, 1.0).sample()
        x4 = Normal(x2, 1.0).sample()
        x5 = Normal(x2, 1.0).sample()
        x6 = Normal(x3, 1.0).sample()
        x7 = Normal((x4 * x5), 1.0).sample()
        x8 = Normal((x6 - x5), 1.0).sample()
        x9 = Normal((x0 + x6 * x8), 1.0).sample()
        x10 = Normal(x7 * x8 + 2, 1.0).sample()
        x11 = Normal(x7 + (x8 * x5), 1.0).sample()
        x12 = Normal((x6 * x8) - x5, 1.0).sample()
        x13 = Normal((x9 - 2*x6), 1.0).sample()
        x14 = Normal(x11 + x10, 1.0).sample()
        x15 = Normal(x2 + (x12 * x11), 1.0).sample()
        x16 = Normal(x13 - x12, 1.0).sample()

        y17 = Normal(x14, 0.1).sample()
        y18 = Normal(x15, 0.1).sample()
        y19 = Normal(x15, 0.1).sample()
        y20 = Normal(x16, 0.1).sample()
        y21 = Normal(x9, 0.1).sample()

        return (torch.tensor([x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16]),
                torch.tensor([y17, y18, y19, y20, y21]))

    def log_prior(self, x):
        PI =  Normal(0, 1.0).log_prob(x[:, 0])
        PI += Normal(x[:, 0], 1.0).log_prob(x[:, 1])
        PI += Normal(x[:, 1], 1.0).log_prob(x[:, 2])
        PI += Normal(x[:, 1], 1.0).log_prob(x[:, 3])
        PI += Normal(x[:, 2], 1.0).log_prob(x[:, 4])
        PI += Normal(x[:, 2], 1.0).log_prob(x[:, 5])
        PI += Normal(x[:, 3], 1.0).log_prob(x[:, 6])
        PI += Normal(x[:, 4] * x[:, 5], 1.0).log_prob(x[:, 7])
        PI += Normal(x[:, 6] - x[:, 5], 1.0).log_prob(x[:, 8])
        PI += Normal(x[:, 0] + x[:, 6] * x[:, 8], 1.0).log_prob(x[:, 9])
        PI += Normal(x[:, 7] * x[:, 8] + 2, 1.0).log_prob(x[:, 10])
        PI += Normal(x[:, 7] + (x[:, 8] * x[:, 5]), 1.0).log_prob(x[:, 11])
        PI += Normal((x[:, 6] * x[:, 8]) - x[:, 5], 1.0).log_prob(x[:, 12])
        PI += Normal((x[:, 9] - 2*x[:, 6]), 1.0).log_prob(x[:, 13])
        PI += Normal(x[:, 11] + x[:, 10], 1.0).log_prob(x[:, 14])
        PI += Normal(x[:, 2] + (x[:, 12] * x[:, 11]), 1.0).log_prob(x[:, 15])
        PI += Normal(x[:, 13] - x[:, 12], 1.0).log_prob(x[:, 16])

        return PI

    def log_likelihood(self, x, y):
        LAMBDA =   Normal(x[:, 14], 0.1).log_prob(y[:, 0])
        LAMBDA +=  Normal(x[:, 15], 0.1).log_prob(y[:, 1])
        LAMBDA +=  Normal(x[:, 15], 0.1).log_prob(y[:, 2])
        LAMBDA +=  Normal(x[:, 16], 0.1).log_prob(y[:, 3])
        LAMBDA +=  Normal(x[:, 9], 0.1).log_prob(y[:, 4])

        return LAMBDA




class BiggerGraph5:
    def __init__(self):
        self.dim_latent = 17
        self.dim_condition = 5

        self.faithful_adjacency = [[0,1],[0,6],[0,9],
                                   [1,12],[1,2],[1,3],[1,6],[1,20],[1,21],
                                   [10,14],[10,15],[10,20],[10,21],
                                   [11,10],[11,14],[11,15],[11,20],[11,21],
                                   [12,11],[12,15],[12,2],[12,5],[12,8],[12,20],[12,21],
                                   [13,1],[13,12],[13,16],[13,6],[13,21],
                                   [14,15],[14,17],[14,20],[14,21],
                                   [15,17],[15,18],[15,19],[15,20],[15,21],
                                   [16,1],[16,12],[16,6],[16,20],[16,21],
                                   [2,11],[2,15],[2,4],[2,5],[2,8],[2,20],[2,21],
                                   [3,12],[3,2],[3,6],[3,20],[3,21],
                                   [4,11],[4,15],[4,5],[4,7],[4,8],[4,20],[4,21],
                                   [5,11],[5,15],[5,7],[5,8],[5,20],[5,21],
                                   [6,12],[6,2],[6,5],[6,8],[6,20],[6,21],
                                   [7,10],[7,11],[7,15],[7,8],[7,20],[7,21],
                                   [8,10],[8,11],[8,15],[8,20],[8,21],
                                   [9,1],[9,13],[9,6],[9,21]] + [[i, i] for i in range(self.dim_latent)]

        self.rand_adjacency = []
        while len(self.rand_adjacency) != len(self.faithful_adjacency):
            self.rand_adjacency = list(set([(i, j)
                                            for i in range(self.dim_latent)
                                            for j in range(self.dim_latent + self.dim_condition) if np.random.rand() > 0.75] + [(i, i) for i in range(self.dim_latent)]))

    def sample(self):
        x0 = Normal(0, 5).sample()
        x1 = Normal(x0, 1.0).sample()
        x2 = Normal(x1, 1.0).sample()
        x3 = Normal(x1, 1.0).sample()
        x4 = Normal(x2, 1.0).sample()
        x5 = Normal(x2, 1.0).sample()
        x6 = Normal(x3, 1.0).sample()
        x7 = Normal((x4 + x5), 1.0).sample()
        x8 = Normal((x6 + x5), 1.0).sample()
        x9 = Normal((x0 + x6), 1.0).sample()
        x10 = Normal(x7 * x8 + 2, 1.0).sample()
        x11 = Normal(x7 + (x8 * x5), 1.0).sample()
        x12 = Normal((x6 * x8) - x5, 1.0).sample()
        x13 = Normal((x9 - 2*x6), 1.0).sample()
        x14 = Normal(x11 * x10, 1.0).sample()
        x15 = Normal(x2 + (x12 * x11), 1.0).sample()
        x16 = Normal(x13 - x12, 1.0).sample()

        y17 = Normal(x14, 0.1).sample()
        y18 = Normal(x15, 0.1).sample()
        y19 = Normal(x15, 0.1).sample()
        y20 = Normal(x16, 0.1).sample()
        y21 = Normal(x9, 0.1).sample()

        return (torch.tensor([x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16]),
                torch.tensor([y17, y18, y19, y20, y21]))

    def log_prior(self, x):
        PI =  Normal(0, 5).log_prob(x[:, 0])
        PI += Normal(x[:, 0], 1.0).log_prob(x[:, 1])
        PI += Normal(x[:, 1], 1.0).log_prob(x[:, 2])
        PI += Normal(x[:, 1], 1.0).log_prob(x[:, 3])
        PI += Normal(x[:, 2], 1.0).log_prob(x[:, 4])
        PI += Normal(x[:, 2], 1.0).log_prob(x[:, 5])
        PI += Normal(x[:, 3], 1.0).log_prob(x[:, 6])
        PI += Normal(x[:, 4] + x[:, 5], 1.0).log_prob(x[:, 7])
        PI += Normal(x[:, 6] + x[:, 5], 1.0).log_prob(x[:, 8])
        PI += Normal(x[:, 0] + x[:, 6], 1.0).log_prob(x[:, 9])
        PI += Normal(x[:, 7] * x[:, 8] + 2, 1.0).log_prob(x[:, 10])
        PI += Normal(x[:, 7] + (x[:, 8] + x[:, 5]), 1.0).log_prob(x[:, 11])
        PI += Normal((x[:, 6] * x[:, 8]) - x[:, 5], 1.0).log_prob(x[:, 12])
        PI += Normal((x[:, 9] - 2*x[:, 6]), 1.0).log_prob(x[:, 13])
        PI += Normal(x[:, 11] * x[:, 10], 1.0).log_prob(x[:, 14])
        PI += Normal(x[:, 2] + x[:, 12] * x[:, 11], 1.0).log_prob(x[:, 15])
        PI += Normal(x[:, 13] - x[:, 12], 1.0).log_prob(x[:, 16])

        return PI

    def log_likelihood(self, x, y):
        LAMBDA =   Normal(x[:, 14], 0.1).log_prob(y[:, 0])
        LAMBDA +=  Normal(x[:, 15], 0.1).log_prob(y[:, 1])
        LAMBDA +=  Normal(x[:, 15], 0.1).log_prob(y[:, 2])
        LAMBDA +=  Normal(x[:, 16], 0.1).log_prob(y[:, 3])
        LAMBDA +=  Normal(x[:, 9], 0.1).log_prob(y[:, 4])

        return LAMBDA




# like 4 but proper division inversion
class BiggerGraph6:
    def __init__(self):
        self.dim_latent = 17
        self.dim_condition = 5

        self.faithful_adjacency = [[0,1],[0,6],[0,9],
                                   [1,12],[1,2],[1,3],[1,6],[1,20],[1,21],
                                   [10,14],[10,15],[10,20],[10,21],
                                   [11,10],[11,14],[11,15],[11,20],[11,21],
                                   [12,11],[12,15],[12,2],[12,5],[12,8],[12,20],[12,21],
                                   [13,1],[13,12],[13,16],[13,6],[13,21],
                                   [14,15],[14,17],[14,20],[14,21],
                                   [15,17],[15,18],[15,19],[15,20],[15,21],
                                   [16,1],[16,12],[16,6],[16,20],[16,21],
                                   [2,11],[2,15],[2,4],[2,5],[2,8],[2,20],[2,21],
                                   [3,12],[3,2],[3,6],[3,20],[3,21],
                                   [4,11],[4,15],[4,5],[4,7],[4,8],[4,20],[4,21],
                                   [5,11],[5,15],[5,7],[5,8],[5,20],[5,21],
                                   [6,12],[6,2],[6,5],[6,8],[6,20],[6,21],
                                   [7,10],[7,11],[7,15],[7,8],[7,20],[7,21],
                                   [8,10],[8,11],[8,15],[8,20],[8,21],
                                   [9,1],[9,13],[9,6],[9,21]] + [[i, i] for i in range(self.dim_latent)]

        self.rand_adjacency = []
        while len(self.rand_adjacency) != len(self.faithful_adjacency):
            self.rand_adjacency = list(set([(i, j)
                                            for i in range(self.dim_latent)
                                            for j in range(self.dim_latent + self.dim_condition) if np.random.rand() > 0.75] + [(i, i) for i in range(self.dim_latent)]))

    def sample(self):
        x0 = Normal(10.0, 1.0).sample()
        x1 = Normal(x0 + 2.0, 1.0).sample()
        x2 = Normal(x1 - 3.0, 1.0).sample()
        x3 = Normal(x1 + 1.0, 1.0).sample()
        x4 = Normal(x2 * 2.0, 1.0).sample()
        x5 = Normal(x2 * 3.0, 1.0).sample()
        x6 = Normal(x3 * 0.5, 1.0).sample()
        x7 = Normal((x4 * x5), 1.0).sample()
        x8 = Normal((x6 - x5), 1.0).sample()
        x9 = Normal((x0 + x6 * x8), 1.0).sample()
        x10 = Normal(x7 * x8 + 2, 1.0).sample()
        x11 = Normal(x7 + (x8 * x5), 1.0).sample()
        x12 = Normal((x6 * x8) - x5, 1.0).sample()
        x13 = Normal((x9 - 2*x6), 1.0).sample()
        x14 = Normal(x11 + x10, 1.0).sample()
        x15 = Normal(x2 + (x12 - x11), 1.0).sample()
        x16 = Normal(x13 - x12, 1.0).sample()

        y17 = Normal(x14, 0.1).sample()
        y18 = Normal(x15, 0.1).sample()
        y19 = Normal(x15, 0.1).sample()
        y20 = Normal(x16, 0.1).sample()
        y21 = Normal(x9, 0.1).sample()

        return (torch.tensor([x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16]),
                torch.tensor([y17, y18, y19, y20, y21]))

    def log_prior(self, x):
        PI =  Normal(10.0, 1.0).log_prob(x[:, 0])
        PI += Normal(x[:, 0] + 2.0, 1.0).log_prob(x[:, 1])
        PI += Normal(x[:, 1] - 3.0, 1.0).log_prob(x[:, 2])
        PI += Normal(x[:, 1] * 2.0, 1.0).log_prob(x[:, 3])
        PI += Normal(x[:, 2] * 3.0, 1.0).log_prob(x[:, 4])
        PI += Normal(x[:, 2] * 3.0, 1.0).log_prob(x[:, 5])
        PI += Normal(x[:, 3] * 0.5, 1.0).log_prob(x[:, 6])
        PI += Normal(x[:, 4] * x[:, 5], 1.0).log_prob(x[:, 7])
        PI += Normal(x[:, 6] - x[:, 5], 1.0).log_prob(x[:, 8])
        PI += Normal(x[:, 0] + x[:, 6] * x[:, 8], 1.0).log_prob(x[:, 9])
        PI += Normal(x[:, 7] * x[:, 8] + 2, 1.0).log_prob(x[:, 10])
        PI += Normal(x[:, 7] + (x[:, 8] * x[:, 5]), 1.0).log_prob(x[:, 11])
        PI += Normal((x[:, 6] * x[:, 8]) - x[:, 5], 1.0).log_prob(x[:, 12])
        PI += Normal((x[:, 9] - 2*x[:, 6]), 1.0).log_prob(x[:, 13])
        PI += Normal(x[:, 11] + x[:, 10], 1.0).log_prob(x[:, 14])
        PI += Normal(x[:, 2] + (x[:, 12] - x[:, 11]), 1.0).log_prob(x[:, 15])
        PI += Normal(x[:, 13] - x[:, 12], 1.0).log_prob(x[:, 16])

        return PI

    def log_likelihood(self, x, y):
        LAMBDA =   Normal(x[:, 14], 0.1).log_prob(y[:, 0])
        LAMBDA +=  Normal(x[:, 15], 0.1).log_prob(y[:, 1])
        LAMBDA +=  Normal(x[:, 15], 0.1).log_prob(y[:, 2])
        LAMBDA +=  Normal(x[:, 16], 0.1).log_prob(y[:, 3])
        LAMBDA +=  Normal(x[:, 9], 0.1).log_prob(y[:, 4])

        return LAMBDA




class SimpleArithmeticCircuit:
    def __init__(self):
        self.dim_latent = 6
        self.dim_condition = 2

        self.faithful_adjacency = [[0,1],[0,2],[0,3],[1,3],[2,1],[2,3],[3,6],[3,7],[4,3],[4,5],[5,3],[5,7]] + [[i, i] for i in range(self.dim_latent)]

        self.rand_adjacency = []
        while len(self.rand_adjacency) != len(self.faithful_adjacency):
            self.rand_adjacency = list(set([(i, j)
                                            for i in range(self.dim_latent)
                                            for j in range(self.dim_latent + self.dim_condition)
                                            if np.random.rand() > 0.75] + [(i, i) for i in range(self.dim_latent)]))



    def sample(self):
        x0 = Laplace(5, 1.0).sample()
        x1 = Laplace(-2, 1.0).sample()

        x2 = Normal(torch.tanh(x0 + x1 - 2.8), 0.1).sample()
        x3 = Normal(x0 * x1, 0.1).sample()

        x4 = Normal(7.0, 2.0).sample()
        x5 = Normal(torch.tanh(x3 + x4), 0.1).sample()

        y0 = Normal(x3, 0.1).sample()
        y1 = Normal(x5, 0.1).sample()

        return torch.tensor([x0, x1, x2, x3, x4, x5]), torch.tensor([y0, y1])

    def log_prior(self, x):
        PI  = Laplace(5, 1.0).log_prob(x[:, 0])
        PI += Laplace(-2, 1.0).log_prob(x[:, 1])

        PI += Normal(torch.tanh(x[:, 0] + x[:, 1] - 2.8), 0.1).log_prob(x[:, 2])
        PI += Normal(x[:, 0] * x[:, 1], 0.1).log_prob(x[:, 3])

        PI += Normal(7.0, 2.0).log_prob(x[:, 4])
        PI += Normal(torch.tanh(x[:, 3] + x[:, 4]), 0.1).log_prob(x[:, 5])

        return PI

    def log_likelihood(self, x, y):
        LAMBDA  = Normal(x[:, 3], 0.1).log_prob(y[:, 0])
        LAMBDA += Normal(x[:, 5], 0.1).log_prob(y[:, 1])

        return LAMBDA


class BernoulliGMM:
    def __init__(self, prob=0.5):
        self.dim_latent = 1
        self.dim_condition = 1
        self.prob = prob

        self.faithful_adjacency = [[0, 1], [0, 0]]

        self.rand_adjacency = [[0, 1], [0, 0]]


    def sample(self):
        y0 = Bernoulli(self.prob).sample()
        x0 = Normal(y0 * 2 - 1, 0.1).sample()

        return torch.tensor([x0]), torch.tensor([y0])


    def log_prior(self, x):
        # TODO parametrize mixture prob for 0.5
        m0 = Normal(-1, 0.1).log_prob(x[:, 0]) + math.log(0.5)
        m1 = Normal( 1, 0.1).log_prob(x[:, 0]) + math.log(0.5)

        return torch.logsumexp(torch.cat([m0.unsqueeze(0), m1.unsqueeze(0)], dim=0), dim=0)

    def log_likelihood(self, x, y):
        return Normal(y[:, 0] * 2 - 1, 0.1).log_prob(x[:, 0])

