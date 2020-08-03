import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import copy
import random


class OUNoise:
    """From https://github.com/IgnacioCarlucho/DDPG_MountainCar/blob/master/ou_noise.py#L29"""
    def __init__(self,action_dimension,mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

class ReplayBuffer():
    def __init__(self, size=1000000):
        self.buffer = []
        self.n_samples = 128
        self.max_size = size

    def len(self):
        return len(self.buffer)

    def __len__(self):
        return len(self.buffer)

    def add(self, sample):
        self.buffer.append(sample)
        if len(self.buffer) > self.max_size:
            del self.buffer[0]

    def sample(self):
        samples = random.sample(self.buffer, k=self.n_samples)
        data = [*zip(samples)]
        data_dict = {"o": data[0], "a": data[1], "r": data[2], "o_next": data[3], "done": data[3]}
        return data_dict

    def sample_tensors(self, n=128):
        samples = random.sample(self.buffer, k=n)
        data = [*zip(*samples)]
        #data_dict = {"o": torch.stack(data[0]), "a": torch.stack(data[1]), "r": torch.stack(data[2]), "o_next": torch.stack(data[3]), "done": torch.stack(data[4])}
        data_dict = {"o": torch.cat(data[0]).view(n,-1), "a":torch.cat(data[1]).view(n, -1), "r": torch.stack(data[2]), "o_next": torch.cat(data[3]).view(n, -1), "done": torch.stack(data[4])}
        return data_dict


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims_actor=(256, 256), hidden_dims_critic=(256,256)):
        super().__init__()
        self.actor = Actor(obs_dim, hidden_dims_actor, action_dim)
        self.critic = Critic(obs_dim + action_dim, hidden_dims_critic)


class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        layers += [nn.Linear(input_dim, hidden_dims[0]), nn.ReLU()]
        for i in range(len(hidden_dims)-1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.ReLU()]
        layers += [nn.Linear(hidden_dims[-1], output_dim), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, observation):
        return self.net(observation)

class StochActor(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        layers += [nn.Linear(input_dim, hidden_dims[0]), nn.ReLU()]
        for i in range(len(hidden_dims)-1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.ReLU()]
        #layers += [nn.Linear(hidden_dims[-1], output_dim), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)
        self.mu_lay = nn.Linear(hidden_dims[-1], output_dim)
        self.sigma_lay = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, observation, sample=True, greedy=False):
        res = self.net(observation)
        mu, sigma = self.mu_lay(res), torch.exp(torch.clamp(self.sigma_lay(res), -20, 2))
        pi_dist = torch.distributions.normal.Normal(mu, sigma)
        if greedy:
            act = mu
        else:
            act = pi_dist.rsample()

        if not sample:
            logp_pi = pi_dist.log_prob(act).sum(dim=-1)
            logp_pi -= (2 * (np.log(2) - act - F.softplus(-2 * act))).sum(dim=1)
            return torch.tanh(act), logp_pi
        else:
            return torch.tanh(act)


class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        layers = []
        layers += [nn.Linear(input_dim, hidden_dims[0]), nn.ReLU()]
        for i in range(len(hidden_dims) - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.ReLU()]
        layers += [nn.Linear(hidden_dims[-1], 1), nn.Identity()]
        self.net = nn.Sequential(*layers)

    def forward(self, observation, action):
        x = torch.cat([observation, action], dim=-1)
        return self.net(x)


def loss_critic(val, target, f_hyst=1.0):
    diffs = target - val
    if not np.isclose(f_hyst, 1.0):
        diffs[diffs < 0] *= f_hyst
    return torch.mean(diffs**2)

def check_cuda():
    if torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda:0")
    else:
        print("No CUDA found")
        return torch.device("cpu")

def convert_inputs_to_tensors(o, r, done, device):
    o = torch.tensor(o, dtype=torch.float, device=device)
    r = torch.tensor(np.array(float(r)), dtype=torch.float, device=device)
    done = torch.tensor(np.array(float(done)), dtype=torch.float, device=device)
    return o, r, done

def update_target_networks(networks, targets, tau):
    with torch.no_grad():
        for network, target in zip(networks, targets):
            for par, par_target in zip(network.parameters(), target.parameters()):
                par_target.data.copy_((1 - tau) * par_target + tau * par.data)
