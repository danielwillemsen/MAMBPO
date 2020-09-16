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

class EfficientReplayBuffer():
    def __init__(self, size=50000, batch_size=128, device="cpu"):
        self.initialized = False
        self.o = None
        self.r = None
        self.a = None
        self.o_next = None
        self.done = None
        self.device = device
        self.n_samples = batch_size
        self.max_size = size
        self.next_idx = 0
        self.current_size = 0

    def reallocate(self, size=None):
        if size:
            self.max_size = size
        self.initialized = False
        self.next_idx = 0
        self.current_size = 0

    def len(self):
        return self.current_size

    def __len__(self):
        return self.current_size

    def add_multiple(self, sample):
        self.initialize(sample)
        n_samples = sample[0].shape[0]
        self.o[self.next_idx:self.next_idx+n_samples] = sample[0]
        self.a[self.next_idx:self.next_idx+n_samples] = sample[1]
        self.r[self.next_idx:self.next_idx+n_samples] = sample[2]
        self.o_next[self.next_idx:self.next_idx+n_samples] = sample[3]
        self.done[self.next_idx:self.next_idx+n_samples] = sample[4]
        self.next_idx += n_samples
        self.current_size += n_samples
        if self.current_size > self.max_size:
            self.current_size = self.max_size
        if self.next_idx >= self.max_size:
            self.next_idx = self.next_idx % self.max_size

    def add(self, sample):
        #Initialize buffer if not existing yet
        self.initialize(sample)
        self.o[self.next_idx] = sample[0]
        self.a[self.next_idx] = sample[1]
        self.r[self.next_idx] = sample[2]
        self.o_next[self.next_idx] = sample[3]
        self.done[self.next_idx] = sample[4]
        self.next_idx += 1
        if self.current_size < self.max_size:
            self.current_size += 1
        if self.next_idx >= self.max_size:
            self.next_idx = 0

    def initialize(self, sample):
        if not self.initialized:
            self.o = torch.zeros((self.max_size, sample[0].shape[-1]), device=self.device, dtype=torch.float32)
            self.a = torch.zeros((self.max_size, sample[1].shape[-1]), device=self.device, dtype=torch.float32)
            self.r = torch.zeros((self.max_size), device=self.device, dtype=torch.float32)
            self.o_next = torch.zeros((self.max_size, sample[3].shape[-1]), device=self.device, dtype=torch.float32)
            self.done = torch.zeros((self.max_size), device=self.device, dtype=torch.float32)
            self.initialized = True

    def sample(self):
        raise NotImplementedError
        # samples = random.sample(self.buffer, k=self.n_samples)
        # data = [*zip(samples)]
        # data_dict = {"o": data[0], "a": data[1], "r": data[2], "o_next": data[3], "done": data[3]}
        # return data_dict

    def sample_tensors(self, n=None):
        if not n:
            n=self.n_samples
        idx = np.random.randint(self.current_size, size=n)
        data_dict = {"o": self.o[idx], "a":self.a[idx], "r": self.r[idx], "o_next": self.o_next[idx], "done": self.done[idx]}
        return data_dict

    def get_all(self):
        # buffer_copy = random.sample(self.buffer, k=len(self.buffer))
        # n = len(buffer_copy)
        # data = [*zip(*buffer_copy)]

        # data_dict = {"o": torch.stack(data[0]), "a": torch.stack(data[1]), "r": torch.stack(data[2]), "o_next": torch.stack(data[3]), "done": torch.stack(data[4])}
        data_dict = {"o": self.o[:self.current_size], "a": self.a[:self.current_size],
                     "r": self.r[:self.current_size], "o_next": self.o_next[:self.current_size], "done": self.done[:self.current_size]}
        return data_dict

    def get_all_split(self, holdout):
        idx_shuffled = np.random.permutation(self.current_size)
        n = len(idx_shuffled)
        n_val = int(holdout*n)
        idx_val = idx_shuffled[:n_val]
        idx_train = idx_shuffled[n_val:]
        data_val_dict = {"o": self.o[idx_val], "a": self.a[idx_val],
                     "r": self.r[idx_val], "o_next": self.o_next[idx_val],
                         "done": self.done[idx_val]}
        data_train_dict = {"o": self.o[idx_train], "a": self.a[idx_train],
                     "r": self.r[idx_train], "o_next": self.o_next[idx_train],
                         "done": self.done[idx_train]}

        return data_train_dict, data_val_dict, n_val, n-n_val

    def get_buffer_split(self, holdout):
        n = len(self)
        n_val = int(holdout*n)
        buff_train = EfficientReplayBuffer(size=self.max_size, device=self.device)
        buff_val = EfficientReplayBuffer(size=self.max_size, device=self.device)
        buff_train.add_multiple((self.o[n_val:self.current_size].clone(), self.a[n_val:self.current_size].clone(), self.r[n_val:self.current_size].clone(), self.o_next[n_val:self.current_size].clone(), self.done[n_val:self.current_size].clone()))
        buff_val.add_multiple((self.o[:n_val].clone(), self.a[:n_val].clone(), self.r[:n_val].clone(), self.o_next[:n_val].clone(), self.done[:n_val].clone()))
        return buff_train, buff_val

class ReplayBuffer():
    def __init__(self, size=1000000, batch_size=128, device="cpu"):
        self.buffer = []
        self.n_samples = batch_size
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

    def sample_tensors(self, n=None):
        if not n:
            n=self.n_samples
        samples = random.choices(self.buffer, k=n)
        data = [*zip(*samples)]
        #data_dict = {"o": torch.stack(data[0]), "a": torch.stack(data[1]), "r": torch.stack(data[2]), "o_next": torch.stack(data[3]), "done": torch.stack(data[4])}
        data_dict = {"o": torch.cat(data[0]).view(n,-1), "a":torch.cat(data[1]).view(n, -1), "r": torch.stack(data[2]), "o_next": torch.cat(data[3]).view(n, -1), "done": torch.stack(data[4])}
        return data_dict

    def get_all(self):
        buffer_copy = random.sample(self.buffer, k=len(self.buffer))
        n = len(buffer_copy)
        data = [*zip(*buffer_copy)]

        # data_dict = {"o": torch.stack(data[0]), "a": torch.stack(data[1]), "r": torch.stack(data[2]), "o_next": torch.stack(data[3]), "done": torch.stack(data[4])}
        data_dict = {"o": torch.cat(data[0]).view(n, -1), "a": torch.cat(data[1]).view(n, -1),
                     "r": torch.stack(data[2]), "o_next": torch.cat(data[3]).view(n, -1), "done": torch.stack(data[4])}
        return data_dict

    def get_all_split(self, holdout):
        buffer_copy = random.sample(self.buffer, k=len(self.buffer))
        n = len(buffer_copy)
        n_val = int(holdout*n)
        data_val = [*zip(*buffer_copy[:n_val])]
        data_train = [*zip(*buffer_copy[n_val:])]
        data_val_dict = {"o": torch.cat(data_val[0]).view(n_val, -1), "a": torch.cat(data_val[1]).view(n_val, -1),
                     "r": torch.stack(data_val[2]), "o_next": torch.cat(data_val[3]).view(n_val, -1),
                         "done": torch.stack(data_val[4])}
        data_train_dict = {"o": torch.cat(data_train[0]).view(n-n_val, -1), "a": torch.cat(data_train[1]).view(n-n_val, -1),
                     "r": torch.stack(data_train[2]), "o_next": torch.cat(data_train[3]).view(n-n_val, -1),
                           "done": torch.stack(data_train[4])}

        return data_train_dict, data_val, n_val, n-n_val

    def get_buffer_split(self, holdout):
        buffer_copy = random.sample(self.buffer, k=len(self.buffer))
        n = len(buffer_copy)
        n_val = int(holdout*n)
        buff_train = ReplayBuffer()
        buff_val = ReplayBuffer()
        buff_train.buffer = buffer_copy[n_val:]
        buff_val.buffer = buffer_copy[:n_val]
        return buff_train, buff_val



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
        mu, sigma = self.mu_lay(res), torch.exp(torch.clamp(self.sigma_lay(res), -20., 2.))
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
