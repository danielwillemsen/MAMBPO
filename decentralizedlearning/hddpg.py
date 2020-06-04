import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random
import numpy as np

class HDDPGAgent:
    def __init__(self, obs_dim, action_dim):
        self.ac = ActorCritic(obs_dim, action_dim)
        self.ac_target = copy.deepcopy(self.ac)
        for par in self.ac_target.parameters():
            par.requires_grad = False

        self.buffer = ReplayBuffer()
        self.o_old = None
        self.a_old = None
        self.gamma = 0.99
        self.tau = 0.001

        self.optimizer_critic = torch.optim.Adam(self.ac.critic.parameters(), lr=0.001)
        self.optimizer_actor = torch.optim.Adam(self.ac.actor.parameters(), lr=0.0001)

    def reset(self):
        self.o_old = None
        self.a_old = None

    def step(self, o, r):
        o = torch.Tensor(o)
        r = torch.Tensor(np.array(r[0]))
        #print(o)
        #print("R: " + str(r))
        if self.o_old is not None:
            self.buffer.add((self.o_old, self.a_old, r, o))

        if self.buffer.len() > self.buffer.n_samples:
            # Sample Minibatch
            b = self.buffer.sample_tensors()

            # Update Critic
            self.optimizer_critic.zero_grad()
            with torch.no_grad():
                y = b["r"].unsqueeze(-1) + self.gamma * self.ac_target.critic(b["o_next"],self.ac_target.actor(b["o_next"]))
            loss_critic = F.mse_loss(self.ac.critic(b["o"], b["a"]), y)
            #print(loss_critic)
            loss_critic.backward()
            self.optimizer_critic.step()

            # Update Actor
            for par in self.ac.critic.parameters():
                par.requires_grad = False

            self.optimizer_actor.zero_grad()
            loss_actor = -torch.mean(self.ac.critic(b["o"],self.ac.actor(b["o"])))
            #print("Loss actor: " + str(loss_actor))

            loss_actor.backward()

            self.optimizer_actor.step()

            for par in self.ac.critic.parameters():
                par.requires_grad = True

            # Update Target Networks
            with torch.no_grad():
                for par, par_target in zip(self.ac.parameters(), self.ac_target.parameters()):
                    par_target.data.copy_((1-self.tau) * par_target + self.tau * par.data)
                    #par_target.data._add((1-self.polyak) * par.data)



        # Select Action
        with torch.no_grad():
            action = self.ac.actor(o.unsqueeze(0)).squeeze()
            action = torch.clamp(action + torch.randn(action.size())*0.1,0., 1.0)

        self.o_old = o
        if action.size() == torch.Size([]):
            self.a_old = action.unsqueeze(0)
        else:
            self.a_old = action
        #print(action)
        return action.detach().numpy()

class ReplayBuffer():
    def  __init__(self):
        self.buffer = []
        self.n_samples = 128

    def len(self):
        return len(self.buffer)

    def add(self, sample):
        self.buffer.append(sample)

    def sample(self):
        samples = random.choices(self.buffer, k=self.n_samples)
        data = [*zip(samples)]
        data_dict = {"o": data[0], "a": data[1], "r": data[2], "o_next": data[3]}
        return data_dict

    def sample_tensors(self):
        samples = random.choices(self.buffer, k=self.n_samples)
        data = [*zip(*samples)]
        data_dict = {"o": torch.stack(data[0]), "a": torch.stack(data[1]), "r": torch.stack(data[2]), "o_next": torch.stack(data[3])}
        return data_dict

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims_actor = (128,128,128), hidden_dims_critic=(128,128,128)):
        super().__init__()
        self.actor =  Actor(obs_dim, hidden_dims_actor, action_dim)
        self.critic = Critic(obs_dim + action_dim, hidden_dims_critic)

class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        layers += [nn.Linear(input_dim, hidden_dims[0]), nn.ReLU()]
        for i in range(len(hidden_dims)-1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.ReLU()]
        layers += [nn.Linear(hidden_dims[-1], output_dim), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, observation):
        return self.net(observation)


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

