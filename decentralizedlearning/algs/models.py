import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import copy
import random
import logging

class EnsembleModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, obs_dim, n_models, monitor_losses=False, use_stochastic=True):
        super().__init__()
        self.monitor_losses = monitor_losses
        self.use_stochastic = use_stochastic
        if monitor_losses:
            self.logger = logging.getLogger('root')
        self.models = nn.ModuleList([Model(input_dim, hidden_dims, obs_dim, use_stochastic=use_stochastic) for i in range(n_models)])

    def forward(self, observation, action):
        model_outs = [model(observation, action) for model in self.models]
        o_next_pred, r_pred = (zip(*model_outs))
        o_next_pred = [torch.stack(item) for item in zip(*o_next_pred)]
        r_pred = [torch.stack(item) for item in zip(*r_pred)]
        return o_next_pred, r_pred

    def update_step(self, optim, samples):
        o_next_pred, r_pred = self(samples["o"], samples["a"])
        target_o = samples["o_next"]
        target_r = samples["r"].unsqueeze(-1)

        sigma_o = o_next_pred[1]
        sigma_r = r_pred[1]
        mu_o = o_next_pred[0]
        mu_r = r_pred[0]

        loss1 = torch.mean((mu_o - target_o) * (mu_o - target_o))
        loss2 = torch.mean((mu_r - target_r) * (mu_r - target_r))
        if self.use_stochastic:
            loss3 = torch.mean(torch.log(torch.prod(sigma_o, 2)))# + torch.mean(torch.log(torch.prod(sigma_r, 2)))
        else:
            loss3 = 0.
        loss = loss1 + loss2 + loss3

        optim.zero_grad()
        loss.backward()
        optim.step()

    def log_loss(self, samples, name):
        o_next_pred, r_pred = self(samples["o"], samples["a"])
        target_o = samples["o_next"]
        target_r = samples["r"].unsqueeze(-1)

        sigma_o = o_next_pred[1]
        sigma_r = r_pred[1]
        mu_o = o_next_pred[0]
        mu_r = r_pred[0]

        loss_o = torch.mean((mu_o - target_o) * (mu_o - target_o))
        loss_r = torch.mean((mu_r - target_r) * (mu_r - target_r))
        self.logger.info(name+"_loss_o:"+str(loss_o.data))
        self.logger.info(name+"_loss_r:"+str(loss_r.data))
        self.logger.info(name+"_abs_o:"+str(torch.mean(torch.abs(target_o)).data))
        self.logger.info(name+"_abs_r:"+str(torch.mean(torch.abs(target_r)).data))
        self.logger.info("\n")

    def generate(self, samples, actor, diverse=True):
        with torch.no_grad():
            o = samples["o"]
            if not diverse:
                a = samples["a"]#actor(o, greedy=False)
            else:
                a = actor(o, greedy=False)
            n_models = len(self.models)
            o_next_pred, r_pred = self(o, a)
            mu_o = o_next_pred[0]
            mu_r = r_pred[0]
            ret_samples = []
            randomlist = random.choices(range(0, n_models), k=128)
            for i in range(128):
                ret_samples.append((o[i], a[i], mu_r[randomlist[i],i][0], mu_o[randomlist[i],i], samples["done"][i]))
                # ret_samples.append((samples["o"][i], samples["a"][i], samples["r"][i], samples["o_next"][i], samples["done"][i]))

        return ret_samples

class Model(nn.Module):
    """Model.
    Contains a probabilistic world model. Outputs 2 lists: one containing mu, sigma of reward, second containing mu, sigma of observation
    """
    def __init__(self, input_dim, hidden_dims, obs_dim, use_stochastic=True):
        """__init__.

        :param input_dim:
        :param hidden_dims:
        :param output_dim:
        """
        super().__init__()
        self.use_stochastic = use_stochastic
        layers = []
        layers += [nn.Linear(input_dim, hidden_dims[0]), nn.Tanh()]
        for i in range(len(hidden_dims) - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        self.mu_output = nn.Linear(hidden_dims[-1], obs_dim)
        self.mu_reward = nn.Linear(hidden_dims[-1], 1)

        if self.use_stochastic:
            self.sigma_output = nn.Linear(hidden_dims[-1], obs_dim)
            self.sigma_reward = nn.Linear(hidden_dims[-1], 1)


    def forward(self, observation, action):
        x = torch.cat([observation, action], dim=-1)
        x = self.net(x)
        if self.use_stochastic:
            s_output = 10.*torch.tanh(self.sigma_output(x))
            s_reward = 10.*torch.tanh(self.sigma_reward(x))
            return [self.mu_output(x), torch.exp(s_output)], [self.mu_reward(x), torch.exp(s_reward)]
        else:
            mu_out = self.mu_output(x)
            mu_rew = self.mu_reward(x)

            s_output = mu_out*0.
            s_reward = mu_out*0.
            return [mu_out, s_output], [mu_rew, s_reward]


    def sample(self, observation, action):
        with torch.no_grad():
            new_o, r = self.forward(observation, action)
            new_o = torch.normal(new_o[0], torch.sqrt(new_o[1]))
            r = torch.normal(r[0], 0.*torch.sqrt(r[1]))
        return new_o, r
