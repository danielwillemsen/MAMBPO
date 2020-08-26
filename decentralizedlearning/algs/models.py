import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import copy
import random
import logging
import itertools

class EnsembleModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, obs_dim, n_models, monitor_losses=False, use_stochastic=True, name="cheetah", device=None):
        super().__init__()
        self.device = device
        self.monitor_losses = monitor_losses
        self.use_stochastic = use_stochastic
        self.name = name
        if monitor_losses:
            self.logger = logging.getLogger('root')
        self.models = nn.ModuleList([Model(input_dim, hidden_dims, obs_dim, use_stochastic=use_stochastic) for i in range(n_models)])

    def termination_fn(self, next_obs, action):
        if self.name == "Hopper-v2":
            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done = (height > .7) \
                       * (torch.abs(angle) < .2)
            return (~not_done).float()
        else:
            return torch.zeros(action.shape[0],device=self.device)

    def forward(self, observation, action):
        x = torch.cat([observation, action], dim=-1)
        model_outs = [model(x) for model in self.models]
        o_next_pred, r_pred = (zip(*model_outs))
        o_next_pred = [torch.stack(item) for item in zip(*o_next_pred)]
        r_pred = [torch.stack(item) for item in zip(*r_pred)]
        return o_next_pred, r_pred

    def update_step(self, optim, samples):
        o_next_pred, r_pred = self(samples["o"], samples["a"])
        target_o = samples["o_next"]
        target_r = samples["r"].unsqueeze(-1)

        log_var_o = o_next_pred[1]
        log_var_r = r_pred[1]
        inv_var_o = torch.exp(-log_var_o)
        inv_var_r = torch.exp(-log_var_r)
        mu_o = o_next_pred[0]
        mu_r = r_pred[0]

        if self.use_stochastic:
            loss1 = torch.mean((mu_o - target_o) * inv_var_o * (mu_o - target_o))
            loss2 = torch.mean((mu_r - target_r) * inv_var_r * (mu_r - target_r))
            loss3 = torch.mean(log_var_o) + torch.mean(log_var_r)
        else:
            loss1 = torch.mean((mu_o - target_o) * (mu_o - target_o))
            loss2 = torch.mean((mu_r - target_r) * (mu_r - target_r))
            loss3 = 0.
        loss = loss1 + loss2 + loss3

        optim.zero_grad()
        loss.backward()
        optim.step()

    def get_mse_losses(self, samples):
        with torch.no_grad():
            o_next_pred, r_pred = self(samples["o"], samples["a"])
            target_o = samples["o_next"]
            target_r = samples["r"].unsqueeze(-1)

            log_var_o = o_next_pred[1]
            log_var_r = r_pred[1]
            inv_var_o = torch.exp(-log_var_o)
            inv_var_r = torch.exp(-log_var_r)
            mu_o = o_next_pred[0]
            mu_r = r_pred[0]

            if self.use_stochastic:
                loss1 = torch.mean((mu_o - target_o) * (mu_o - target_o))
                loss2 = torch.mean((mu_r - target_r) * (mu_r - target_r))
            else:
                loss1 = torch.mean((mu_o - target_o) * (mu_o - target_o))
                loss2 = torch.mean((mu_r - target_r) * (mu_r - target_r))
        return loss1, loss2

    def train_models(self, optim, buffer, holdout=0.1):
        batch_size = 256
        buff_train, buff_val = buffer.get_buffer_split(holdout=0.1)
        epoch_iter = range(1000)#itertools.count()
        grad_steps = 0
        stop_count = 0
        sum_loss = 999.
        best = self.state_dict()
        for epoch in epoch_iter:
            for batch_num in range(int(len(buff_train)/batch_size)):
                self.update_step(optim, buff_train.sample_tensors(n=batch_size))
                grad_steps += 1
            loss_o, loss_r = self.get_mse_losses(buff_val.get_all())
            self.logger.info("Loss_mod:"+str((loss_o + loss_r).item()))
            if sum_loss > loss_o + loss_r:
                sum_loss = loss_o + loss_r
                stop_count = 0
                best = self.state_dict()
            else:
                stop_count += 1
            if stop_count >= 5:
                break
        self.load_state_dict(best)
        self.logger.info("Stopped. Epoch:"+ str(epoch)+ "Grad_steps:" +str(grad_steps))
        return

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
        self.logger.info(name+"_loss_o:"+str(loss_o.item()))
        self.logger.info(name+"_loss_r:"+str(loss_r.item()))
        self.logger.info(name + "_log_var_o:" + str(torch.mean(sigma_o.data).item()))
        self.logger.info(name + "_log_var_r:" + str(torch.mean(sigma_r.data).item()))

        self.logger.info(name+"_abs_o:"+str(torch.mean(torch.abs(target_o)).item()))
        self.logger.info(name+"_abs_r:"+str(torch.mean(torch.abs(target_r)).item()))
        self.logger.info("\n")

    def generate(self, samples, actor, diverse=True, batch_size=128, rollout_length=1):
        with torch.no_grad():
            o = samples["o"]
            if not diverse:
                a = samples["a"]#actor(o, greedy=False)
            else:
                a = actor(o, greedy=False)
            n_models = len(self.models)
            o_next_pred, r_pred = self(o, a)
            mu_o = o_next_pred[0]
            log_var_o = o_next_pred[1]
            mu_r = r_pred[0]
            log_var_r = r_pred[1]
            ret_samples = []
            randomlist = random.choices(range(0, n_models), k=batch_size)
            if self.use_stochastic:
                new_o = torch.normal(mu_o, torch.exp(0.5*log_var_o))
                r = torch.normal(mu_r, torch.exp(0.5*log_var_r))
            for i in range(batch_size):
                if self.use_stochastic:
                    ret_samples.append((o[i], a[i], r[randomlist[i],i][0], new_o[randomlist[i],i], samples["done"][i]))
                else:
                    ret_samples.append((o[i], a[i], mu_r[randomlist[i],i][0], mu_o[randomlist[i],i], samples["done"][i]))
                # ret_samples.append((samples["o"][i], samples["a"][i], samples["r"][i], samples["o_next"][i], samples["done"][i]))

        return ret_samples

    def generate_efficient(self, samples, actor, diverse=True, batch_size=128, rollout_length=1):
        ret_samples = []
        with torch.no_grad():
            #for i in range(rollout_length):
                #if i == 0:
            o = samples["o"]
            if not diverse:
                a = samples["a"] #actor(o, greedy=False)
            else:
                a = actor(o, greedy=False)
                #else:
                #    o = new_o_ret
                #    a = actor(o, greedy=False)
            n_models = len(self.models)
            o_next_pred, r_pred = self(o, a)
            mu_o = o_next_pred[0]
            log_var_o = o_next_pred[1]
            mu_r = r_pred[0]
            log_var_r = r_pred[1]
            ret_samples = []
            idx = [i for i in range(batch_size)]
            randomlist = random.choices(range(n_models), k=batch_size)
            if self.use_stochastic:
                new_o = torch.normal(mu_o, torch.exp(0.5*log_var_o))
                r = torch.normal(mu_r, torch.exp(0.5*log_var_r))
            new_o_ret = new_o[randomlist, idx, :]
            done = samples["done"]
            #done = self.termination_fn(new_o_ret, a)
            ret_samples.append((o, a, r[randomlist, idx, :].squeeze(-1), new_o_ret, done))
            return ret_samples
            # for i in range(batch_size):
            #     if self.use_stochastic:
            #         ret_samples.append((o[i], a[i], r[randomlist[i],i][0], new_o[randomlist[i],i], samples["done"][i]))
            #     else:
            #         ret_samples.append((o[i], a[i], mu_r[randomlist[i],i][0], mu_o[randomlist[i],i], samples["done"][i]))

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
        self.MAX_LOG_VAR = torch.tensor(0.5, dtype=torch.float32)
        self.MIN_LOG_VAR = torch.tensor(-10., dtype=torch.float32)
        layers = []
        layers += [nn.Linear(input_dim, hidden_dims[0]), nn.Tanh()]
        for i in range(len(hidden_dims) - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        self.mu_output = nn.Linear(hidden_dims[-1], obs_dim)
        self.mu_reward = nn.Linear(hidden_dims[-1], 1)

        if self.use_stochastic:
            self.var_output = nn.Linear(hidden_dims[-1], obs_dim)
            self.var_reward = nn.Linear(hidden_dims[-1], 1)

    def forward(self, input):
        # x = torch.cat([observation, action], dim=-1)
        x = self.net(input)
        if self.use_stochastic:
            log_var_output = self.var_output(x)
            log_var_reward = self.var_reward(x)

            log_var_output = self.MAX_LOG_VAR - F.softplus(self.MAX_LOG_VAR - log_var_output)
            log_var_reward = self.MAX_LOG_VAR - F.softplus(self.MAX_LOG_VAR - log_var_reward)

            log_var_output = self.MIN_LOG_VAR + F.softplus(log_var_output - self.MIN_LOG_VAR)
            log_var_reward = self.MIN_LOG_VAR + F.softplus(log_var_reward - self.MIN_LOG_VAR)

            return [self.mu_output(x), log_var_output], [self.mu_reward(x), log_var_reward]
        else:
            mu_out = self.mu_output(x)
            mu_rew = self.mu_reward(x)

            var_output = mu_out*0.
            var_reward = mu_out*0.
            return [mu_out, var_output], [mu_rew, var_reward]

    #
    # def sample(self, observation, action):
    #     with torch.no_grad():
    #         new_o, r = self.forward(observation, action)
    #         new_o = torch.normal(new_o[0], torch.sqrt(new_o[1]))
    #         r = torch.normal(r[0], torch.sqrt(r[1]))
    #     return new_o, r
