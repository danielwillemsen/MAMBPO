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
        self.n_models = n_models
        self.n_elites = 5
        if monitor_losses:
            self.logger = logging.getLogger('root')
        self.models = nn.ModuleList([Model(input_dim, hidden_dims, obs_dim, use_stochastic=use_stochastic) for i in range(n_models)])
        self.elites = [self.models[i] for i in range(self.n_elites)]

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
        o_next_pred[0] = observation + o_next_pred[0]
        r_pred = [torch.stack(item) for item in zip(*r_pred)]
        return o_next_pred, r_pred

    def forward_elites(self, observation, action):
        x = torch.cat([observation, action], dim=-1)
        model_outs = [model(x) for model in self.elites]
        o_next_pred, r_pred = (zip(*model_outs))
        o_next_pred = [torch.stack(item) for item in zip(*o_next_pred)]
        o_next_pred[0] = observation + o_next_pred[0]
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
            l1 = torch.sum(torch.mean(torch.cat((((mu_o - target_o) * inv_var_o * (mu_o - target_o)),((mu_r - target_r) * inv_var_r * (mu_r - target_r))), dim=-1),dim=(-1,-2)))
            l2 = torch.sum(torch.mean(torch.cat((log_var_o, log_var_r), dim=-1),dim=(-1,-2)))
            #loss1 = torch.sum(torch.mean((mu_o - target_o) * inv_var_o * (mu_o - target_o),dim=(-1,-2)))
            #loss2 = torch.sum(torch.mean(,dim=(-1,-2)))
            #loss3 = torch.mean(log_var_o) + torch.mean(log_var_r)
        else:
            loss1 = torch.mean((mu_o - target_o) * (mu_o - target_o))
            loss2 = torch.mean((mu_r - target_r) * (mu_r - target_r))
            loss3 = 0.
        loss = l1+l2

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
                l1 = torch.mean(torch.cat((((mu_o - target_o) * (mu_o - target_o)),
                                                     ((mu_r - target_r) * (mu_r - target_r))), dim=-1),
                                          dim=(-1, -2))
                #l1 = torch.mean(l1)
                loss1 = torch.mean((mu_o - target_o) * (mu_o - target_o))
                loss2 = torch.mean((mu_r - target_r) * (mu_r - target_r))
            else:
                loss1 = torch.mean((mu_o - target_o) * (mu_o - target_o))
                loss2 = torch.mean((mu_r - target_r) * (mu_r - target_r))
        #return loss1, loss2
        return l1

    def train_models(self, optim, buffer, holdout=0.2):
        batch_size = 256
        buff_train, buff_val = buffer.get_buffer_split(holdout=holdout)
        epoch_iter = range(1000)#itertools.count()
        grad_steps = 0
        stop_count = 0
        loss = self.get_mse_losses(buff_val.get_all())
        sum_loss = loss
        #best = self.state_dict()
        best = [model.state_dict() for model in self.models]
        for epoch in epoch_iter:
            for batch_num in range(int(len(buff_train)/batch_size)):
                self.update_step(optim, buff_train.sample_tensors(n=batch_size))
                grad_steps += 1
            loss = self.get_mse_losses(buff_val.get_all())
            self.logger.info("Loss_mod:"+str((loss)))
            improved = False
            for i_model in range(self.n_models):
                if (sum_loss[i_model] - loss[i_model])/sum_loss[i_model] > 0.01:
                    sum_loss[i_model] = loss[i_model]
                    best[i_model] = self.models[i_model].state_dict()
                    improved = True
            if improved:
                stop_count = 0
            else:
                stop_count += 1
            if stop_count >= 5:
                break
        for model, best_model in zip(self.models, best):
            model.load_state_dict(best_model)
        self.elites = []
        for i in range(self.n_elites):
            idx = torch.argmin(sum_loss)
            self.elites.append(self.models[idx])
            sum_loss[idx] = 9999.
        # self.load_state_dict(best)
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
            n_models = len(self.elites)
            o_next_pred, r_pred = self.forward_elites(o, a)
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
        layers += [nn.Linear(input_dim, hidden_dims[0]), nn.ReLU()]
        for i in range(len(hidden_dims) - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.ReLU()]
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
