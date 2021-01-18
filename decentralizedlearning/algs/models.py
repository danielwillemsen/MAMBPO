import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import random
import logging
from decentralizedlearning.algs.utils import scale_action

class EnsembleModel(nn.Module):
    """ Ensemble of models as used in the MAMBPO algorithm.

    """
    def __init__(self, input_dim, hidden_dims, obs_dim, n_models, monitor_losses=False, use_stochastic=True, name="cheetah", device=None):
        super().__init__()
        self.device = device
        self.monitor_losses = monitor_losses
        self.use_stochastic = use_stochastic
        self.name = name
        self.n_models = n_models
        self.n_elites = 10
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
        target_r = samples["r"].unsqueeze(-1)#.unsqueeze(0).repeat(10,1,1)

        log_var_o = o_next_pred[1]
        log_var_r = r_pred[1]
        inv_var_o = torch.exp(-log_var_o)
        inv_var_r = torch.exp(-log_var_r)
        mu_o = o_next_pred[0]
        mu_r = r_pred[0]

        if np.random.rand()<0.01:
            vx = target_r - torch.mean(target_r)
            vy = mu_r[0] - torch.mean(mu_r[0])
            l1 = torch.mean(torch.cat((((mu_o - target_o) * (mu_o - target_o)),
                                       ((mu_r - target_r) * (mu_r - target_r))), dim=-1),
                            dim=(-1, -2))
            print("R corr_train", torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))))
            print("Loss train", l1)

        if self.use_stochastic:
            l1 = torch.sum(torch.mean(torch.cat((((mu_o - target_o) * inv_var_o * (mu_o - target_o)),1*((mu_r - target_r) * inv_var_r * (mu_r - target_r))), dim=-1),dim=(-1,-2)))
            l2 = torch.sum(torch.mean(torch.cat((log_var_o, 1*log_var_r), dim=-1),dim=(-1,-2)))
        else:
            l1 = torch.sum(torch.mean(torch.cat((((mu_o - target_o) * (mu_o - target_o)),1*((mu_r - target_r) * (mu_r - target_r))), dim=-1),dim=(-1,-2)))
            l2 = 0.#torch.sum(torch.mean(torch.cat((log_var_o, 1*log_var_r), dim=-1),dim=(-1,-2)))

#            l1 = torch.mean((mu_o - target_o) * (mu_o - target_o))
#            l2 = torch.mean((mu_r - target_r) * (mu_r - target_r))
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
            vx = target_r - torch.mean(target_r)
            vy = mu_r[0] - torch.mean(mu_r[0])
            print("R corr", torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))))
            l1 = torch.mean(torch.cat((((mu_o - target_o) * (mu_o - target_o)),
                                                 ((mu_r - target_r) * (mu_r - target_r))), dim=-1),
                                      dim=(-1, -2))

        return l1

    def step_single(self, observation, action, **kwargs):
        if type(observation) == list:
            o = torch.tensor(np.concatenate(observation,axis=0), device=self.device, dtype=torch.float)
        else:
            o = torch.tensor(observation, device=self.device, dtype=torch.float)
        if type(action) == list:
            a = torch.tensor(np.concatenate(action,axis=0), device=self.device, dtype=torch.float)
        else:
            a = torch.tensor(action, device=self.device, dtype=torch.float)
        o_next_pred, r_pred = self.forward_elites(o, a)
        mu_o = o_next_pred[0]
        mu_r = r_pred[0]
        log_var_o = o_next_pred[1]
        log_var_r = r_pred[1]
        new_o = torch.normal(mu_o, torch.exp(0.5 * log_var_o))
        r = torch.normal(mu_r, torch.exp(0.5 * log_var_r))

        model_i = random.randint(0, self.n_elites-1)
        new_o = new_o[model_i]
        r = r[model_i]

        return new_o.detach().cpu().numpy(), r.detach().cpu().numpy()



    #
    # def train_models(self, optim, buffer, holdout=0.2, multistep_buffer=None):
    #     batch_size = 256
    #     max_epochs = 99
    #     buff_train, buff_val = buffer.get_buffer_split(holdout=holdout)
    #     epoch_iter = range(max_epochs)#itertools.count()
    #     grad_steps = 0
    #     stop_count = 0
    #     loss = self.get_mse_losses(buff_val.get_all())
    #     sum_loss = loss
    #     #best = self.state_dict()
    #     best = [model.state_dict() for model in self.models]
    #     for epoch in epoch_iter:
    #         loss = self.get_mse_losses(buff_train.get_all())
    #         self.logger.info("Loss_mod_train:"+str((loss)))
    #         self.train()
    #         for batch_num in range(int(len(buff_train)/batch_size)):
    #             self.update_step(optim, buff_train.sample_tensors(n=batch_size))
    #             if multistep_buffer:
    #                 self.multistep_update_step(optim,
    #                                            multistep_buffer.sample_tensors(n=batch_size),
    #                                            multistep_buffer.length)
    #             grad_steps += 1
    #         self.eval()
    #         loss = self.get_mse_losses(buff_val.get_all())
    #         self.logger.info("Loss_mod:"+str((loss)))
    #         improved = False
    #         for i_model in range(self.n_models):
    #             if (sum_loss[i_model] - loss[i_model])/sum_loss[i_model] > 0.01:
    #                 sum_loss[i_model] = loss[i_model]
    #                 best[i_model] = self.models[i_model].state_dict()
    #                 improved = True
    #         if improved:
    #             stop_count = 0
    #         else:
    #             stop_count += 1
    #         if stop_count >= 5:
    #             break
    #     for model, best_model in zip(self.models, best):
    #         model.load_state_dict(best_model)
    #     self.elites = []
    #     for i in range(self.n_elites):
    #         idx = torch.argmin(sum_loss)
    #         self.elites.append(self.models[idx])
    #         sum_loss[idx] = 9999.
    #     # self.load_state_dict(best)
    #     self.logger.info("Stopped. Epoch:"+ str(epoch)+ "Grad_steps:" +str(grad_steps))
    #     return
    def train_models(self, optim, buffer, **kwargs):
        """ Method to train models

        :param optim: optimizer (usually adam)
        :param buffer: replay buffer on which to train (B_env)
        :param kwargs:
        :return:
        """
        batch_size = 512
        n_steps_model = 500
        for step in range(n_steps_model):
            self.update_step(optim, buffer.sample_tensors(n=batch_size))
        self.logger.info("Stopped.")

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
        return ret_samples

    def generate_efficient(self, samples, actor, diverse=True, batch_size=128, rollout_length=1, obs_i_per_agent=None, act_i_per_agent=None):
        """ Generate a batch of model samples from a batch of real environment samples.

        :param samples: input transitions
        :param actor: (list of) actors that are used to select actions
        :param diverse: used for debugging. Whether or not the actors select actions or that the actions remain default.
        :param batch_size:
        :param rollout_length: amount of steps to perform a rollout. In the thesis 1 is used.
        :param obs_i_per_agent:
        :param act_i_per_agent:
        :return:
        """
        ret_samples = []
        with torch.no_grad():
            for i in range(rollout_length):
                if i == 0:
                    o = samples["o"]
                else:
                    o = new_o_ret
                if not diverse:
                    a = samples["a"]
                else:
                    if type(actor) is not list:
                        a = actor(o, greedy=False)
                    else:
                        a = samples["a"].clone()
                        for i, actor_i in enumerate(actor):
                            a_i = actor_i(o[:,obs_i_per_agent[i]:obs_i_per_agent[i+1]])
                            a[:, act_i_per_agent[i]:act_i_per_agent[i + 1]] = a_i
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
                else:
                    # If not no stochastic networks are used, we do this hacky stuff to get a nearly deterministic result.
                    new_o = torch.normal(mu_o, 0.00001)
                    r = torch.normal(mu_r, 0.00001)
                new_o_ret = new_o[randomlist, idx, :]
                done = samples["done"]
                ret_samples.append((o, a, r[randomlist, idx, :].squeeze(-1), new_o_ret, done))
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
        self.MAX_LOG_VAR = torch.tensor(-2, dtype=torch.float32) # 0.5 was -2.
        self.MIN_LOG_VAR = torch.tensor(-5., dtype=torch.float32) #-10 was -5.
        layers = []
        #layers += [nn.Linear(input_dim, hidden_dims[0]), nn.BatchNorm1d(hidden_dims[0]), nn.LeakyReLU(), ]
        layers += [nn.Linear(input_dim, hidden_dims[0]), nn.LeakyReLU(), ]
        for i in range(len(hidden_dims) - 1):
            #layers += [nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.BatchNorm1d(hidden_dims[i+1]) , nn.LeakyReLU()]
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.LeakyReLU()]
        self.net = nn.Sequential(*layers)
        self.mu_output = nn.Linear(hidden_dims[-1], obs_dim)
        self.mu_reward = nn.Linear(hidden_dims[-1], 1)

        self.var_output = nn.Linear(hidden_dims[-1], obs_dim)
        self.var_reward = nn.Linear(hidden_dims[-1], 1)

    def forward(self, input):
        x = self.net(input)
        if True:
            log_var_output = self.var_output(x)
            log_var_reward = self.var_reward(x)

            log_var_output = self.MAX_LOG_VAR - F.softplus(self.MAX_LOG_VAR - log_var_output)
            log_var_reward = self.MAX_LOG_VAR - F.softplus(self.MAX_LOG_VAR - log_var_reward)

            log_var_output = self.MIN_LOG_VAR + F.softplus(log_var_output - self.MIN_LOG_VAR)
            log_var_reward = self.MIN_LOG_VAR + F.softplus(log_var_reward - self.MIN_LOG_VAR)
            if not self.use_stochastic:
                log_var_reward *= 0.
                log_var_output *= 0.

            return [self.mu_output(x), log_var_output], [self.mu_reward(x), log_var_reward]
