import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random
import numpy as np
from decentralizedlearning.algs.utils import OUNoise
from decentralizedlearning.algs.utils import ActorCritic
from decentralizedlearning.algs.utils import ReplayBuffer
from decentralizedlearning.algs.utils import loss_critic
from decentralizedlearning.algs.utils import Model

class HDDPGHyperPar:
    def __init__(self, **kwargs):
        self.hidden_dims_actor = tuple(kwargs.get("hidden_dims_actor",
                                             (128, 128)))
        self.hidden_dims_critic = tuple(kwargs.get("hidden_dims_critic",
                                              (128,)))
        self.hidden_dims_model = tuple(kwargs.get("hidden_dims_model",
                                             (256, 256)))
        self.use_OU = bool(kwargs.get("use_OU", False))
        self.gamma = float(kwargs.get("gamma", 0.99))
        self.tau = float(kwargs.get("tau", 0.005))
        self.delay = int(kwargs.get("delay", 2))
        self.lr_actor = float(kwargs.get("lr_actor", 0.001))
        self.lr_critic = float(kwargs.get("lr_critic", 0.001))
        self.lr_model = float(kwargs.get("lr_model", 0.0005))
        self.step_random = int(kwargs.get("step_random", 500))
        self.update_every_n_steps = int(kwargs.get("update_every_n_steps", 1))
        self.update_steps = int(kwargs.get("update_steps", 1))
        self.n_models = int(kwargs.get("n_models", 5))
        self.batch_size = int(kwargs.get("batch_size", 128))
        self.action_noise = float(kwargs.get("action_noise", 0.3))
        self.weight_decay = float(kwargs.get("weight_decay", 0.0))
        self.use_model = bool(kwargs.get("use_model", False))
        self.n_steps = int(kwargs.get("n_steps", 1))
        self.f_hyst = float(kwargs.get("f_hyst", 1.0))

class HDDPGAgent:
    def __init__(self, obs_dim, action_dim, hyperpar=None, **kwargs):
        # Initialize arguments
        if hyperpar:
            self.par = hyperpar
        else:
            self.par = HDDPGHyperPar(**kwargs)

        self.action_dim = action_dim

        self.ac = ActorCritic(obs_dim, action_dim, hidden_dims_actor=self.par.hidden_dims_actor, hidden_dims_critic=self.par.hidden_dims_critic)
        self.ac_target = copy.deepcopy(self.ac)
        for par in self.ac_target.parameters():
            par.requires_grad = False

        if self.par.use_OU:
            self.ou = OUNoise(action_dim)

        self.buffer = ReplayBuffer()
        self.o_old = None
        self.a_old = None

        self.optimizer_critic = torch.optim.Adam(self.ac.critic.parameters(), lr=self.par.lr_critic, amsgrad=True)
        self.optimizer_actor = torch.optim.Adam(self.ac.actor.parameters(), lr=self.par.lr_actor, amsgrad=True)

        # Initialize models
        if self.par.use_model:
            self.models = []
            self.optimizer_models = []
            for k in range(self.par.n_models):
                model = Model(obs_dim + action_dim, self.par.hidden_dims_model, obs_dim)
                self.models.append(model)
                self.optimizer_models.append(torch.optim.Adam(model.parameters(),
                                                              lr=self.par.lr_model, amsgrad=True))

    def reset(self):
        self.o_old = None
        self.a_old = None
        if self.par.use_OU:
            self.ou.reset()

    def step(self, o, r, eval=False, done=False):
        o = torch.Tensor(-1.*o)
        r = torch.Tensor(np.array(float(r)))
        done = torch.Tensor(np.array(float(done)))
        if eval:
            action = self.ac.actor(o.unsqueeze(0)).squeeze()
            action = torch.clamp(action, 0., 1.0)
            return action.detach().numpy()
        if self.o_old is not None:
            self.buffer.add((self.o_old, self.a_old, r, o, done))

        if self.buffer.len() > self.buffer.n_samples:
            for i in range(self.par.n_steps):
                if self.par.use_model:
                    self.update_models()

                self.update_step()

        # Select Action
        action = self.select_action(o, "noisy")

        self.o_old = o

        if action.size() == torch.Size([]):
            self.a_old = action.unsqueeze(0)
        else:
            self.a_old = action
        return action.detach().numpy()

    def update_step(self):
        # Sample Minibatch
        if not self.par.use_model:
            b = self.buffer.sample_tensors(n=self.par.batch_size)
        else:
            model = random.choice(self.models)
            b = self.sample_from_model(model)

        # Update Critic
        self.update_critic(b)
        # Update Actor
        self.update_actor(b)
        # Update Target Networks
        self.update_target_networks()

    def update_target_networks(self):
        with torch.no_grad():
            for par, par_target in zip(self.ac.parameters(), self.ac_target.parameters()):
                par_target.data.copy_((1 - self.par.tau) * par_target + self.par.tau * par.data)

    def update_actor(self, b):
        for par in self.ac.critic.parameters():
            par.requires_grad = False
        self.optimizer_actor.zero_grad()
        loss_actor = -torch.mean(self.ac.critic(b["o"], self.ac.actor(b["o"])))
        loss_actor.backward()
        self.optimizer_actor.step()
        for par in self.ac.critic.parameters():
            par.requires_grad = True

    def update_critic(self, b):
        self.optimizer_critic.zero_grad()
        with torch.no_grad():
            y = b["r"].unsqueeze(-1) + (1 - b["done"]) * self.par.gamma * self.ac_target.critic(b["o_next"],
                                                                                                self.ac_target.actor(
                                                                                                    b["o_next"]))
            y2 = b["r"].unsqueeze(-1) + (1 - b["done"]) * self.par.gamma * self.ac_target.critic(b["o_next"],
                                                                                            self.ac_target.actor(
                                                                                                b["o_next"])+0.05)
            y3 = b["r"].unsqueeze(-1) + (1 - b["done"]) * self.par.gamma * self.ac_target.critic(b["o_next"],
                                                                                            self.ac_target.actor(
                                                                                                b["o_next"])-0.05)
        # print(torch.mean(y3-y2))
        loss_c = loss_critic(self.ac.critic(b["o"], b["a"]), y, f_hyst=self.par.f_hyst)
        loss_c.backward()
        self.optimizer_critic.step()

    def select_action(self, o, method):
        assert method in ["random", "noisy", "greedy"], "Invalid action selection method"
        if method == "random":
            return torch.rand(self.action_dim)

        with torch.no_grad():
            action = self.ac.actor(o.unsqueeze(0)).squeeze()
            if method == "noisy":
                if self.par.use_OU:
                    action = action + torch.Tensor(self.ou.noise())[0]
                else:
                    action = action + torch.randn(action.size()) * 0.3
                action = torch.clamp(action, 0., 1.0)
        return action

    def update_models(self):
        samples = self.buffer.sample_tensors(n=self.par.batch_size)
        for optim, model in zip(self.optimizer_models, self.models):
            self.model_step(model, optim, samples)

    def sample_from_model(self, model):
        # Sample Minibatch
        b = self.buffer.sample_tensors(n=self.par.batch_size)
        with torch.no_grad():
            action = self.ac.actor(b["o"])
            if self.par.use_OU:
                action_noisy = action + torch.Tensor(self.ou.noise())[0]
            else:
                action_noisy = action + torch.randn(action.size()) * 0.3
            b["a"] = torch.clamp(action_noisy, 0., 1.0)
        new_o, r = model.sample(b["o"], b["a"])
        b["o_next"] = new_o
        b["r"] = r.squeeze()
        return b

    def model_step(self, model, optim, samples):
        o_next_pred, r_pred = model(samples["o"], samples["a"])
        sigma = o_next_pred[1]
        sigma_2 = r_pred[1]
        mu = o_next_pred[0]
        target = samples["o_next"]
        loss1 = torch.mean((mu - target) / sigma ** 2 * (mu - target))
        loss3 = torch.mean(torch.log(torch.prod(sigma ** 2, 1) * torch.prod(sigma_2 ** 2, 1)))
        mu = r_pred[0]
        target = samples["r"].unsqueeze(1)
        loss2 = torch.mean((mu - target) / sigma_2 ** 2 * (mu - target))
        loss = loss1 + loss2 + loss3
        optim.zero_grad()
        loss.backward()
        optim.step()
