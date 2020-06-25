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

class HDDPGHyperPar:
    def __init__(self, **kwargs):
        self.hidden_dims_actor = tuple(kwargs.get("hidden_dims_actor",
                                             (256, 256)))
        self.hidden_dims_critic = tuple(kwargs.get("hidden_dims_critic",
                                              (256, 256)))
        self.hidden_dims_model = tuple(kwargs.get("hidden_dims_model",
                                             (256, 256)))
        self.use_OU = bool(kwargs.get("use_OU", False))
        self.gamma = float(kwargs.get("gamma", 0.99))
        self.tau = float(kwargs.get("tau", 0.005))
        self.delay = int(kwargs.get("delay", 2))
        self.lr_actor = float(kwargs.get("lr_actor", 0.001))
        self.lr_critic = float(kwargs.get("lr_critic", 0.001))
        self.lr_model = float(kwargs.get("lr_model", 0.001))
        self.step_random = int(kwargs.get("step_random", 500))
        self.update_every_n_steps = int(kwargs.get("update_every_n_steps", 1))
        self.update_steps = int(kwargs.get("update_steps", 1))
        self.n_models = int(kwargs.get("n_models", 10))
        self.batch_size = int(kwargs.get("batch_size", 128))
        self.action_noise = float(kwargs.get("action_noise", 0.3))
        self.weight_decay = float(kwargs.get("weight_decay", 0.0))

        self.f_hyst = float(kwargs.get("f_hyst", 1.0))

class HDDPGAgent:
    def __init__(self, obs_dim, action_dim, hyperpar=None, **kwargs):
        # Initialize arguments
        if hyperpar:
            self.par = hyperpar
        else:
            self.par = HDDPGHyperPar(**kwargs)

        self.ac = ActorCritic(obs_dim, action_dim, hidden_dims_actor=self.par.hidden_dims_actor, hidden_dims_critic=self.par.hidden_dims_critic)
        self.ac_target = copy.deepcopy(self.ac)
        for par in self.ac_target.parameters():
            par.requires_grad = False

        if self.par.use_OU:
            self.ou = OUNoise(action_dim)

        self.buffer = ReplayBuffer()
        self.o_old = None
        self.a_old = None

        self.optimizer_critic = torch.optim.Adam(self.ac.critic.parameters(), lr=self.par.lr_critic)
        self.optimizer_actor = torch.optim.Adam(self.ac.actor.parameters(), lr=self.par.lr_actor)

    def reset(self):
        self.o_old = None
        self.a_old = None
        if self.par.use_OU:
            self.ou.reset()

    def step(self, o, r, eval=False, done=False):
        o = torch.Tensor(o)
        r = torch.Tensor(np.array(float(r)))
        done = torch.Tensor(np.array(float(done)))
        #print(o)
        #print("R: " + str(r))
        if not eval:
            if self.o_old is not None:
                self.buffer.add((self.o_old, self.a_old, r, o, done))

            if self.buffer.len() > self.buffer.n_samples:
                # Sample Minibatch
                b = self.buffer.sample_tensors()

                # Update Critic
                self.optimizer_critic.zero_grad()
                with torch.no_grad():
                    y = b["r"].unsqueeze(-1) + (1-b["done"])*self.par.gamma * self.ac_target.critic(b["o_next"],self.ac_target.actor(b["o_next"]))
                loss_c = loss_critic(self.ac.critic(b["o"], b["a"]), y, f_hyst=self.par.f_hyst)

                loss_c.backward()
                self.optimizer_critic.step()

                # Update Actor
                for par in self.ac.critic.parameters():
                    par.requires_grad = False

                self.optimizer_actor.zero_grad()
                loss_actor = -torch.mean(self.ac.critic(b["o"],self.ac.actor(b["o"])))

                loss_actor.backward()

                self.optimizer_actor.step()

                for par in self.ac.critic.parameters():
                    par.requires_grad = True

                # Update Target Networks
                with torch.no_grad():
                    for par, par_target in zip(self.ac.parameters(), self.ac_target.parameters()):
                        par_target.data.copy_((1-self.par.tau) * par_target + self.par.tau * par.data)

            # Select Action
            with torch.no_grad():
                action = self.ac.actor(o.unsqueeze(0)).squeeze()
                if self.par.use_OU:
                    action_noisy = action + torch.Tensor(self.ou.noise())[0]
                else:
                    action_noisy = action + torch.randn(action.size())*self.par.action_noise
                action = torch.clamp(action_noisy,0., 1.0)

            self.o_old = o
            if action.size() == torch.Size([]):
                self.a_old = action.unsqueeze(0)
            else:
                self.a_old = action
        else:
            action = self.ac.actor(o.unsqueeze(0)).squeeze()
            action = torch.clamp(action, 0., 1.0)
        return action.detach().numpy()

