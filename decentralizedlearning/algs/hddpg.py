import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random
import numpy as np
from decentralizedlearning.algs.utils import OUNoise
from decentralizedlearning.algs.utils import ActorCritic
from decentralizedlearning.algs.utils import ReplayBuffer


class HDDPGAgent:
    def __init__(self, obs_dim, action_dim, use_OU=False):
        self.ac = ActorCritic(obs_dim, action_dim)
        self.ac_target = copy.deepcopy(self.ac)
        for par in self.ac_target.parameters():
            par.requires_grad = False
        self.use_OU = False
        if use_OU:
            self.use_OU = True
            self.ou = OUNoise(action_dim)

        self.buffer = ReplayBuffer()
        self.o_old = None
        self.a_old = None
        self.gamma = 0.95
        self.tau = 0.001
        self.f_hyst = 1.0

        self.optimizer_critic = torch.optim.Adam(self.ac.critic.parameters(), lr=0.001)
        self.optimizer_actor = torch.optim.Adam(self.ac.actor.parameters(), lr=0.001)

    def reset(self):
        self.o_old = None
        self.a_old = None
        if self.use_OU:
            self.ou.reset()

    def loss_critic(self, val, target):
        diffs = target - val
        diffs[diffs < 0] *= self.f_hyst
        # if random.random() < 0.01:
        #     print(torch.mean(target))
        #     print(torch.mean(diffs**2))
        return torch.mean(diffs**2)

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
                    y = b["r"].unsqueeze(-1) + (1-b["done"])*self.gamma * self.ac_target.critic(b["o_next"],self.ac_target.actor(b["o_next"]))
                loss_critic = self.loss_critic(self.ac.critic(b["o"], b["a"]), y)

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
                if self.use_OU:
                    action_noisy = action + torch.Tensor(self.ou.noise())[0]
                else:
                    action_noisy = action + torch.randn(action.size())*0.3
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

