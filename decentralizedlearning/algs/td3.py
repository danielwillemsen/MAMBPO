import torch
import copy
import numpy as np

from decentralizedlearning.algs.utils import ReplayBuffer

from decentralizedlearning.algs.utils import Critic
from decentralizedlearning.algs.utils import Actor
from decentralizedlearning.algs.utils import OUNoise

class TD3HyperPar:
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
        self.step_random = int(kwargs.get("step_random", 50))
        self.update_every_n_steps = int(kwargs.get("update_every_n_steps", 50))
        self.update_steps = int(kwargs.get("update_steps", 200))
        self.n_models = int(kwargs.get("n_models", 10))
        self.batch_size = int(kwargs.get("batch_size", 128))
        self.action_noise = float(kwargs.get("action_noise", 0.3))
        self.weight_decay = float(kwargs.get("weight_decay", 0.0))

        self.f_hyst = float(kwargs.get("f_hyst", 1.0))

class TD3:
    def __init__(self, obs_dim, action_dim, hyperpar=None, **kwargs):
        # Initialize arguments
        if hyperpar:
            self.par = hyperpar
        else:
            self.par = TD3HyperPar(**kwargs)

        # Initialize actor
        self.actor = Actor(obs_dim, self.par.hidden_dims_actor,  action_dim)
        self.actor_target = copy.deepcopy(self.actor)
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.par.lr_actor,
                                                weight_decay=self.par.weight_decay)

        for par in self.actor_target.parameters():
            par.requires_grad = False

        # Initialize 2 critics
        self.critics = []
        self.critics_target = []
        self.optimizer_critics = []
        for k in range(2):
            critic = Critic(obs_dim + action_dim, self.par.hidden_dims_critic)
            self.critics.append(critic)
            self.critics_target.append(copy.deepcopy(critic))
            self.optimizer_critics.append(torch.optim.Adam(critic.parameters(),
                                                           lr=self.par.lr_critic,
                                                           weight_decay=self.par.weight_decay))
            for par in self.critics_target[k].parameters():
                par.requires_grad = False

        # Initialize noise
        if self.par.use_OU:
            self.ou = OUNoise(action_dim)

        self.buffer = ReplayBuffer()
        self.o_old = None
        self.a_old = None

        self.step_i = 0

    def reset(self):
        self.o_old = None
        self.a_old = None
        if self.par.use_OU:
            self.ou.reset()

    def loss_critic(self, val, target):
        diffs = target - val
        diffs[diffs < 0] *= self.par.f_hyst
        return torch.mean(diffs**2)

    def step(self, o, r, eval=False, done=False):
        o = torch.Tensor(o)
        r = torch.Tensor(np.array(float(r)))
        done = torch.Tensor(np.array(float(done)))
        if not eval:
            if self.o_old is not None:
                self.buffer.add((self.o_old, self.a_old, r, o, done))

            if self.buffer.len() > self.par.batch_size:
                # Sample Minibatch
                b = self.buffer.sample_tensors(n=self.par.batch_size)

                # Update Critic
                for optimizer in self.optimizer_critics:
                    optimizer.zero_grad()
                with torch.no_grad():
                    a_target = self.actor_target(b["o_next"])
                    a_target = torch.clamp(
                        a_target + torch.clamp(torch.randn(a_target.size())*0.0, -0.5, 0.5), 0., 1.)
                    y = b["r"].unsqueeze(-1) + (1-b["done"])*self.par.gamma * torch.min(
                        *[critic_target(b["o_next"], a_target) for critic_target in self.critics_target])
                loss_critics = [self.loss_critic(critic(b["o"], b["a"]), y)
                                for critic in self.critics]

                if(np.random.random() < 0.0025):
                    print("Loss:", loss_critics)
                    print("Mean Q:", torch.mean(y))

                for loss in loss_critics:
                    loss.backward()

                for optimizer in self.optimizer_critics:
                    optimizer.step()

                # Update Actor
                if self.step_i % self.par.delay == 0:
                    for par in self.critics[0].parameters():
                        par.requires_grad = False

                    self.optimizer_actor.zero_grad()
                    loss_actor = - \
                        torch.mean(self.critics[0](b["o"], self.actor(b["o"])))
                    loss_actor.backward()
                    self.optimizer_actor.step()

                    for par in self.critics[0].parameters():
                        par.requires_grad = True

                    # Update Target Networks
                    with torch.no_grad():
                        for par, par_target in zip(self.actor.parameters(), self.actor_target.parameters()):
                            par_target.data.copy_(
                                (1-self.par.tau) * par_target + self.par.tau * par.data)
                        for k in range(2):
                            for par, par_target in zip(self.critics[k].parameters(), self.critics_target[k].parameters()):

                                par_target.data.copy_(
                                    (1-self.par.tau) * par_target + self.par.tau * par.data)
            # Select Action
            with torch.no_grad():
                action = self.actor(o.unsqueeze(0)).squeeze()
                if self.par.use_OU:
                    action_noisy = action + torch.Tensor(self.ou.noise())[0]
                else:
                    action_noisy = action + torch.randn(action.size())*self.par.action_noise
                action = torch.clamp(action_noisy, 0., 1.0)

            self.o_old = o
            if action.size() == torch.Size([]):
                self.a_old = action.unsqueeze(0)
            else:
                self.a_old = action
        else:
            action = self.actor(o.unsqueeze(0)).squeeze()
            action = torch.clamp(action, 0., 1.0)
        self.step_i += 1
        return action.detach().numpy()