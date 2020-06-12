import torch
import copy
import numpy as np

from decentralizedlearning.algs.utils import ReplayBuffer

from decentralizedlearning.algs.utils import Critic
from decentralizedlearning.algs.utils import Actor
from decentralizedlearning.algs.utils import OUNoise


class TD3:
    def __init__(self, obs_dim, action_dim, *args, **kwargs):
        # Initialize arguments
        hidden_dims_actor = tuple(kwargs.get("hidden_dims_actor",
                                             (256, 256)))
        hidden_dims_critic = tuple(kwargs.get("hidden_dims_critic",
                                              (256, 256)))

        self.use_OU = bool(kwargs.get("use_OU", False))

        self.gamma = 0.99
        self.tau = 0.005
        self.f_hyst = 1.0
        self.delay = 2
        lr_actor = 0.001
        lr_critic = 0.001
        # Initialize actor
        self.actor = Actor(obs_dim, hidden_dims_actor,  action_dim)
        self.actor_target = copy.deepcopy(self.actor)
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(),
                                                lr=lr_actor)

        for par in self.actor_target.parameters():
            par.requires_grad = False

        # Initialize 2 critics
        self.critics = []
        self.critics_target = []
        self.optimizer_critics = []
        for k in range(2):
            critic = Critic(obs_dim + action_dim, hidden_dims_critic)
            self.critics.append(critic)
            self.critics_target.append(copy.deepcopy(critic))
            self.optimizer_critics.append(torch.optim.Adam(critic.parameters(),
                                                           lr=lr_critic))

            for par in self.critics_target[k].parameters():
                par.requires_grad = False

        # Initialize noise
        if self.use_OU:
            self.ou = OUNoise(action_dim)

        self.buffer = ReplayBuffer()
        self.o_old = None
        self.a_old = None

        self.step_i = 0

    def reset(self):
        self.o_old = None
        self.a_old = None
        if self.use_OU:
            self.ou.reset()

    def loss_critic(self, val, target):
        diffs = target - val
        diffs[diffs < 0] *= self.f_hyst
        return torch.mean(diffs**2)

    def step(self, o, r, eval=False, done=False):
        o = torch.Tensor(o)
        r = torch.Tensor(np.array(float(r)))
        done = torch.Tensor(np.array(float(done)))
        if not eval:
            if self.o_old is not None:
                self.buffer.add((self.o_old, self.a_old, r, o, done))

            if self.buffer.len() > self.buffer.n_samples:
                # Sample Minibatch
                b = self.buffer.sample_tensors()

                # Update Critic
                for optimizer in self.optimizer_critics:
                    optimizer.zero_grad()
                with torch.no_grad():
                    a_target = self.actor_target(b["o_next"])
                    a_target = torch.clamp(
                        a_target + torch.clamp(torch.randn(a_target.size())*0.1, -0.5, 0.5), 0., 1.)
                    y = b["r"].unsqueeze(-1) + (1-b["done"])*self.gamma * torch.min(
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
                if self.step_i % self.delay == 0:
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
                                (1-self.tau) * par_target + self.tau * par.data)
                        for k in range(2):
                            for par, par_target in zip(self.critics[k].parameters(), self.critics_target[k].parameters()):

                                par_target.data.copy_(
                                    (1-self.tau) * par_target + self.tau * par.data)
            # Select Action
            with torch.no_grad():
                action = self.actor(o.unsqueeze(0)).squeeze()
                if self.use_OU:
                    action_noisy = action + torch.Tensor(self.ou.noise())[0]
                else:
                    action_noisy = action + torch.randn(action.size())*0.3
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
