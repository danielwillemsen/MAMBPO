import torch
import copy
import numpy as np
import time

from decentralizedlearning.algs.utils import EfficientReplayBuffer as ReplayBuffer
from decentralizedlearning.algs.utils import MultiStepReplayBuffer
from decentralizedlearning.algs.utils import StateBuffer

from decentralizedlearning.algs.utils import Critic
from decentralizedlearning.algs.utils import StochActor
from decentralizedlearning.algs.utils import OUNoise
from decentralizedlearning.algs.utils import loss_critic
from decentralizedlearning.algs.utils import update_target_networks
from decentralizedlearning.algs.utils import convert_inputs_to_tensors
from decentralizedlearning.algs.models import EnsembleModel
from decentralizedlearning.algs.models import DegradedSim

import logging


class SACHyperPar:
    def __init__(self, **kwargs):
        self.hidden_dims_actor = tuple(kwargs.get("hidden_dims_actor",
                                             (256,256)))
        self.hidden_dims_critic = tuple(kwargs.get("hidden_dims_critic",
                                              (256,256)))
        self.hidden_dims_model = tuple(kwargs.get("hidden_dims_model",
                                             (200,200,200,200)))
        self.use_OU = bool(kwargs.get("use_OU", False))
        self.gamma = float(kwargs.get("gamma", 0.99))
        self.tau = float(kwargs.get("tau", 0.005))
        self.delay = int(kwargs.get("delay", 2))
        self.lr_actor = float(kwargs.get("lr_actor", 0.0003))
        self.lr_critic = float(kwargs.get("lr_critic", 0.0003))
        self.lr_model = float(kwargs.get("lr_model", 0.001))
        self.l2_norm = float(kwargs.get("l2_norm", 0.0))

        self.step_random = int(kwargs.get("step_random", 1000))
        self.update_every_n_steps = int(kwargs.get("update_every_n_steps", 1))
        self.update_model_every_n_steps = int(kwargs.get("update_model_every_n_steps",250))
        self.n_steps = int(kwargs.get("n_steps", 1))
        self.n_models = int(kwargs.get("n_models", 10))
        self.batch_size = int(kwargs.get("batch_size", 256))
        self.weight_decay = float(kwargs.get("weight_decay", 0.0000))
        self.alpha = float(kwargs.get("alpha",0.001))
        self.f_hyst = float(kwargs.get("f_hyst", 1.0))
        self.use_model = bool(kwargs.get("use_model", False))
        self.monitor_losses = bool(kwargs.get("monitor_losses", True))
        self.use_model_stochastic = bool(kwargs.get("use_model_stochastic", True))
        self.diverse = bool(kwargs.get("diverse", True))
        self.autotune = bool(kwargs.get("autotune", False))
        self.target_entropy = float(kwargs.get("target_entropy", -0.05))
        self.rollout_length = int(kwargs.get("rollout_length", 1))
        self.use_rollout_schedule = bool(kwargs.get("use_rollout_schedule", False))
        self.rollout_schedule_time = list(kwargs.get("rollout_schedule_time", [0,1000]))
        self.rollout_schedule_val = list(kwargs.get("rollout_schedule_val", [1,10]))
        self.real_ratio = float(kwargs.get("real_ratio", 0.05))
        self.use_multistep_reg = bool(kwargs.get("use_multistep_reg", False))
        self.use_degraded_sim = bool(kwargs.get("use_degraded_sim", False))
        self.name = str(kwargs.get("name", "cheetah"))
        self.use_common_actor = bool(kwargs.get("use_common_actor", False))
        self.use_common_critic = bool(kwargs.get("use_common_critic", False))

class SAC:
    def __init__(self, obs_dim, action_dim, hyperpar=None, **kwargs):
        # Initialize arguments
        self.logger = logging.getLogger('root')

        if torch.cuda.is_available():
            print("Using CUDA")
            self.device = torch.device("cuda:0")
        else:
            print("No CUDA found")
            self.device = torch.device("cpu")

        # Initialize arguments
        self.use_min = kwargs.get("use_min", True)
        self.use_correct = kwargs.get("use_correct", True)
        if hyperpar:
            self.par = hyperpar
        else:
            self.par = SACHyperPar(**kwargs)

        self.logger.info(self.par.__dict__)
        self.action_dim = action_dim

        # Initialize actor
        self.actor = StochActor(obs_dim, self.par.hidden_dims_actor,  action_dim, device=self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor.to(self.device)
        self.actor_target.to(self.device)

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.par.lr_actor,
                                                weight_decay=self.par.weight_decay)

        for par in self.actor_target.parameters():
            par.requires_grad = False

        if self.par.autotune:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.optimizer_alpha = torch.optim.Adam([self.log_alpha],
                                                    lr=self.par.lr_actor)#self.par.lr_actor)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = self.par.alpha

        if self.par.use_model:
            self.model = EnsembleModel(obs_dim + action_dim,
                                       self.par.hidden_dims_model,
                                       obs_dim,
                                       self.par.n_models,
                                       monitor_losses=self.par.monitor_losses,
                                       use_stochastic=self.par.use_model_stochastic,
                                       name=self.par.name,
                                       device=self.device).to(self.device)
            self.optimizer_model = torch.optim.Adam(self.model.parameters(),
                                                    lr=self.par.lr_model,
                                                    amsgrad=True,
                                                    weight_decay=self.par.l2_norm)
        else:
            self.model = None

        self.real_buffer = ReplayBuffer(size=100000, batch_size=self.par.batch_size, device=self.device)
        if not self.par.use_degraded_sim:
            self.model_sample_buffer = self.real_buffer
        else:
            env_copy = kwargs.get("env_copy", None)
            self.model = DegradedSim(env_copy, degradation=0.0, bias=0.4, device=self.device)
            self.model_sample_buffer = StateBuffer(device=self.device)
        # Initialize 2 critics
        self.critics = []
        self.critics_target = []
        self.optimizer_critics = []
        for k in range(2):
            critic = Critic(obs_dim + action_dim, self.par.hidden_dims_critic)
            critic.to(self.device)
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

        if self.par.use_multistep_reg:
            self.multistep_buffer = MultiStepReplayBuffer(size=100000, batch_size=self.par.batch_size,
                                                          device=self.device, length=5)
        else:
            self.multistep_buffer = None

        if self.par.monitor_losses:
            self.val_buffer = ReplayBuffer(batch_size=500, device=self.device)
        if self.par.use_model:
            self.fake_buffer = ReplayBuffer(size=self.par.n_steps * self.par.update_every_n_steps * self.par.batch_size * 8 * self.par.rollout_length, batch_size=self.par.batch_size, device=self.device)
            self.ac_buffer = self.fake_buffer
        else:
            self.ac_buffer = self.real_buffer

        self.o_old = None
        self.a_old = None
        self.s_old = None
        self.reset()


        self.step_i = 0

    def reset(self):
        self.o_old = None
        self.a_old = None
        self.traj_o = []
        self.traj_r = []
        self.traj_a = []
        self.traj_done = []

        if self.par.use_OU:
            self.ou.reset()

    def step(self, o, r, eval=False, done=False, generate_val_data=False, greedy_eval=True, s=None):
        o, r, done = convert_inputs_to_tensors(o, r, done, self.device)
        # self.logger.info(done)
        if eval:
            # Select greedy action and return
            if greedy_eval:
                action = self.actor.select_action(o, "greedy")
            else:
                action = self.actor.select_action(o, "noisy")
            return action.detach().cpu().numpy()

        if generate_val_data:
            # Select greedy action and return
            action = self.actor.select_action(o, "noisy")
            if self.o_old is not None:
                self.val_buffer.add((self.o_old, self.a_old, r, o, done))
            self.o_old = o
            if action.size() == torch.Size([]):
                self.a_old = action.unsqueeze(0)
            else:
                self.a_old = action
            return action.detach().cpu().numpy()

        # Do training process step
        self.traj_o.append(o)
        self.traj_done.append(done)
        self.traj_r.append(r)

        if self.o_old is not None:
            self.real_buffer.add((self.o_old, self.a_old, r, o, done))

        if self.par.use_degraded_sim and self.s_old is not None:
            self.model_sample_buffer.add((self.o_old, self.a_old, r, o, done), self.s_old)

        if self.par.use_multistep_reg:
            if len(self.traj_o)>self.multistep_buffer.length:
                self.multistep_buffer.add_mini_traj(self.traj_o[-6:], self.traj_a[-5:], self.traj_r[-5:], self.traj_done[-5:])

        if self.step_i % self.par.update_model_every_n_steps == 0:
            # Update model and generate new samples:
            if self.par.use_model and self.real_buffer.len() > self.par.batch_size:
                self.model.train_models(self.optimizer_model, self.real_buffer, multistep_buffer=self.multistep_buffer)
                if self.par.monitor_losses and self.step_i % (250) == 0:
                    self.model.log_loss(self.real_buffer.sample_tensors(), "train")
                    self.model.log_loss(self.val_buffer.sample_tensors(), "test")
        #
        # if self.par.monitor_losses and self.step_i %(250) == 0:
        #     self.logger.info("alpha:" + str(self.alpha))

        if self.step_i % self.par.update_every_n_steps == 0:
            if self.real_buffer.len() >= self.par.batch_size and self.step_i > self.par.step_random:
                if self.par.use_model:
                    self.update_rollout_length()
                    self.generate_fake_samples()
                for step in range(self.par.n_steps):# * self.par.update_every_n_steps):
                    # Train actor and critic
                    b = self.sample_batch()
                    # Update Critic
                    self.update_critics(b)

                    if (self.step_i / self.par.update_every_n_steps) % self.par.delay == 0:
                        # Update Actor
                        self.update_actor(b)

                        # Update Target Networks
                        update_target_networks([self.actor] + self.critics,
                                               [self.actor_target] + self.critics_target,
                                               self.par.tau)

        # Select Action
        if self.step_i > self.par.step_random:
            action = self.actor.select_action(o, "noisy")
        else:
            action = self.actor.select_action(o, "random")

        self.o_old = o
        self.s_old = s
        if action.size() == torch.Size([]):
            self.a_old = action.unsqueeze(0)
        else:
            self.a_old = action
        self.step_i += 1

        self.traj_a.append(action)
        return action.detach().cpu().numpy()

    def generate_fake_samples(self):
        batch_this_epoch = 1000  # self.par.batch_size*self.par.update_steps*self.par.update_every_n_steps*8
        fake_samples = self.model.generate_efficient(self.model_sample_buffer.sample_tensors(n=batch_this_epoch),
                                                     self.actor,
                                                     diverse=self.par.diverse,
                                                     batch_size=batch_this_epoch)  # self.par.batch_size)
        self.fake_buffer.reallocate(size=batch_this_epoch * self.par.rollout_length)
        for item in fake_samples:
            self.fake_buffer.add_multiple(item)

    def sample_batch(self):
        n_real = int(self.par.batch_size * self.par.real_ratio)
        b_fake = self.ac_buffer.sample_tensors(n=self.par.batch_size - n_real)
        b_real = self.real_buffer.sample_tensors(n=n_real)
        b = dict()
        for key in b_fake.keys():
            b[key] = torch.cat([b_fake[key], b_real[key]])
        return b

    def update_rollout_length(self):
        if self.par.use_rollout_schedule:
            normdist = (self.step_i - self.par.rollout_schedule_time[0])/(self.par.rollout_schedule_time[1]-self.par.rollout_schedule_time[0])
            y = self.par.rollout_schedule_val[0]*(1-normdist) + self.par.rollout_schedule_val[1]*normdist
            self.par.rollout_length = int(max(min(y, self.par.rollout_schedule_val[1]), self.par.rollout_schedule_val[0]))

    def update_actor(self, b):
        for par in self.critics[0].parameters():
            par.requires_grad = False
        if self.use_correct:
            act, logp_pi = self.actor(b["o"], sample=False)
        else:
            act, logp_pi = self.actor(b["o_next"], sample=False)

        q1 = self.critics[0](b["o"], act).squeeze()
        q2 = self.critics[1](b["o"], act).squeeze()
        if self.use_min:
            q_min = torch.min(q1, q2)
        else:
            q_min = q1
        loss_actor = - torch.mean(q_min - logp_pi * self.alpha)
        self.optimizer_actor.zero_grad()
        loss_actor.backward()
        self.optimizer_actor.step()

        if self.par.autotune:
            with torch.no_grad():
                _, logp_pi = self.actor(b["o"], sample=False)
            alpha_loss = (-self.log_alpha * (logp_pi + self.par.target_entropy)).mean()
            self.optimizer_alpha.zero_grad()
            alpha_loss.backward()
            self.optimizer_alpha.step()
            self.alpha = self.log_alpha.exp().item()
        for par in self.critics[0].parameters():
            par.requires_grad = True

    def update_critics(self, b):
        with torch.no_grad():
            next_a, next_logp_pi = self.actor_target(b["o_next"], sample=False)
            min_next_Q = torch.min(
                *[critic_target(b["o_next"], next_a).squeeze() for critic_target in self.critics_target])
            y = b["r"] + (1 - b["done"]) * self.par.gamma * (min_next_Q - self.alpha * next_logp_pi)

        for optimizer, critic in zip(self.optimizer_critics, self.critics):
            loss = loss_critic(critic(b["o"], b["a"]).squeeze(), y, f_hyst=self.par.f_hyst)*0.5 #0.5 is to correspond with code of Janner
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()