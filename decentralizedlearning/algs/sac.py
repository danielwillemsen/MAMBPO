import torch
import copy
import numpy as np
import time

from decentralizedlearning.algs.utils import ReplayBuffer

from decentralizedlearning.algs.utils import Critic
from decentralizedlearning.algs.utils import StochActor
from decentralizedlearning.algs.utils import OUNoise
from decentralizedlearning.algs.utils import loss_critic
from decentralizedlearning.algs.utils import update_target_networks
from decentralizedlearning.algs.utils import convert_inputs_to_tensors
from decentralizedlearning.algs.models import EnsembleModel
import logging
class SACHyperPar:
    def __init__(self, **kwargs):
        self.hidden_dims_actor = tuple(kwargs.get("hidden_dims_actor",
                                             (16,)))
        self.hidden_dims_critic = tuple(kwargs.get("hidden_dims_critic",
                                              (16,)))
        self.hidden_dims_model = tuple(kwargs.get("hidden_dims_model",
                                             (16,)))
        self.use_OU = bool(kwargs.get("use_OU", False))
        self.gamma = float(kwargs.get("gamma", 0.99))
        self.tau = float(kwargs.get("tau", 0.005))
        self.delay = int(kwargs.get("delay", 2))
        self.lr_actor = float(kwargs.get("lr_actor", 0.0003))
        self.lr_critic = float(kwargs.get("lr_critic", 0.0003))
        self.lr_model = float(kwargs.get("lr_model", 0.001))
        self.l2_norm = float(kwargs.get("l2_norm", 0.0))

        self.step_random = int(kwargs.get("step_random", 000))
        self.update_every_n_steps = int(kwargs.get("update_every_n_steps", 1))
        self.update_model_every_n_steps = int(kwargs.get("update_model_every_n_steps",1000))
        self.update_steps = int(kwargs.get("n_steps", 5))
        self.n_models = int(kwargs.get("n_models", 10))
        self.batch_size = int(kwargs.get("batch_size", 256))
        self.weight_decay = float(kwargs.get("weight_decay", 0.0000))
        self.alpha = float(kwargs.get("alpha",0.05))
        self.f_hyst = float(kwargs.get("f_hyst", 1.0))
        self.use_model = bool(kwargs.get("use_model", False))
        self.monitor_losses = bool(kwargs.get("monitor_losses", True))
        self.use_model_stochastic = bool(kwargs.get("use_model_stochastic", False))
        self.diverse = bool(kwargs.get("diverse", True))
        self.autotune = bool(kwargs.get("autotune", True))
        self.target_entropy = float(kwargs.get("target_entropy", -3.))

class SAC:
    def __init__(self, obs_dim, action_dim, hyperpar=None, **kwargs):
        # Initialize arguments
        self.logger = logging.getLogger('root')

        if torch.cuda.is_available():
            print("Using CUDA")
            self.device = torch.device("cuda:0")
            #self.device = torch.device("cpu")
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
        self.action_dim = action_dim

        # Initialize actor
        self.actor = StochActor(obs_dim, self.par.hidden_dims_actor,  action_dim)
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
                                                    lr=0.001)#self.par.lr_actor)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = self.par.alpha

        if self.par.use_model:
            self.model = EnsembleModel(obs_dim + action_dim,
                                       self.par.hidden_dims_model,
                                       obs_dim,
                                       self.par.n_models,
                                       monitor_losses=self.par.monitor_losses,
                                       use_stochastic=self.par.use_model_stochastic).to(self.device)
            self.optimizer_model = torch.optim.Adam(self.model.parameters(),
                                                              lr=self.par.lr_model, amsgrad=True, weight_decay=self.par.l2_norm)
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

        self.real_buffer = ReplayBuffer(batch_size=self.par.batch_size)
        if self.par.monitor_losses:
            self.val_buffer = ReplayBuffer(batch_size=100)
        if self.par.use_model:
            self.fake_buffer = ReplayBuffer(size=256, batch_size=self.par.batch_size)
            self.ac_buffer = self.fake_buffer
        else:
            self.ac_buffer = self.real_buffer

        self.o_old = None
        self.a_old = None



        self.step_i = 0

    def reset(self):
        self.o_old = None
        self.a_old = None
        if self.par.use_OU:
            self.ou.reset()

    def step(self, o, r, eval=False, done=False, generate_val_data=False):
        o, r, done = convert_inputs_to_tensors(o, r, done, self.device)

        if eval:
            # Select greedy action and return
            action = self.select_action(o, "greedy")
            return action.detach().cpu().numpy()

        if generate_val_data:
            # Select greedy action and return
            action = self.select_action(o, "noisy")
            if self.o_old is not None:
                self.val_buffer.add((self.o_old, self.a_old, r, o, done))
            self.o_old = o
            if action.size() == torch.Size([]):
                self.a_old = action.unsqueeze(0)
            else:
                self.a_old = action
            return action.detach().cpu().numpy()

        # Do training process step
        if self.o_old is not None:
            self.real_buffer.add((self.o_old, self.a_old, r, o, done))

        if self.step_i % self.par.update_model_every_n_steps == 0:
            #Update model and generate new samples:
            if self.par.use_model and self.real_buffer.len() > self.par.batch_size:
                #for i in range(1*self.par.update_every_n_steps):
                    #self.model.update_step(self.optimizer_model, self.real_buffer.sample_tensors())
                self.model.train_models(self.optimizer_model, self.real_buffer)
                #self.model.generate_batch(self.real_buffer.sample_tensors())
                if self.par.monitor_losses and self.step_i % (250) == 0:
                    self.model.log_loss(self.real_buffer.sample_tensors(), "train")
                    self.model.log_loss(self.val_buffer.sample_tensors(), "test")
                    self.logger.info("alpha:"+str(self.alpha))
                    print(len(self.fake_buffer))


        if self.step_i % self.par.update_every_n_steps == 0:
            for step in range(self.par.update_steps*self.par.update_every_n_steps):
                if self.real_buffer.len() >= self.par.batch_size and self.step_i>self.par.step_random:
                    if self.par.use_model:
                        fake_samples = self.model.generate(self.real_buffer.sample_tensors(n=1024), self.actor, diverse=self.par.diverse, batch_size=1024)#self.par.batch_size)
                        for item in fake_samples:
                            self.fake_buffer.add(item)
                    # Train actor and critic
                    b = self.ac_buffer.sample_tensors(n=self.par.batch_size)

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
            action = self.select_action(o, "noisy")
        else:
            action = self.select_action(o, "random")

        self.o_old = o
        if action.size() == torch.Size([]):
            self.a_old = action.unsqueeze(0)
        else:
            self.a_old = action
        self.step_i += 1
        return action.detach().cpu().numpy()

    def select_action(self, o, method):
        assert method in ["random", "noisy", "greedy"], "Invalid action selection method"
        if method == "random":
            return torch.rand(self.action_dim, dtype=torch.float, device=self.device)*2.-1.

        with torch.no_grad():
            if method == "greedy":
                action = self.actor(o.unsqueeze(0), greedy=True).squeeze()
            else:
                action = self.actor(o.unsqueeze(0), greedy=False).squeeze()
            return action

    def update_actor(self, b):
        for par in self.critics[0].parameters():
            par.requires_grad = False
        self.optimizer_actor.zero_grad()
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
            a_target, logp_pi = self.actor_target(b["o_next"], sample=False)
            y = b["r"].unsqueeze(-1) + (1 - b["done"]) * self.par.gamma * (torch.min(
                *[critic_target(b["o_next"], a_target) for critic_target in self.critics_target]) - self.alpha * logp_pi)

        for optimizer, critic in zip(self.optimizer_critics, self.critics):
            loss = loss_critic(critic(b["o"], b["a"]), y, f_hyst=self.par.f_hyst)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def update_models(self):
        samples = self.buffer.sample_tensors(n=self.par.batch_size)
        for optim, model in zip(self.optimizer_models, self.models):
            self.model_step(model, optim, samples)

    def sample_from_model(self, model):
        # Sample Minibatch
        b = self.buffer.sample_tensors(n=self.par.batch_size)
        with torch.no_grad():
            action = b["a"]#self.ac.actor(b["o"])
            if self.par.use_OU:
                action_noisy = action + torch.Tensor(self.ou.noise())[0]
            else:
                action_noisy = action + torch.randn(action.size()).to(self.device) * 0.3
            b["a"] = torch.clamp(action_noisy, -1.0, 1.0)
        new_o, r = model.sample(b["o"], b["a"])
        b["o_next"] = new_o
        b["r"] = r.squeeze()
        return b

    def generate_from_model(self, rollout=1, batches=10, size=128):
        for batch in range(batches):
            b = self.buffer.sample_tensors(n=size).copy()
            for _ in range(rollout):
                model = random.choice(self.models)
                with torch.no_grad():
                    action = self.ac.actor(b["o"])
                    if self.par.use_OU:
                        action_noisy = action + torch.Tensor(self.ou.noise())[0]
                    else:
                        action_noisy = action + torch.randn(action.size()) * 0.3
                    b["a"] = torch.clamp(action_noisy, -1.0, 1.0)
                new_o, r = model.sample(b["o"], b["a"])
                for o, a, new_oi, new_ri, done in zip(b["o"], b["a"], new_o, r, b["done"]):
                    self.tempbuf.add((o, a, new_ri, new_oi, done))
                b = {"o": new_o, "done": b["done"]}

    def model_step(self, model, optim, samples):
        o_next_pred, r_pred = model(samples["o"], samples["a"])
        sigma = o_next_pred[1]
        sigma_2 = r_pred[1]
        mu = o_next_pred[0]
        #print(torch.max(torch.abs(mu)))
        target = samples["o_next"]
        #print(torch.max(sigma))
        #print(torch.max(sigma_2))
        loss1 = torch.mean((mu - target) / sigma * (mu - target))
        # print(mu-target)
        loss3 = torch.mean(torch.log(torch.prod(sigma, 1))) + torch.mean(torch.log(torch.prod(sigma_2, 1)))
        mu2 = r_pred[0]
        #print(torch.max(torch.abs(mu)))
        target = samples["r"].unsqueeze(-1)
        loss2 = torch.mean((mu2 - target) / sigma_2 * (mu2 - target))
        # print(mu2-target)
        # print(sigma_2)
        loss = loss1 + loss2 + loss3
        optim.zero_grad()
        loss.backward()
        optim.step()