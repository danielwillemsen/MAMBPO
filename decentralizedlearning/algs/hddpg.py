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
                                              (128, 128)))
        self.hidden_dims_model = tuple(kwargs.get("hidden_dims_model",
                                             (128, 128, 128)))
        self.use_OU = bool(kwargs.get("use_OU", False))
        self.gamma = float(kwargs.get("gamma", 0.99))
        self.tau = float(kwargs.get("tau", 0.005))
        self.delay = int(kwargs.get("delay", 2))
        self.lr_actor = float(kwargs.get("lr_actor", 0.0001))
        self.lr_critic = float(kwargs.get("lr_critic", 0.0001))
        self.lr_model = float(kwargs.get("lr_model", 0.0001))
        self.l2_norm = float(kwargs.get("l2_norm", 0.0))
        self.step_random = int(kwargs.get("step_random", 250))
        self.update_every_n_steps = int(kwargs.get("update_every_n_steps", 1))
        self.update_steps = int(kwargs.get("update_steps", 1))
        self.n_models = int(kwargs.get("n_models", 5))
        self.batch_size = int(kwargs.get("batch_size", 128))
        self.action_noise = float(kwargs.get("action_noise", 0.3))
        self.weight_decay = float(kwargs.get("weight_decay", 0.0))
        self.use_model = bool(kwargs.get("use_model", False))
        self.n_steps = int(kwargs.get("n_steps", 1))
        self.f_hyst = float(kwargs.get("f_hyst", 1.0))
        self.use_double = bool(kwargs.get("use_double", False))
        self.use_real_model = bool(kwargs.get("use_real_model", False))

class HDDPGAgent:
    def __init__(self, obs_dim, action_dim, hyperpar=None, **kwargs):
        # Initialize arguments
        if torch.cuda.is_available():
            print("Using not CUDA")
            self.device = torch.device("cpu")
        else:
            print("No CUDA found")
            self.device = torch.device("cpu")
        if hyperpar:
            self.par = hyperpar
        else:
            self.par = HDDPGHyperPar(**kwargs)
        if self.par.use_real_model:
            from gym.envs.classic_control.pendulum import PendulumEnv
            self.fake_env = PendulumEnv()

        self.action_dim = action_dim

        self.ac = ActorCritic(obs_dim, action_dim, hidden_dims_actor=self.par.hidden_dims_actor, hidden_dims_critic=self.par.hidden_dims_critic)
        self.ac_target = copy.deepcopy(self.ac)

        self.ac.to(self.device)
        self.ac_target.to(self.device)

        for par in self.ac_target.parameters():
            par.requires_grad = False

        if self.par.use_OU:
            self.ou = OUNoise(action_dim)

        self.buffer = ReplayBuffer()
        self.tempbuf = ReplayBuffer(size=5000)

        self.o_old = None
        self.a_old = None

        self.optimizer_critic = torch.optim.Adam(self.ac.critic.parameters(), lr=self.par.lr_critic, amsgrad=True, weight_decay=self.par.l2_norm)
        self.optimizer_actor = torch.optim.Adam(self.ac.actor.parameters(), lr=self.par.lr_actor, amsgrad=True, weight_decay=self.par.l2_norm)

        # Initialize models
        if self.par.use_model:
            self.models = []
            self.optimizer_models = []
            for k in range(self.par.n_models):
                model = Model(obs_dim + action_dim, self.par.hidden_dims_model, obs_dim).to(self.device)
                self.models.append(model)
                self.optimizer_models.append(torch.optim.Adam(model.parameters(),
                                                              lr=self.par.lr_model, amsgrad=True, weight_decay=self.par.l2_norm))
        self.i_step = 0

    def reset(self):
        self.o_old = None
        self.a_old = None
        if self.par.use_OU:
            self.ou.reset()

    def step(self, o, r, eval=False, done=False):
        o = torch.Tensor(o).to(self.device)
        r = torch.Tensor(np.array(float(r))).to(self.device)
        done = torch.Tensor(np.array(float(done))).to(self.device)
        if eval:
            action = self.ac.actor(o.unsqueeze(0)).squeeze()
            action = torch.clamp(action, -1., 1.0)
            return action.detach().numpy()
        if self.o_old is not None:
            self.buffer.add((self.o_old, self.a_old, r, o, done))

        if self.buffer.len() > self.buffer.n_samples:
            for i in range(5):
                if self.par.use_model and not self.par.use_real_model:
                    self.update_models()

            # if self.par.use_model:
            #     self.generate_from_model()
            if self.i_step % self.par.update_every_n_steps == 0:
                for i in range(self.par.n_steps*self.par.update_every_n_steps):
                    self.update_step()

        # Select Action
        action = self.select_action(o, "noisy")

        self.o_old = o

        if action.size() == torch.Size([]):
            self.a_old = action.unsqueeze(0)
        else:
            self.a_old = action

        self.i_step += 1
        return action.detach().cpu().numpy()

    def update_step(self):
        # Sample Minibatch
        if not self.par.use_model:
            b = self.buffer.sample_tensors(n=self.par.batch_size)
        elif not self.par.use_real_model:
            model = random.choice(self.models)
            b = self.sample_from_model(model)
        else:
            b = self.sample_from_real_model()
            # b = self.tempbuf.sample_tensors(n=self.par.batch_size)

        # Update Critic
        self.update_critic(b)
        # Update Actor
        self.update_actor(b)
        # Update Target Networks
        self.update_target_networks()

    def sample_from_real_model(self):
        # Sample Minibatch
        b = self.buffer.sample_tensors(n=self.par.batch_size)
        with torch.no_grad():
            action = b["a"] #self.ac.actor(b["o"])
            if self.par.use_OU:
                action_noisy = action + torch.Tensor(self.ou.noise())[0]
            else:
                action_noisy = action + torch.randn(action.size()).to(self.device) * 0.3
            b["a"] = torch.clamp(action_noisy, -1.0, 1.0)
        for i in range(len(b["a"])):
            a = b["a"][i].cpu().numpy()
            o = b["o"][i].cpu().numpy()
            obs_should = b["o_next"][i].cpu().numpy()
            r_should = b["r"][i].cpu().numpy()
            self.fake_env.state = np.array([np.arctan2(o[1], o[0]), o[2]])
            o2 = self.fake_env._get_obs()
            obs, r, _, _2 = self.fake_env.step(a[0]*0.5*(self.fake_env.action_space.high-self.fake_env.action_space.low))

            b["o_next"][i] = torch.from_numpy(obs).to("self.device")
            b["r"][i] = torch.from_numpy(np.array([r])).to("self.device")
        return b

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
            y = b["r"].unsqueeze(-1) + (1 - b["done"].unsqueeze(-1)) * self.par.gamma * self.ac_target.critic(b["o_next"],
                                                                                                self.ac_target.actor(
                                                                                                    b["o_next"]))
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
                    action = action + torch.randn(action.size()).to(self.device) * 0.3
                action = torch.clamp(action, -1.0, 1.0)
        return action

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
