import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random
import numpy as np
from decentralizedlearning.algs.utils import OUNoise
from decentralizedlearning.algs.utils import ActorCritic
from decentralizedlearning.algs.utils import EfficientReplayBuffer as ReplayBuffer
from decentralizedlearning.algs.utils import loss_critic
from decentralizedlearning.algs.models import Model
from decentralizedlearning.algs.utils import check_cuda
from decentralizedlearning.algs.utils import convert_inputs_to_tensors
from decentralizedlearning.algs.utils import update_target_networks
from decentralizedlearning.algs.utils import Actor
from decentralizedlearning.algs.utils import Critic
from decentralizedlearning.algs.utils import convert_multi_inputs_to_tensors


class HDDPGHyperPar:
    def __init__(self, **kwargs):
        self.hidden_dims_actor = tuple(kwargs.get("hidden_dims_actor",
                                             (128, 128)))
        self.hidden_dims_critic = tuple(kwargs.get("hidden_dims_critic",
                                              (128, 128)))
        self.hidden_dims_model = tuple(kwargs.get("hidden_dims_model",
                                             (128, 128, 128)))
        self.use_OU = bool(kwargs.get("use_OU", False))
        self.gamma = float(kwargs.get("gamma", 0.95))
        self.tau = float(kwargs.get("tau", 0.005))
        self.delay = int(kwargs.get("delay", 2))
        self.lr_actor = float(kwargs.get("lr_actor", 0.001))
        self.lr_critic = float(kwargs.get("lr_critic", 0.001))
        self.lr_model = float(kwargs.get("lr_model", 0.0001))
        self.l2_norm = float(kwargs.get("l2_norm", 0.0))
        self.step_random = int(kwargs.get("step_random", 500))
        self.update_every_n_steps = int(kwargs.get("update_every_n_steps", 100))
        self.update_steps = int(kwargs.get("update_steps", 1))
        self.n_models = int(kwargs.get("n_models", 10))
        self.batch_size = int(kwargs.get("batch_size", 512))
        self.action_noise = float(kwargs.get("action_noise", 0.1))
        self.weight_decay = float(kwargs.get("weight_decay", 0.0))
        self.use_model = bool(kwargs.get("use_model", False))
        self.n_steps = int(kwargs.get("n_steps", 1))
        self.f_hyst = float(kwargs.get("f_hyst", 1.0))
        self.use_double = bool(kwargs.get("use_double", False))
        self.use_real_model = bool(kwargs.get("use_real_model", False))

class MADDPGAgent:
    def __init__(self, obs_dim, hidden_dims_actor, action_dim, device, par):
        self.step_i = 0
        self.par = par
        self.device = device
        self.action_dim = action_dim
        self.actor = Actor(obs_dim, hidden_dims_actor, action_dim)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor.to(device)
        self.actor_target.to(device)
        self.model = None
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.par.lr_actor,
                                                weight_decay=self.par.weight_decay)
        for par in self.actor_target.parameters():
            par.requires_grad = False

    def reset(self):
        pass

    def step(self, o, r, eval=False, done=False, generate_val_data=False, greedy_eval=True, s=None):
        o, r, done = convert_inputs_to_tensors(o, r, done, self.device)
        if self.step_i > self.par.step_random:
            action = self.select_action(o, "noisy")
        else:
            action = self.select_action(o, "random")

        # Update previous action and observations
        self.o_old = o

        if action.size() == torch.Size([]):
            self.a_old = action.unsqueeze(0)
        else:
            self.a_old = action

        # print(r)
        return action.detach().cpu().numpy()

    def select_action(self, o, method):
        assert method in ["random", "noisy", "greedy"], "Invalid action selection method"
        if method == "random":
            return torch.rand(self.action_dim, dtype=torch.float, device=self.device)*2.-1.

        with torch.no_grad():
            action = self.actor(o.unsqueeze(0)).squeeze()
            if method == "noisy":
                if self.par.use_OU:
                    NotImplementedError()
                else:
                    action = action + torch.randn(action.size(), dtype=torch.float, device=self.device) * 0.3
                    # print(action)
                action = torch.clamp(action, -1.0, 1.0)
        return action

class MADDPG:
    def __init__(self, n_agents, obs_dim, action_dim, hyperpar=None, **kwargs):
        # Initialize arguments
        self.device = check_cuda()

        if hyperpar:
            self.par = hyperpar
        else:
            self.par = HDDPGHyperPar(**kwargs)

        self.obs_dim = obs_dim
        self.n_agents = n_agents
        self.action_dim = action_dim

        self.critic = Critic((obs_dim + action_dim)*n_agents, self.par.hidden_dims_critic)
        self.critic_target = copy.deepcopy(self.critic)

        self.critic.to(self.device)
        self.critic_target.to(self.device)

        for par in self.critic_target.parameters():
            par.requires_grad = False

        if self.par.use_OU:
            self.ou = OUNoise(action_dim)

        self.buffer = ReplayBuffer()

        self.o_old = None
        self.a_old = None
        self.agents = [MADDPGAgent(obs_dim, self.par.hidden_dims_actor, action_dim, self.device, self.par) for i in range(n_agents)]

        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.par.lr_critic, amsgrad=True, weight_decay=self.par.l2_norm)
        # self.optimizer_actor = torch.optim.Adam(self.ac.actor.parameters(), lr=self.par.lr_actor, amsgrad=True, weight_decay=self.par.l2_norm)

        # Initialize models
        if self.par.use_model:
            if self.par.use_real_model:
                from gym.envs.classic_control.pendulum import PendulumEnv
                self.fake_env = PendulumEnv()
            self.tempbuf = ReplayBuffer(size=5000)
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

    def step(self, o, r, a, eval=False, done=False, generate_val_data=False, **kwargs):
        o, r, done, a = convert_multi_inputs_to_tensors(o, r, done, a, self.device)
        for agent in self.agents:
            agent.step_i += 1
        # If evaluation mode: only select action and exit.
        # if eval:
        #     action = self.ac.actor(o.unsqueeze(0)).squeeze()
        #     action = torch.clamp(action, -1., 1.0)
        #     return action.detach().numpy()

        # If not in evaluation mode:
        # Update buffer with new observations
        if self.o_old is not None:
            self.buffer.add((self.o_old, self.a_old, r, o, done))

        # Update model (if need be)
        if self.par.use_model and not self.par.use_real_model:
            if self.buffer.len() > self.par.batch_size:
                for i in range(10):
                    self.update_models()

        # Update actors and critics
        if self.i_step % self.par.update_every_n_steps == 0 and self.buffer.len() > 10000:#self.par.batch_size:
            for i in range(self.par.n_steps):#*self.par.update_every_n_steps):
                self.update_step()

        # Update previous action and observations
        self.o_old = o
        self.a_old = a
        self.i_step += 1
        # print(r)
        return

    def update_step(self):
        # Sample Minibatch
        if not self.par.use_model:
            b = self.buffer.sample_tensors(n=self.par.batch_size)
        elif not self.par.use_real_model:
            model = random.sample(self.models)
            b = self.sample_from_model(model)
        else:
            b = self.sample_from_real_model()
            # b = self.tempbuf.sample_tensors(n=self.par.batch_size)

        # Update Critic
        self.update_critic(b)
        # Update Actor
        self.update_actor(b)
        # Update Target Networks
        update_target_networks([agent.actor for agent in self.agents] + [self.critic],
                               [agent.actor_target for agent in self.agents] + [self.critic_target], self.par.tau)

    def sample_from_real_model(self):
        # Sample Minibatch
        b = self.buffer.sample_tensors(n=self.par.batch_size)
        with torch.no_grad():
            action = b["a"] #self.ac.actor(b["o"])
            if self.par.use_OU:
                action_noisy = action + torch.Tensor(self.ou.noise())[0]
            else:
                action_noisy = action + torch.randn(action.size()).to(self.device) * 0.1
            b["a"] = torch.clamp(action_noisy, -1.0, 1.0)
        for i in range(len(b["a"])):
            a = b["a"][i].cpu().numpy()
            o = b["o"][i].cpu().numpy()
            obs_should = b["o_next"][i].cpu().numpy()
            r_should = b["r"][i].cpu().numpy()
            self.fake_env.state = np.array([np.arctan2(o[1], o[0]), o[2]])
            o2 = self.fake_env._get_obs()
            obs, r, _, _2 = self.fake_env.step(a[0]*0.5*(self.fake_env.action_space.high-self.fake_env.action_space.low))

            b["o_next"][i] = torch.from_numpy(obs).to(self.device)
            b["r"][i] = torch.from_numpy(np.array([r])).to(self.device)
        return b

    def update_actor(self, b):
        for par in self.critic.parameters():
            par.requires_grad = False
        act_n = []
        for i, agent in enumerate(self.agents):
            act = agent.actor(b["o"][:, self.obs_dim*i:(self.obs_dim)*(i+1)])
            act_n.append(act)
            agent.optimizer_actor.zero_grad()

        act = torch.cat(act_n, dim=-1)
        loss_actor = -torch.mean(self.critic(b["o"], act))
        loss_actor.backward()
        # print(loss_actor)
        # save = copy.deepcopy(self.agents[0].actor.net[0].weight)
        # print(self.agents[0].actor.net[0].weight[0:5])
        # for i, agent in enumerate(self.agents):
        #     agent.optimizer_actor.step()
        # for i, agent in enumerate(self.agents):
        #     act = agent.actor(b["o"][:, self.obs_dim*i:(self.obs_dim)*(i+1)])
        #     act_n.append(act)
        #     agent.optimizer_actor.zero_grad()
        # loss_actor = -torch.mean(self.critic(b["o"], act))
        # print(loss_actor)
        # print("Change: " + str(torch.sum(save-self.agents[0].actor.net[0].weight)))
        for par in self.critic.parameters():
            par.requires_grad = True

    def update_critic(self, b):
        self.optimizer_critic.zero_grad()
        with torch.no_grad():
            next_a_n = []
            for i, agent in enumerate(self.agents):
                next_a = agent.actor_target(b["o_next"][:, self.obs_dim*i:(self.obs_dim)*(i+1)])
                next_a_n.append(next_a)
            next_a = torch.cat(next_a_n, dim=-1)

            y = b["r"].unsqueeze(-1) + (1 - b["done"].unsqueeze(-1)) * self.par.gamma * self.critic_target(b["o_next"],
                                                                                                next_a)
        loss_c = loss_critic(self.critic(b["o"], b["a"]), y, f_hyst=self.par.f_hyst)
        if np.random.rand() < 1.0:
            print("Mean Q:", torch.mean(y))
            print("LOSSQ:" + str(loss_c))
        loss_c.backward()
        # print(loss_c)
        self.optimizer_critic.step()



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
                model = random.sample(self.models)
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
