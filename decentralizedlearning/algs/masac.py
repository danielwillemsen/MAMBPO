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
from decentralizedlearning.algs.utils import convert_multi_inputs_to_tensors
from decentralizedlearning.algs.models import EnsembleModel
from decentralizedlearning.algs.models import DegradedSim
from decentralizedlearning.algs.utils import convert_inputs_to_tensors
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

        self.step_random = int(kwargs.get("step_random", 500))
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
        self.use_stochastic_actor = bool(kwargs.get("use_stochastic_actor", True))

class MASACAgent:
    def __init__(self, action_dim, device, par,
                 actor, actor_target, actor_optimizer,
                 critics, critics_target, critics_optimzer):
        self.step_i = 0
        self.par = par
        self.device = device
        self.action_dim = action_dim
        self.actor = actor
        self.actor_target = actor_target
        self.actor_optimizer = actor_optimizer

        self.critics = critics
        self.critics_target = critics_target
        self.critics_optimizer = critics_optimzer
        self.model = None
        for par in self.actor_target.parameters():
            par.requires_grad = False
        if self.par.autotune:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.optimizer_alpha = torch.optim.Adam([self.log_alpha],
                                                    lr=self.par.lr_actor)#self.par.lr_actor)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = self.par.alpha
            
    def step(self, o, r, eval=False, done=False, generate_val_data=False, greedy_eval=True, s=None):
        o, r, done = convert_inputs_to_tensors(o, r, done, self.device)
        # Select Action
        if eval:
            action = self.actor.select_action(o, "greedy")
        elif self.step_i > self.par.step_random:
            action = self.actor.select_action(o, "noisy")
        else:
            action = self.actor.select_action(o, "random")
        return action.detach().cpu().numpy()

    def reset(self):
        return



class MASAC:
    def __init__(self, n_agents, observation_space, action_space, hyperpar=None, discrete=False, **kwargs):
        # Initialize arguments
        self.discrete = discrete
        self.obs_dims = [obs_dim.shape[0] for obs_dim in observation_space]
        self.action_dims = [action_dim.shape[0] for action_dim in action_space]
        self.act_i_per_agent = list(np.cumsum([0]+self.action_dims))
        self.obs_i_per_agent = list(np.cumsum([0]+self.obs_dims))
        self.logger = logging.getLogger('root')
        self.n_agents = n_agents
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
        self.action_dim = self.action_dims[0]
        self.obs_dim = self.obs_dims[0]

        self.agents = self.initialize_agents(self.action_dim, n_agents, self.obs_dim)

        if self.par.use_model:
            self.model = EnsembleModel(np.sum(self.obs_dims) + np.sum(self.action_dims),
                                       self.par.hidden_dims_model,
                                       np.sum(self.obs_dims),
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

        self.real_buffer = ReplayBuffer(size=1000000, batch_size=self.par.batch_size, device=self.device)
        if not self.par.use_degraded_sim:
            self.model_sample_buffer = self.real_buffer
        else:
            env_copy = kwargs.get("env_copy", None)
            self.model = DegradedSim(env_copy, degradation=0.0, bias=0.4, device=self.device)
            self.model_sample_buffer = StateBuffer(device=self.device)

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



        self.step_i = 0

    def initialize_agents(self, action_dim, n_agents, obs_dim):
        agents = []
        if self.par.use_common_actor:
            actor = StochActor(self.obs_dims[0], self.par.hidden_dims_actor, self.action_dims[0], discrete=self.discrete, device=self.device)
            actor_target = copy.deepcopy(actor)
            actor.to(self.device)
            actor_target.to(self.device)
            actor_optimizer = torch.optim.Adam(actor.parameters(),
                                               lr=self.par.lr_actor,
                                               weight_decay=self.par.weight_decay)
        if self.par.use_common_critic:
            critics = [Critic(np.sum(self.obs_dims)+np.sum(self.action_dims), self.par.hidden_dims_critic) for i in range(2)]
            critics_target = copy.deepcopy(critics)
            critics_optimizers = []
            for critic, critic_target in zip(critics, critics_target):
                critic.to(self.device)
                critic_target.to(self.device)
                critics_optimizers.append(torch.optim.Adam(critic.parameters(),
                                                           lr=self.par.lr_critic,
                                                           weight_decay=self.par.weight_decay))
        for i_agent in range(n_agents):
            if not self.par.use_common_actor:
                actor = StochActor(self.obs_dims[i_agent], self.par.hidden_dims_actor, self.action_dims[i_agent], discrete=self.discrete, device=self.device)
                actor_target = copy.deepcopy(actor)
                actor.to(self.device)
                actor_target.to(self.device)
                actor_optimizer = torch.optim.Adam(actor.parameters(),
                                                   lr=self.par.lr_actor,
                                                   weight_decay=0.001)
            if not self.par.use_common_critic:
                critics = [Critic(np.sum(self.obs_dims)+np.sum(self.action_dims), self.par.hidden_dims_critic) for i in range(2)]
                critics_target = copy.deepcopy(critics)
                critics_optimizers = []
                for critic, critic_target in zip(critics, critics_target):
                    critic.to(self.device)
                    critic_target.to(self.device)
                    critics_optimizers.append(torch.optim.Adam(critic.parameters(),
                                                               lr=self.par.lr_critic,
                                                               weight_decay=self.par.weight_decay))
            agents.append(MASACAgent(self.action_dims[i_agent], self.device, self.par,
                                     actor, actor_target, actor_optimizer,
                                     critics, critics_target, critics_optimizers))
        return agents

    def reset(self):
        self.o_old = None
        self.a_old = None
        self.traj_o = []
        self.traj_r = []
        self.traj_a = []
        self.traj_done = []

        if self.par.use_OU:
            self.ou.reset()

    def step(self, o, r, a, eval=False, done=False, generate_val_data=False, greedy_eval=True, s=None):
        o, r, done, a = convert_multi_inputs_to_tensors(o, r, done, a, self.device)
        # r+= 10.0
        # self.logger.info(done)
        for agent in self.agents:
            agent.step_i += 1
        # Do training process step
        # self.traj_o.append(o)
        # self.traj_done.append(done)
        # self.traj_r.append(r)

        if self.o_old is not None:
            self.real_buffer.add((self.o_old, self.a_old, r, o, done))
        if self.par.use_degraded_sim and self.s_old is not None:
            self.model_sample_buffer.add((self.o_old, self.a_old, r, o, done), self.s_old)

        if self.par.use_multistep_reg:
            if len(self.traj_o)>self.multistep_buffer.length:
                self.multistep_buffer.add_mini_traj(self.traj_o[-6:], self.traj_a[-5:], self.traj_r[-5:], self.traj_done[-5:])
        if self.step_i % self.par.update_model_every_n_steps == 0:
            #Update model and generate new samples:
            if self.par.use_model and self.real_buffer.len() > self.par.batch_size:
                #for i in range(1*self.par.update_every_n_steps):
                    #self.model.update_step(self.optimizer_model, self.real_buffer.sample_tensors())
                self.model.train_models(self.optimizer_model, self.real_buffer, multistep_buffer=self.multistep_buffer)
                #self.model.generate_batch(self.real_buffer.sample_tensors())
                # if self.par.monitor_losses and self.step_i % (250) == 0:
                #     self.model.log_loss(self.real_buffer.sample_tensors(), "train")
                #     self.model.log_loss(self.val_buffer.sample_tensors(), "test")

        if self.step_i % self.par.update_every_n_steps == 0 and self.step_i > 25*1000:
            if self.real_buffer.len() >= self.par.batch_size and self.step_i > self.par.step_random:
                if self.par.use_model:
                    self.update_rollout_length()
                    batch_this_epoch = 2000*self.par.n_steps#self.par.batch_size*self.par.update_steps*self.par.update_every_n_steps*8
                    fake_samples = self.model.generate_efficient(self.model_sample_buffer.sample_tensors(n=batch_this_epoch),
                                                                 [agent.actor for agent in self.agents],
                                                                 diverse=self.par.diverse,
                                                                 batch_size=batch_this_epoch,
                                                                 obs_i_per_agent=self.obs_i_per_agent,
                                                                 act_i_per_agent=self.act_i_per_agent)  # self.par.batch_size)
                    self.fake_buffer.reallocate(size=batch_this_epoch*self.par.rollout_length)
                    for item in fake_samples:
                        self.fake_buffer.add_multiple(item)
                for step in range(self.par.n_steps):#*self.par.update_every_n_steps):
                    # Train actor and critic
                    n_real = int(self.par.batch_size * self.par.real_ratio)
                    b_fake = self.ac_buffer.sample_tensors(n=self.par.batch_size-n_real)
                    b_real = self.real_buffer.sample_tensors(n=n_real)
                    b = dict()
                    for key in b_fake.keys():
                        b[key] = torch.cat([b_fake[key], b_real[key]])
                    # Update Critic
                    self.update_critics(b)

                    if (self.step_i / self.par.update_every_n_steps) % self.par.delay == 0:
                        # Update Actor
                        self.update_actor(b)

                        # Update Target Networks
                        if self.par.use_common_actor:
                            update_target_networks([self.agents[0].actor], [self.agents[0].actor_target], self.par.tau)
                        else:
                            update_target_networks([agent.actor for agent in self.agents],
                                                   [agent.actor_target for agent in self.agents],
                                                   self.par.tau)
                        if self.par.use_common_critic:
                            update_target_networks(self.agents[0].critics, self.agents[0].critics_target, self.par.tau)
                        else:
                            for agent in self.agents:
                                update_target_networks(agent.critics, agent.critics_target, self.par.tau)

        self.o_old = o
        self.s_old = s
        self.a_old = a
        self.step_i += 1


    def update_rollout_length(self):
        if self.par.use_rollout_schedule:
            normdist = (self.step_i - self.par.rollout_schedule_time[0])/(self.par.rollout_schedule_time[1]-self.par.rollout_schedule_time[0])
            y = self.par.rollout_schedule_val[0]*(1-normdist) + self.par.rollout_schedule_val[1]*normdist
            self.par.rollout_length = int(max(min(y, self.par.rollout_schedule_val[1]), self.par.rollout_schedule_val[0]))

    def update_actor(self, b):
        if self.par.use_common_critic:
            for critic in self.agents[0].critics:
                for par in critic.parameters():
                    par.requires_grad = False
        else:
            for agent in self.agents:
                for critic in agent.critics:
                    for par in critic.parameters():
                        par.requires_grad = False

        for i, agent in enumerate(self.agents):
            b = self.sample_batch()
            act, logp_pi = agent.actor(b["o"][:, self.obs_i_per_agent[i]:self.obs_i_per_agent[i+1]], sample=False)
            actions = b["a"].clone()
            # print(act)
            actions[:, self.act_i_per_agent[i]:self.act_i_per_agent[i+1]] = act

            q1 = agent.critics[0](b["o"], actions).squeeze()
            q2 = agent.critics[1](b["o"], actions).squeeze()
            if self.use_min:
                q_min = torch.min(q1, q2)
            else:
                q_min = q1
            loss_actor = - torch.mean(q_min - logp_pi * self.agents[0].alpha)
            agent.actor_optimizer.zero_grad()
            loss_actor.backward()
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)

            # print(str(loss_actor) + "---" + str(torch.mean(q_min)))
            agent.actor_optimizer.step()

        # for agent in self.agents:
            #
            # if self.par.autotune:
            #     with torch.no_grad():
            #         _, logp_pi = agent.actor(b["o"], sample=False)
            #     alpha_loss = (-agent.log_alpha * (logp_pi + self.par.target_entropy)).mean()
            #     agent.optimizer_alpha.zero_grad()
            #     alpha_loss.backward()
            #     agent.optimizer_alpha.step()
            #     agent.alpha = agent.log_alpha.exp().item()
        if self.par.use_common_critic:
            for critic in self.agents[0].critics:
                for par in critic.parameters():
                    par.requires_grad = True
        else:
            for agent in self.agents:
                for critic in agent.critics:
                    for par in critic.parameters():
                        par.requires_grad = True

    def get_batch_single_agent(self, batch, name, i):
        return batch[name][:]

    def update_critics(self, b):
        if self.par.use_common_critic:
            with torch.no_grad():
                next_a_n = []
                next_logp_pi_n = []
                for i, agent in enumerate(self.agents):
                    next_a, next_logp_pi = agent.actor(b["o_next"][:, self.obs_i_per_agent[i]:self.obs_i_per_agent[i+1]],
                                                              sample=False)
                    next_a_n.append(next_a)
                    next_logp_pi_n.append(next_logp_pi)
                next_a = torch.cat(next_a_n, dim=-1)
                next_logp_pi = torch.stack(next_logp_pi_n, dim=0).mean(dim=0)
            # if np.random.random()<0.05:

            for optimizer, critic in zip(self.agents[0].critics_optimizer, self.agents[0].critics):
                if self.use_min:
                    min_next_Q = torch.min(
                        *[critic_target(b["o_next"], next_a).squeeze() for critic_target in self.agents[0].critics_target])
                else:
                    min_next_Q = self.agents[0].critics_target[0](b["o_next"], next_a).squeeze()
                y = b["r"] + (1 - b["done"]) * self.par.gamma * (min_next_Q - self.agents[0].alpha * next_logp_pi)

                loss = loss_critic(critic(b["o"], b["a"]).squeeze(), y, f_hyst=self.par.f_hyst)*0.5 #0.5 is to correspond with code of Janner
                # print(torch.mean(min_next_Q))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                optimizer.step()
        else:
            for agent in self.agents:
                b = self.sample_batch()
                with torch.no_grad():
                    next_a_n = []
                    next_logp_pi_n = []
                    for i, agent2 in enumerate(self.agents):
                        next_a, next_logp_pi = agent2.actor_target(
                            b["o_next"][:, self.obs_i_per_agent[i]:self.obs_i_per_agent[i+1]], sample=False)
                        next_a_n.append(next_a)
                        if agent2 == agent:
                            next_logp_pi_n.append(next_logp_pi)
                        # if np.random.rand()<0.01:
                        #     print("LOGPOL:" + str(torch.mean(next_logp_pi)))
                    next_a = torch.cat(next_a_n, dim=-1)
                    next_logp_pi = torch.stack(next_logp_pi_n, dim=0).sum(dim=0) # How to calculate this?
                for optimizer, critic in zip(agent.critics_optimizer, agent.critics):
                    if self.use_min:
                        min_next_Q = torch.min(
                            *[critic_target(b["o_next"], next_a).squeeze() for critic_target in
                              agent.critics_target])
                    else:
                        min_next_Q = agent.critics_target[0](b["o_next"], next_a).squeeze()

                    y = b["r"] + (1 - b["done"]) * self.par.gamma * (min_next_Q - self.agents[0].alpha * next_logp_pi)

                    loss = loss_critic(critic(b["o"], b["a"]).squeeze(), y,
                                       f_hyst=self.par.f_hyst)#* 0.5  # 0.5 is to correspond with code of Janner
                    # print(torch.mean(min_next_Q))
                    # if np.random.rand() < 0.01:
                    #     print("Mean target: ", torch.mean(y))
                    #     print("LOSSQ:" + str(loss))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

    def sample_batch(self):
        n_real = int(self.par.batch_size * self.par.real_ratio)
        b_fake = self.ac_buffer.sample_tensors(n=self.par.batch_size - n_real)
        b_real = self.real_buffer.sample_tensors(n=n_real)
        b = dict()
        for key in b_fake.keys():
            b[key] = torch.cat([b_fake[key], b_real[key]])
        return b
