import pickle as p
import time
import os, sys

sys.path.insert(1, os.path.join(sys.path[0], '../decentralizedlearning/submodules/multi-agent-particle-envs'))
sys.path.insert(1, os.path.join(sys.path[0], '../decentralizedlearning'))
sys.path.insert(1, os.path.join(sys.path[0], '../'))
import pybullet_envs
import argparse
import numpy as np
from envwrapper import EnvWrapper

from decentralizedlearning.algs.hddpg import HDDPGAgent
from decentralizedlearning.algs.td3 import TD3
from decentralizedlearning.algs.modelbased import ModelAgent
from decentralizedlearning.algs.sac import SAC
from decentralizedlearning.algs.models import *

import logging
import matplotlib.pyplot as plt


class LoggerWriter:
    def __init__(self, level):
        # self.level is really like using log.debug(message)
        # at least in my case
        self.level = level

    def write(self, message):
        # if statement reduces the amount of newlines that are
        # printed to the logger
        if message != '\n':
            self.level(message)

    def flush(self):
        # create a flush method so things can be flushed when
        # the system wants to. Not sure if simply 'printing'
        # sys.stderr is the correct way to do it, but it seemed
        # to work properly for me.
        self.level(sys.stderr)


def scale_action(env, agent_id, action):
    return (env.action_space[agent_id].high - env.action_space[agent_id].low) * action * 0.5


def run_episode(env, agents, eval=False, render=False, generate_val_data=False, steps=1000):
    obs_n = env.reset()
    reward_tot = [0.0 for i in range(len(agents))]
    reward_n = [0.0 for i in range(len(agents))]

    # Start env
    for i in range(steps):
        # query for action from each agent's policy
        act_n = []
        for j, agent in enumerate(agents):
            action = scale_action(env, j,
                                  agent.step(obs_n[j], reward_n[j], eval=eval, generate_val_data=generate_val_data))
            act_n.append(action)
        # step environment
        obs_n, reward_n, done_n, _ = env.step(act_n)

        for j, r in enumerate(reward_n):
            reward_tot[j] += r
        if done_n[0]:
            for j, agent in enumerate(agents):
                action = scale_action(env, j,
                                      agent.step(obs_n[j], reward_n[j], eval=eval, generate_val_data=generate_val_data))
                act_n.append(action)
                agent.reset()
            print("Episode finished after {} timesteps".format(i + 1))
            break
        # render all agent views
        if eval or render:
            env.render()
    for j, agent in enumerate(agents):
        agent.reset()
    return reward_tot


def train(env, agents, n_episodes=10000, generate_val_data=False):
    logger = logging.getLogger('root')
    scores = []
    times = []
    time_start = time.time()
    if generate_val_data:
        logger.info("Generating val data")
        for _ in range(1):
            score = run_episode(env, agents, eval=False, generate_val_data=True)

    for i in range(n_episodes):
        logger.info("episode:" + str(i))

        if i % 200 == 9990:
            score = run_episode(env, agents, eval=False)
        if i % 2 == 0:
            score = run_episode(env, agents, eval=False, generate_val_data=True)
        score = run_episode(env, agents, render=False)
        scores.append(score)
        t = time.time() - time_start
        times.append(t)
        logger.info("time_elapsed:" + str(t))
        logger.info("score:" + str(score))
    return {"scores": scores, "times": times}


def single_run(env, agent_fn, logdata, seed, agent_kwargs=dict(), n_episodes=50):
    logger = logging.getLogger(__name__)
    name = agent_fn.__name__ + str(agent_kwargs)
    logger.info("agent:" + name)
    env.env.seed(seed=seed)
    if name not in logdata:
        logdata[name] = []
    # env.render()
    obs_n = env.reset()
    agents = []
    for i in range(env.n_agents):
        agents.append(agent_fn(env.observation_space[i].shape[0], env.action_space[i].shape[0], **agent_kwargs))
    logdata[name].append(train(env, agents, n_episodes=n_episodes, generate_val_data=True))
    # env.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', type=str, default="gym", help='Environment Suite to use')
    parser.add_argument('--name', type=str, default="Pendulum-v0", help='Environment name to use')
    parser.add_argument('--alg', type=str, default="ModelAgent", help='Name of alg to use: hddpg or TD3')
    args = parser.parse_args()
    agent_dict = {"HDDPGAgent": HDDPGAgent, "TD3": TD3, "ModelAgent": ModelAgent}

    # Create environment
    #  "HalfCheetahBulletEnv-v0"
    # "ReacherBulletEnv-v0"
    env = EnvWrapper("gym", "Pendulum-v0")

    # execution loop
    n_runs = 5
    logdata = dict()
    logfile = "../logs/test11"
    logging.basicConfig(filename=logfile + ".log", filemode='w', level=logging.DEBUG)
    logger = logging.getLogger('root')
    handler = logging.StreamHandler(sys.stdout)

    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    #    sys.stderr = LoggerWriter(logger.warning)
    try:
        obs_dim = env.observation_space[0].shape[0]
        action_dim = env.action_space[0].shape[0]
        print("Generating training data")
        agents_train = [SAC(env.observation_space[0].shape[0], env.action_space[0].shape[0], use_model=True)]
        agents_test = [SAC(env.observation_space[0].shape[0], env.action_space[0].shape[0], use_model=True)]
        for _ in range(5):
            env.env.seed(12345)
            env.reset()
            run_episode(env, agents_train, eval=False, generate_val_data=True)
        for _ in range(5):
            env.env.seed(888)
            env.reset()
            run_episode(env, agents_test, eval=False, generate_val_data=True)
        buffer_train = agents_train[0].val_buffer
        buffer_test = agents_test[0].val_buffer
        model = EnsembleModel(obs_dim + action_dim,
                                   (200,200,200,200),
                                   obs_dim,
                                   2,
                                   monitor_losses=True,
                                   use_stochastic=False)#.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(),
                         lr=0.001, amsgrad=True, weight_decay=0.)

        for _ in range(25000):
            model.update_step(optimizer, buffer_train.sample_tensors())
            if _%100 == 0:
                model.log_loss(buffer_train.sample_tensors(), "train")
                model.log_loss(buffer_test.sample_tensors(), "test")

    except Exception:
        logger.exception("Fatal error.")