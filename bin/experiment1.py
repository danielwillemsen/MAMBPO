import pickle as p

import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '../decentralizedlearning/submodules/multi-agent-particle-envs'))
sys.path.insert(1, os.path.join(sys.path[0], '../decentralizedlearning'))

import argparse
import numpy as np
from envwrapper import EnvWrapper

from hddpg import HDDPGAgent
import matplotlib.pyplot as plt

def scale_action(env, agent_id, action):
    return (env.action_space[agent_id].high-env.action_space[agent_id].low)*action + env.action_space[agent_id].low

def run_episode(env, agents, eval=False, steps=200):
    obs_n = env.reset()
    reward_tot = 0.0
    reward_n = [0.0]
    
    #Start env
    for i in range(steps):
        # query for action from each agent's policy
        act_n = []
        for j, agent in enumerate(agents):
            action = scale_action(env, j, agent.step(obs_n[j], reward_n, eval=False))
            act_n.append(action)
        # step environment
        obs_n, reward_n, done_n, _ = env.step(act_n)
        reward_tot += reward_n[0]
        # render all agent views
        if eval:
            env.render()
    for i, agent in enumerate(agents):
        agent.reset()
    return reward_tot / steps

def train(env, agents, n_episodes=1000):
    scores = []
    for i in range(n_episodes):
        print("Episode: " + str(i))
        if i%1 == 0:
            score = run_episode(env, agents, eval=True)
        else:
            score = run_episode(env, agents)
        scores.append(score)
        print("Score: " + str(score))
    return scores

if __name__ == '__main__':
    env = EnvWrapper("gym", "Pendulum-v0")

    # execution loop
    n_runs = 10
    logdata = dict()
    logdata["f_hyst=1.0"] = []
    logdata["f_hyst=0.5"] = []
    logdata["f_hyst=0.25"] = []

    for run in range(n_runs):
        name = "f_hyst=1.0"
        obs_n = env.reset()
        agents = []
        for i in range(env.n_agents):
            agents.append(HDDPGAgent(env.observation_space[i].shape[0], env.action_space[i].shape[0]))
        for agent in agents:
            agent.f_hyst = 1.0
        logdata[name].append(train(env,agents))
        p.dump(logdata, open("./logs/exp_mult6","wb"))

        name = "f_hyst=0.5"
        obs_n = env.reset()
        agents = []
        for i in range(env.n_agents):
            agents.append(HDDPGAgent(env.observation_space[i].shape[0], env.action_space[i].shape[0]))
        for agent in agents:
            agent.f_hyst = 0.5
        logdata[name].append(train(env,agents))
        p.dump(logdata, open("./logs/exp_mult6","wb"))

        name = "f_hyst=0.25"
        obs_n = env.reset()
        agents = []
        for i in range(env.n_agents):
            agents.append(HDDPGAgent(env.observation_space[i].shape[0], env.action_space[i].shape[0]))
        for agent in agents:
            agent.f_hyst = 0.25
        logdata[name].append(train(env,agents))
        p.dump(logdata, open("./logs/exp_mult6","wb"))

    # name = "f_hyst=0.5"
    # run_data = []
    # for run in range(n_runs):
    #     obs_n = env.reset()
    #     agents = [HDDPGAgent(len(obs_n[0]), 8), HDDPGAgent(len(obs_n[0]), 8)]  # , HDDPGAgent(len(obs_n[0]),8)]#, HDDPGAgent(len(obs_n[0]),8), HDDPGAgent(len(obs_n[0]),8)]
    #     agents[0].f_hyst = 0.5
    #     run_data.append(train(env,agents))
    #
    # for data in run_data:
    #     plt.plot(data)
    # plt.show()
    # logdata[name] = run_data
