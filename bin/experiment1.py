import pickle as p

import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '../decentralizedlearning/submodules/multi-agent-particle-envs'))
sys.path.insert(1, os.path.join(sys.path[0], '../decentralizedlearning'))

import argparse
import numpy as np
from multiagent.environment import MultiAgentEnv
from multiagent.policy import InteractivePolicy
import multiagent.scenarios as scenarios
from hddpg import HDDPGAgent
import matplotlib.pyplot as plt

def run_episode(env, agents, eval=False, steps=50):
    obs_n = env.reset()
    reward_tot = 0.0
    reward_n = [0.0]
    #Start env
    for i in range(steps):
        # query for action from each agent's policy
        act_n = []
        for j, agent in enumerate(agents):
            act_n.append(agent.step(obs_n[j], reward_n, eval=eval))
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
        if i%20 == 0:
            score = run_episode(env, agents, eval=True)
        else:
            score = run_episode(env, agents)
        scores.append(score)
        print("Score: " + str(score))
    return scores

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='simple_spread.py', help='Path of the scenario Python script.')
    args = parser.parse_args()

    # load scenario from script
    scenario = scenarios.load(args.scenario).Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer = False)

    # execution loop
    n_runs = 10
    logdata = dict()
    logdata["f_hyst=1.0"] = []
    logdata["f_hyst=0.5"] = []
    logdata["f_hyst=0.25"] = []

    for run in range(n_runs):
        name = "f_hyst=0.25"
        obs_n = env.reset()
        agents = []
        for i in range(3):
            agents.append(HDDPGAgent(len(obs_n[0]), 8))
        for agent in agents:
            agent.f_hyst = 0.25
        logdata[name].append(train(env,agents))
        p.dump(logdata, open("../logs/exp_mult6","wb"))

        name = "f_hyst=1.0"
        obs_n = env.reset()
        agents = []
        for i in range(3):
            agents.append(HDDPGAgent(len(obs_n[0]), 8))
        for agent in agents:
            agent.f_hyst = 1.0
        logdata[name].append(train(env,agents))
        p.dump(logdata, open("../logs/exp_mult6","wb"))

        name = "f_hyst=0.5"
        obs_n = env.reset()
        agents = []
        for i in range(3):
            agents.append(HDDPGAgent(len(obs_n[0]), 8))
        for agent in agents:
            agent.f_hyst = 0.5
        logdata[name].append(train(env,agents))
        p.dump(logdata, open("../logs/exp_mult6","wb"))


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
