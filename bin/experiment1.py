import pickle as p

import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '../decentralizedlearning/submodules/multi-agent-particle-envs'))
sys.path.insert(1, os.path.join(sys.path[0], '../decentralizedlearning'))

import argparse
import numpy as np
from envwrapper import EnvWrapper

from decentralizedlearning.algs.hddpg import HDDPGAgent
from decentralizedlearning.algs.td3 import TD3
import matplotlib.pyplot as plt

def scale_action(env, agent_id, action):
    return (env.action_space[agent_id].high-env.action_space[agent_id].low)*action + env.action_space[agent_id].low

def run_episode(env, agents, eval=False,render=False, steps=100):
    obs_n = env.reset()
    reward_tot = [0.0 for i in range(len(agents))]
    reward_n = [0.0 for i in range(len(agents))]
    
    #Start env
    for i in range(steps):
        # query for action from each agent's policy
        act_n = []
        for j, agent in enumerate(agents):
            action = scale_action(env, j, agent.step(obs_n[j], reward_n[j], eval=eval))
            act_n.append(action)
        # step environment
        obs_n, reward_n, done_n, _ = env.step(act_n)
        
        for j, r in enumerate(reward_n):
            reward_tot[j] += r
        if done_n[0]:
            for j, agent in enumerate(agents):
                print(reward_n)
                action = scale_action(env, j, agent.step(obs_n[j], reward_n[j], eval=eval))
                act_n.append(action)
                agent.reset()
            print("Episode finished after {} timesteps".format(i+1))
            break
        # render all agent views
        if eval or render:
            env.render()
    for j, agent in enumerate(agents):
        agent.reset()
    return reward_tot 

def train(env, agents, n_episodes=10000):
    scores = []
    for i in range(n_episodes):
        print("Episode: " + str(i))
        if i%10 == 0:
            score = run_episode(env, agents, eval=True)
        elif i%5 ==0:
            score = run_episode(env, agents, render=True)
        else:
            score = run_episode(env, agents)
        scores.append(score)
        print("Score: " + str(score))
    return scores

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', type=str, default="gym", help='Environment Suite to use')
    parser.add_argument('--name', type=str, default="Pendulum-v0", help='Environment name to use')
    parser.add_argument('--alg', type=str, default="HDDPGAgent", help='Name of alg to use: hddpg or TD3')
    args = parser.parse_args()
    agent_dict = {"HDDPGAgent": HDDPGAgent, "TD3": TD3}

    # Checks for valid argument
    assert args.alg in agent_dict.keys(), "Invalid algorithm selected: {}. Available are: {}".format(args.alg, str(agent_dict.keys()))
    agent_fn = agent_dict[args.alg] 
    
    # Create environment
    env = EnvWrapper(args.suite, args.name)

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
            agents.append(agent_fn(env.observation_space[i].shape[0], env.action_space[i].shape[0]))
        for agent in agents:
            agent.f_hyst = 1.0
        logdata[name].append(train(env,agents))
        p.dump(logdata, open("./logs/exp_mult6","wb"))

        name = "f_hyst=0.5"
        obs_n = env.reset()
        agents = []
        for i in range(env.n_agents):
            agents.append(agent_fn(env.observation_space[i].shape[0], env.action_space[i].shape[0]))
        for agent in agents:
            agent.f_hyst = 0.5
        logdata[name].append(train(env,agents))
        p.dump(logdata, open("./logs/exp_mult6","wb"))

        name = "f_hyst=0.25"
        obs_n = env.reset()
        agents = []
        for i in range(env.n_agents):
            agents.append(agent_fn(env.observation_space[i].shape[0], env.action_space[i].shape[0]))
        for agent in agents:
            agent.f_hyst = 0.25
        logdata[name].append(train(env,agents))
        p.dump(logdata, open("./logs/exp_mult6","wb"))

    # name = "f_hyst=0.5"
    # run_data = []
    # for run in range(n_runs):
    #     obs_n = env.reset()
    #     agents = [agent_fn(len(obs_n[0]), 8), HDDPGAgent(len(obs_n[0]), 8)]  # , HDDPGAgent(len(obs_n[0]),8)]#, HDDPGAgent(len(obs_n[0]),8), HDDPGAgent(len(obs_n[0]),8)]
    #     agents[0].f_hyst = 0.5
    #     run_data.append(train(env,agents))
    #
    # for data in run_data:
    #     plt.plot(data)
    # plt.show()
    # logdata[name] = run_data
