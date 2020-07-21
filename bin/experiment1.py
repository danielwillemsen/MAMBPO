import pickle as p
import time
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '../decentralizedlearning/submodules/multi-agent-particle-envs'))
sys.path.insert(1, os.path.join(sys.path[0], '../decentralizedlearning'))
sys.path.insert(1, os.path.join(sys.path[0], '../'))

import argparse
import numpy as np
from envwrapper import EnvWrapper

from decentralizedlearning.algs.hddpg import HDDPGAgent
from decentralizedlearning.algs.td3 import TD3
from decentralizedlearning.algs.modelbased import ModelAgent

import matplotlib.pyplot as plt

def scale_action(env, agent_id, action):
    return (env.action_space[agent_id].high-env.action_space[agent_id].low)*action*0.5

def run_episode(env, agents, eval=False,render=False, steps=250):
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
    times = []
    time_start = time.time()
    for i in range(n_episodes):
        print("Episode: " + str(i))
        if i%200 == 9990:
            score = run_episode(env, agents, eval=False)
        else:
            score = run_episode(env, agents, render=False)
        scores.append(score)
        t = time.time() - time_start
        times.append(t)
        print("Time elapsed:", t, " s")
        print("Score: " + str(score))
    return {"scores": scores, "times": times}

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', type=str, default="gym", help='Environment Suite to use')
    parser.add_argument('--name', type=str, default="Pendulum-v0", help='Environment name to use')
    parser.add_argument('--alg', type=str, default="ModelAgent", help='Name of alg to use: hddpg or TD3')
    args = parser.parse_args()
    agent_dict = {"HDDPGAgent": HDDPGAgent, "TD3": TD3, "ModelAgent": ModelAgent}

    # Create environment
    env = EnvWrapper("gym", "LunarLanderContinuous-v2")

    # execution loop
    n_runs = 1
    logdata = dict()

    for run in range(n_runs):
        # agent_fn = ModelAgent
        # agent_kwargs = {"update_every_n_steps": 50, "update_steps": 200, "n_models": 20}
        # name = agent_fn.__name__+ str(agent_kwargs)
        # print(name)
        # if name not in logdata:
        #     logdata[name] = []
        # obs_n = env.reset()
        # agents = []
        # for i in range(env.n_agents):
        #     agents.append(agent_fn(env.observation_space[i].shape[0], env.action_space[i].shape[0], **agent_kwargs))
        # logdata[name].append(train(env, agents, n_episodes=10))
        # p.dump(logdata, open("./logs/pendu3", "wb"))
        # env.close()
        # #
        # agent_fn = ModelAgent
        # agent_kwargs = {"update_every_n_steps": 5, "update_steps": 10, "n_models": 10}
        # name = agent_fn.__name__+ str(agent_kwargs)
        # print(name)
        # env.env.seed(seed=run)
        # if name not in logdata:
        #     logdata[name] = []
        # obs_n = env.reset()
        # agents = []
        # for i in range(env.n_agents):
        #     agents.append(agent_fn(env.observation_space[i].shape[0], env.action_space[i].shape[0], **agent_kwargs))
        # logdata[name].append(train(env, agents, n_episodes=10))
        # p.dump(logdata, open("./logs/pendu4", "wb"))
        # env.close()
        # agent_fn = ModelAgent
        # agent_kwargs = {"update_every_n_steps": 50, "update_steps": 100, "n_models": 20}
        # name = agent_fn.__name__+ str(agent_kwargs)
        # print(name)
        # if name not in logdata:
        #     logdata[name] = []
        # obs_n = env.reset()
        # agents = []
        # for i in range(env.n_agents):
        #     agents.append(agent_fn(env.observation_space[i].shape[0], env.action_space[i].shape[0], **agent_kwargs))
        # logdata[name].append(train(env, agents, n_episodes=10))
        # p.dump(logdata, open("./logs/pendu4", "wb"))
        # env.close()
        logfile = "./logs/pend5"
        #
        agent_fn = HDDPGAgent
        agent_kwargs = {"n_steps": 5}
        name = agent_fn.__name__+ str(agent_kwargs)
        print(name)
        env.env.seed(seed=run)
        if name not in logdata:
            logdata[name] = []
        obs_n = env.reset()
        agents = []
        for i in range(env.n_agents):
            agents.append(agent_fn(env.observation_space[i].shape[0], env.action_space[i].shape[0], **agent_kwargs))
        logdata[name].append(train(env, agents, n_episodes=10))
        p.dump(logdata, open(logfile, "wb"))
        env.close()

        # agent_fn = HDDPGAgent
        # agent_kwargs = {"n_steps": 20}
        # name = agent_fn.__name__+ str(agent_kwargs)
        # print(name)
        # env.env.seed(seed=run)
        # if name not in logdata:
        #     logdata[name] = []
        # obs_n = env.reset()
        # agents = []
        # for i in range(env.n_agents):
        #     agents.append(agent_fn(env.observation_space[i].shape[0], env.action_space[i].shape[0], **agent_kwargs))
        # logdata[name].append(train(env, agents, n_episodes=10))
        # p.dump(logdata, open(logfile, "wb"))
        # env.close()

        # agent_fn = HDDPGAgent
        # agent_kwargs = {"use_model": True, "use_real_model": True, "n_steps": 40}
        # name = agent_fn.__name__+ str(agent_kwargs)
        # print(name)
        # env.env.seed(seed=run)
        # if name not in logdata:
        #     logdata[name] = []
        # obs_n = env.reset()
        # agents = []
        # for i in range(env.n_agents):
        #     agents.append(agent_fn(env.observation_space[i].shape[0], env.action_space[i].shape[0], **agent_kwargs))
        # logdata[name].append(train(env, agents, n_episodes=10))
        # p.dump(logdata, open(logfile, "wb"))
        # env.close()
        #
        # agent_fn = HDDPGAgent
        # agent_kwargs = {"use_model": True, "n_steps": 20}
        # name = agent_fn.__name__+ str(agent_kwargs)
        # print(name)
        # env.env.seed(seed=run)
        # if name not in logdata:
        #     logdata[name] = []
        # obs_n = env.reset()
        # agents = []
        # for i in range(env.n_agents):
        #     agents.append(agent_fn(env.observation_space[i].shape[0], env.action_space[i].shape[0], **agent_kwargs))
        # logdata[name].append(train(env, agents, n_episodes=10))
        # p.dump(logdata, open(logfile, "wb"))
        # env.close()
        # agent_fn = HDDPGAgent
        # agent_kwargs = {"use_model": True, "n_steps": 10, "use_real_model": True}
        # name = agent_fn.__name__+ str(agent_kwargs)
        # print(name)
        # env.env.seed(seed=run)
        #
        # if name not in logdata:
        #     logdata[name] = []
        # obs_n = env.reset()
        # agents = []
        # for i in range(env.n_agents):
        #     agents.append(agent_fn(env.observation_space[i].shape[0], env.action_space[i].shape[0], **agent_kwargs))
        # logdata[name].append(train(env, agents, n_episodes=50))
        # p.dump(logdata, open(logfile, "wb"))
        # env.close()
        # agent_fn = HDDPGAgent
        #
        # agent_kwargs = {"use_model": True, "n_steps": 5}
        # name = agent_fn.__name__+ str(agent_kwargs)
        # print(name)
        # env.env.seed(seed=run)
        #
        # if name not in logdata:
        #     logdata[name] = []
        # obs_n = env.reset()
        # agents = []
        # for i in range(env.n_agents):
        #     agents.append(agent_fn(env.observation_space[i].shape[0], env.action_space[i].shape[0], **agent_kwargs))
        # logdata[name].append(train(env, agents, n_episodes=500))
        # p.dump(logdata, open(logfile, "wb"))
        # env.close()
        #
        # agent_fn = HDDPGAgent
        #
        # agent_kwargs = {"use_model": True, "n_steps": 10}
        # name = agent_fn.__name__+ str(agent_kwargs)
        # print(name)
        # env.env.seed(seed=run)
        #
        # if name not in logdata:
        #     logdata[name] = []
        # obs_n = env.reset()
        # agents = []
        # for i in range(env.n_agents):
        #     agents.append(agent_fn(env.observation_space[i].shape[0], env.action_space[i].shape[0], **agent_kwargs))
        # logdata[name].append(train(env, agents, n_episodes=500))
        # p.dump(logdata, open(logfile, "wb"))
        # env.close()
        # agent_fn = TD3
        # agent_kwargs = {"update_every_n_steps": 1, "update_steps": 2}
        # name = agent_fn.__name__+ str(agent_kwargs)
        # print(name)
        # if name not in logdata:
        #     logdata[name] = []
        # obs_n = env.reset()
        # agents = []
        # for i in range(env.n_agents):
        #     agents.append(agent_fn(env.observation_space[i].shape[0], env.action_space[i].shape[0], **agent_kwargs))
        # logdata[name].append(train(env, agents, n_episodes=50))
        # p.dump(logdata, open("./logs/pendu4", "wb"))
        # env.close()
        # name = "f_hyst=0.5"
        # obs_n = env.reset()
        # agents = []
        # for i in range(env.n_agents):
        #     agents.append(agent_fn(env.observation_space[i].shape[0], env.action_space[i].shape[0]))
        # for agent in agents:
        #     agent.f_hyst = 0.5
        # logdata[name].append(train(env,agents))
        # p.dump(logdata, open("./logs/exp_mult6","wb"))
        #
        # name = "f_hyst=0.25"
        # obs_n = env.reset()
        # agents = []
        # for i in range(env.n_agents):
        #     agents.append(agent_fn(env.observation_space[i].shape[0], env.action_space[i].shape[0]))
        # for agent in agents:
        #     agent.f_hyst = 0.25
        # logdata[name].append(train(env,agents))
        # p.dump(logdata, open("./logs/exp_mult6","wb"))

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
