import gym

import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '../decentralizedlearning/submodules/multi-agent-particle-envs'))
sys.path.insert(1, os.path.join(sys.path[0], '../decentralizedlearning'))

import argparse

from multiagent.environment import MultiAgentEnv
from multiagent.policy import InteractivePolicy
import multiagent.scenarios as scenarios
from hddpg import HDDPGAgent
import numpy as np
import torch
import matplotlib.pyplot as plt

env = gym.make('Pendulum-v0')
print(env.observation_space)
print(env.action_space)
actor = HDDPGAgent(3, 1)
reward = 0
for i_episode in range(2000):
    observation = env.reset()
    total_reward = 0
    for t in range(1000):
        env.render()
        #print(observation)
        action = actor.step(observation, [reward])
        observation, reward, done, info = env.step([action*4 - 2.0])
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            actor.reset()
            break
        total_reward += reward
    thetas = np.arange(-3.14, 3.14, 0.1)
    acts = np.arange(0., 1.0, 0.1)
    Qs = []
    Qs = np.zeros((len(thetas),(len(acts))))
    # for i, th in enumerate(thetas):
    #     for j, a in enumerate(acts):
    #         Q = actor.ac.critic(torch.Tensor(np.array([[np.cos(th),np.sin(th),0]])),torch.Tensor(np.array([[a]]))).detach().numpy()
    #         Qs[i,j] = float(Q.squeeze())
    # plt.imshow(Qs)
    # plt.show()
    # for i, th in enumerate(thetas):
    #     for j, a in enumerate(acts):
    #         Q = actor.ac_target.critic(torch.Tensor(np.array([[np.cos(th),np.sin(th),0]])),torch.Tensor(np.array([[a]]))).detach().numpy()
    #         Qs[i,j] = float(Q.squeeze())
    # plt.imshow(Qs)
    # plt.show()
    print("Episode: " + str(i_episode) + " --- R: " + str(total_reward))

env.close()
