#!/usr/bin/env python
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '../decentralizedlearning/submodules/multi-agent-particle-envs'))
sys.path.insert(1, os.path.join(sys.path[0], '../decentralizedlearning'))

import argparse

from multiagent.environment import MultiAgentEnv
from multiagent.policy import InteractivePolicy
import multiagent.scenarios as scenarios
from hddpg import HDDPGAgent

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
    # render call to create viewer window (necessary only for interactive policies)
    #env.render()
    # create interactive policies for each agent
    obs_n = env.reset()

    agents = [HDDPGAgent(len(obs_n[0]),8), HDDPGAgent(len(obs_n[0]),8), HDDPGAgent(len(obs_n[0]),8)]
    # execution loop
    episode = 0
    while True:
        episode += 1
        print(episode)
        obs_n = env.reset()
        reward_n = [0.0]
        for i in range(250):
            # query for action from each agent's policy
            act_n = []
            for j, agent in enumerate(agents):
                act_n.append(agent.step(obs_n[j], reward_n))
            # step environment
            obs_n, reward_n, done_n, _ = env.step(act_n)
            if i%10 ==0:
                print(reward_n)
            # render all agent views
            if episode %10 ==0:
                env.render()
            # display rewards
            #for agent in env.world.agents:
            #    print(agent.name + " reward: %0.3f" % env._get_reward(agent))
        for i, agent in enumerate(agents):
            act_n.append(agent.reset())