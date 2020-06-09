"""Contains a wrapper for openAI gym, multi-agent particles and other custom environments to provide constant interface for RL algorithms"""
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], './submodules/multi-agent-particle-envs'))
import gym
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
import numpy as np

class EnvWrapper:
    def __init__(self, suite: str, env_name: str, *args, **kwargs):
        supported_suites = ["gym", "particle"]
        assert suite in supported_suites, "Suite should be in {} but was {}".format(str(supported_suites), suite)
        self.suite = suite
        if suite=="gym":
            self.env = gym.make(env_name)
            self.n_agents = 1
            self.observation_space = [self.env.observation_space]
            self.action_space = [self.env.action_space]
        if suite=="particle":
            self.scenario = scenarios.load(env_name).Scenario()
            self.world = self.scenario.make_world()
            self.env = MultiAgentEnv(self.world, self.scenario.reset_world,
                    self.scenario.reward, self.scenario.observation, 
                    info_callback=None, shared_viewer=False)
            self.action_space = self.env.action_space
            self.n_agents = len(self.action_space)
            for act_space in self.action_space:
                act_space.low = np.zeros(8)
                act_space.high =  np.zeros(8) + 1.0

    def step(self, actions: list):
        """ Takes a step in the environment given a list of actions, one for
        every agent. Individual actions should be np arrays containing all
        actions"""
        if self.suite == "gym":
            return tuple([obj] for obj in self.env.step(actions[0]))
        return self.env.step(actions)
    
    def reset(self):
        if self.suite == "gym":
            return [self.env.reset()]
        return self.env.reset()

    def render(self):
        return self.env.render()
