"""Contains a wrapper for openAI gym, multi-agent particles and other custom environments to provide constant interface for RL algorithms"""
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], './submodules/multi-agent-particle-envs'))
import gym
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
import numpy as np

class EnvWrapper:
    """EnvWrapper."""

    def __init__(self, suite: str, env_name: str, *args, **kwargs):
        """__init__.
        Creates gym or particle environments that use the same interfacing
        regardless of type of environment

        :param suite: suite to use: "gym" or "particle" for now.
        :type suite: str
        :param env_name: name of env to load in the suite. Example:
            "Pendulum-v0" or "simple_spread.py" 
        :type env_name: str
        :param args:
        :param kwargs:
        """
        # Check if suite name is correct and can be handled
        supported_suites = ["gym", "particle"]
        assert suite in supported_suites, "Suite should be in {} but was {}".format(str(supported_suites), suite)
        self.suite = suite
        
        # Setup environment for "gym" suite.
        if suite=="gym":
            self.env = gym.make(env_name)
            self.n_agents = 1
            self.observation_space = [self.env.observation_space]
            self.action_space = [self.env.action_space]

        # Setup environment for "particle" suite.
        if suite=="particle":
            self.scenario = scenarios.load(env_name).Scenario()
            self.world = self.scenario.make_world()
            self.env = MultiAgentEnv(self.world, self.scenario.reset_world,
                                     self.scenario.reward, self.scenario.observation, 
                                     info_callback=None, shared_viewer=False)
            self.action_space = self.env.action_space
            self.n_agents = len(self.action_space)
            
            # Particle suite does not have proper action spaces, hardcoded in here. 
            for act_space in self.action_space:
                act_space.low = np.zeros(8)
                act_space.high =  np.zeros(8) + 1.0

    def step(self, actions: list):
        """step. Takes a step in the environment given a list of actions, one for
        every agent. Individual actions should be np arrays containing all
        actions

        :param actions: list containing numpy arrays with actions.
        :type actions: list
        """
        if self.suite == "gym":
            return tuple([obj] for obj in self.env.step(actions[0]))
        return self.env.step(actions)
    
    def reset(self):
        """reset."""
        if self.suite == "gym":
            return [self.env.reset()]
        return self.env.reset()

    def render(self):
        return self.env.render()
