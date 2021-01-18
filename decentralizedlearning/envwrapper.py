"""Contains a wrapper for openAI gym, multi-agent particles and other custom environments to provide constant interface for RL algorithms"""
import sys
import os
import gym
from decentralizedlearning.submodules.multiagent_particle_envs.make_env import make_env
from gym.spaces import Box
import numpy as np
from gym import wrappers

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
        supported_suites = ["gym", "particle", "custom", "gym-record", "multiagent_mujoco", "schroeder"]
        benchmark = bool(kwargs.get("benchmark", False))
        assert suite in supported_suites, "Suite should be in {} but was {}".format(str(supported_suites), suite)
        self.suite = suite
        self.wrapped_env = None
        self.env_name = None
        video_dir_name = str(kwargs.get("video_dir_name", "./logs/videos/"))
        # Setup environment for "gym" suite.
        if suite =="gym-record":
            env = gym.make(env_name)
            self.wrapped_env = env
            self.env = wrappers.Monitor(env, video_dir_name, force=True, video_callable=lambda episode_id: True)
            self.n_agents = 1
            self.observation_space = [self.env.observation_space]
            self.action_space = [self.env.action_space]
        if suite == "gym":
            self.env = gym.make(env_name)
            self.n_agents = 1
            self.observation_space = [self.env.observation_space]
            self.action_space = [self.env.action_space]
            # if self.env.action_space.dtype == dtype('float32'):
            #    self.action_type = "continuous"
        if suite == "custom":
            #namedict = {"waypoints.py": WaypointsEnv, "crossing.py": CrossingEnv, "continuouscrossing.py": ContinuousCrossingEnv}
            namedict = {"waypoints.py": WaypointsEnv, "circle.py": CircleEnv}#, "crossing.py": CrossingEnv, "continuouscrossing.py": ContinuousCrossingEnv}

            self.env = namedict[env_name](**kwargs)
            self.n_agents = self.env.n
            self.action_space = self.env.action_space
            self.observation_space = self.env.observation_space
            print(self.env)
        # Setup environment for "particle" suite.
        if suite=="particle" and env_name=="simple_tag_fixed":
            self.env = make_env(env_name, benchmark=False)
            self.env_name = "simple_tag_fixed"
            self.action_space = self.env.action_space[:-1]
            self.n_agents = len(self.action_space)
            self.observation_space = self.env.observation_space[:-1]
            # Particle suite does not have proper action spaces, hardcoded in here. ->>> wrong. it is just discrete.
            for act_space in self.action_space:
                # act_space.low = np.zeros(8) - 1.0
                # act_space.high = np.zeros(8) + 1.0
                act_space.shape = (act_space.n,)

        elif suite=="particle" or suite=="schroeder":
            if suite=="particle":
                self.env = make_env(env_name, benchmark=benchmark)
            else:
                self.env = make_env_schroeder(env_name, benchmark=False)
            # self.scenario = scenarios.load(env_name).Scenario()
            # self.world = self.scenario.make_world()
            # self.env = MultiAgentEnv(self.world, self.scenario.reset_world,
            #                          self.scenario.reward, self.scenario.observation,
            #                          info_callback=self.scenario.info_callback, shared_viewer=False)
            self.action_space = self.env.action_space
            self.n_agents = len(self.action_space)
            self.observation_space = self.env.observation_space 
            # Particle suite does not have proper action spaces, hardcoded in here. ->>> wrong. it is just discrete.
            # if not env_name in ["simple_tag_coop", "simple_tag_coop_partial_obs"]:
            #     for act_space in self.action_space:
            #         # act_space.low = np.zeros(8) - 1.0
            #         # act_space.high = np.zeros(8) + 1.0
            #         act_space.shape = (act_space.n,)
        # Setup multiagent_mujoco
        if suite== "multiagent_mujoco":
            env_args = {"scenario": "HalfCheetah-v2",
                  "agent_conf": "2x3",
                  "agent_obsk": 1,
                  "episode_limit": 1000}
            self.env = MujocoMulti(env_args=env_args)
            env_info = self.env.get_env_info()
            self.n_agents = env_info["n_agents"]
            self.action_space = env_info["action_spaces"]
            self.observation_space = [Box(shape=(env_info["obs_shape"],), low=float("-inf"), high=float("inf"), dtype=np.float32) for i in range(self.n_agents)]

    def step(self, actions: list):
        """step. Takes a step in the environment given a list of actions, one for
        every agent. Individual actions should be np arrays containing all
        actions

        :param actions: list containing numpy arrays with actions.
        :type actions: list
        """
        if self.suite == "gym" or self.suite == "gym-record":
            return tuple([obj] for obj in self.env.step(actions[0]))
        if self.suite == "multiagent_mujoco":
            reward, terminated, info = self.env.step(actions)
            rewards = [reward for i in range(self.n_agents)]
            observations = self.env.get_obs()
            return observations, rewards, [terminated], info
        if self.env_name == "simple_tag_fixed":
            actions.append(np.array([0.25, 0.25, 0.25, 0.25, 0.25]))
            obs_n, reward_n, done_n, info_n = self.env.step(actions)
            return obs_n[:-1], reward_n[:-1], done_n[:-1], info_n
        return self.env.step(actions)
    
    def reset(self):
        """reset."""
        if self.suite == "gym" or self.suite == "gym-record":
            return [self.env.reset()]
        if self.env_name == "simple_tag_fixed":
            obs_n = self.env.reset()
            return obs_n[:-1]
        return self.env.reset()

    def get_state(self):
        return self.env.sim.get_state()

    def set_state(self, state):
        self.env.sim.set_state(state)

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
        if self.wrapped_env:
            self.wrapped_env.close()
