import time
import pickle
import torch
import copy

class DataLog:
    """ DataLog class that allows logging arbitrary variables over multiple runs, multiple agents"""
    def __init__(self, path, name):
        self.current_agent = None
        self.current_episode = None
        self.current_step = None
        self.current_time = None
        self.start_time = time.time()
        self.data = dict()
        self.current_data = {}
        self.path = path
        self.name = name
        self.networks = [{}]


    def set_agent(self, name, hyperpar=None):
        self.current_agent = name
        if name not in self.data.keys():
            self.data[name] = {"hyperpar": hyperpar, "runs": []}
        else:
            assert hyperpar == self.data[name]["hyperpar"], "Agent Hyperpars have changed"

    def init_run(self, name, hyperpar=None, networks=[{}]):
        self.set_agent(name, hyperpar=hyperpar)
        self.current_episode = 0
        self.current_step = 0
        self.start_time = time.time()

        self.networks = networks

        self.data[self.current_agent]["runs"].append({})
        self.current_data = self.data[self.current_agent]["runs"][-1]


    def set_episode(self, ep):
        self.current_episode = ep

    def set_step(self, step):
        self.current_step = step

    def log_var(self, name, value):
        self.current_time = time.time() - self.start_time
        if name not in self.current_data.keys():
            self.current_data[name] = []

        self.current_data[name].append((self.current_episode,
                                        self.current_step,
                                        self.current_time,
                                        copy.deepcopy(value)))

    def save(self):
        torch.save(self.data, self.path + self.name + ".p")

