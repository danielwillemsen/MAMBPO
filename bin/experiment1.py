import pickle as p
import time
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '../decentralizedlearning/submodules/multi-agent-particle-envs'))
sys.path.insert(1, os.path.join(sys.path[0], '../decentralizedlearning'))
sys.path.insert(1, os.path.join(sys.path[0], '../'))
import pybullet_envs
import argparse
import numpy as np
from envwrapper import EnvWrapper
import itertools
from decentralizedlearning.algs.hddpg import HDDPGAgent
from decentralizedlearning.algs.td3 import TD3
from decentralizedlearning.algs.modelbased import ModelAgent
from decentralizedlearning.algs.sac import SAC

import logging
import matplotlib.pyplot as plt
class LoggerWriter:
    def __init__(self, level):
        # self.level is really like using log.debug(message)
        # at least in my case
        self.level = level

    def write(self, message):
        # if statement reduces the amount of newlines that are
        # printed to the logger
        if message != '\n':
            self.level(message)

    def flush(self):
        # create a flush method so things can be flushed when
        # the system wants to. Not sure if simply 'printing'
        # sys.stderr is the correct way to do it, but it seemed
        # to work properly for me.
        self.level(sys.stderr)
def scale_action(env, agent_id, action):
    return (env.action_space[agent_id].high-env.action_space[agent_id].low)*action*0.5

def run_episode(env, agents, eval=False,render=False, generate_val_data=False, steps=1000):
    obs_n = env.reset()
    reward_tot = [0.0 for i in range(len(agents))]
    reward_n = [0.0 for i in range(len(agents))]
    #Start env
    for i in range(steps):
        # query for action from each agent's policy
        act_n = []
        for j, agent in enumerate(agents):
            action = scale_action(env, j, agent.step(obs_n[j], reward_n[j], eval=eval, generate_val_data=generate_val_data))
            act_n.append(action)
        # step environment
        obs_n, reward_n, done_n, _ = env.step(act_n)
        
        for j, r in enumerate(reward_n):
            reward_tot[j] += r
        if done_n[0]:
            for j, agent in enumerate(agents):
                action = scale_action(env, j, agent.step(obs_n[j], reward_n[j], eval=eval, generate_val_data=generate_val_data))
                act_n.append(action)
                agent.reset()
            print("Episode finished after {} timesteps".format(i+1))
            break
        # render all agent views
        if eval or render:
            env.render()
    for j, agent in enumerate(agents):
        agent.reset()
    return reward_tot, i+1

def train(env, agents, n_episodes=10000, n_steps=None, generate_val_data=False):
    logger = logging.getLogger('root')
    scores = []
    scores_eval = []
    times = []
    time_start = time.time()
    step_tot = 0
    steps = []
    ep_generator = range(n_episodes) if n_episodes else itertools.count()
    if generate_val_data:
        logger.info("Generating val data")
        while len(agents[0].val_buffer) < agents[0].val_buffer.n_samples:
            score, _ = run_episode(env, agents, eval=False, generate_val_data=True)

    for i in ep_generator:
        logger.info("episode:"+str(i))
        if i%2 == 0:
            score, _ = run_episode(env, agents, eval=False, generate_val_data=True)
        score, step = run_episode(env, agents, render=False)
        # score_eval, _ = run_episode(env, agents, eval=True)
        scores.append(score)
        # scores_eval.append(score_eval)
        t = time.time() - time_start
        times.append(t)
        logger.info("time_elapsed:"+str(t))
        logger.info("score:"+str(score))
        step_tot += step
        steps.append(step_tot)
        logger.info("step_tot:"+str(step_tot))
        if n_steps and step_tot > n_steps:
            break
        # logger.info("score_eval:"+str(score_eval))
    return {"scores": scores, "steps": steps, "scores_eval": scores_eval, "times": times}

def single_run(env, agent_fn, logdata, seed, agent_kwargs=dict(), n_episodes=None, n_steps=None):
    logger = logging.getLogger(__name__)
    name = agent_fn.__name__ + str(agent_kwargs)
    logger.info("agent:"+name)
    env.env.seed(seed=seed)
    if name not in logdata:
        logdata[name] = []
    #env.render()
    obs_n = env.reset()
    agents = []
    for i in range(env.n_agents):
        agents.append(agent_fn(env.observation_space[i].shape[0], env.action_space[i].shape[0], **agent_kwargs))
    logdata[name].append(train(env, agents, n_episodes=n_episodes, n_steps=n_steps, generate_val_data=True))
    #env.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', type=str, default="gym", help='Environment Suite to use')
    parser.add_argument('--name', type=str, default="Pendulum-v0", help='Environment name to use')
    parser.add_argument('--alg', type=str, default="ModelAgent", help='Name of alg to use: hddpg or TD3')
    args = parser.parse_args()
    agent_dict = {"HDDPGAgent": HDDPGAgent, "TD3": TD3, "ModelAgent": ModelAgent}

    # Create environment
    #  "HalfCheetahBulletEnv-v0"
    # "ReacherBulletEnv-v0"
    env = EnvWrapper("gym", "HalfCheetahBulletEnv-v0")
    # env.env.render()
    # execution loop
    n_runs = 1
    logdata = dict()
    logfile = "./logs/test"
    logging.basicConfig(filename=logfile+".log", filemode='w', level=logging.DEBUG)
    logger = logging.getLogger('root')
    handler = logging.StreamHandler(sys.stdout)

    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
#    sys.stderr = LoggerWriter(logger.warning)
#     try:
    if True:
        for run in range(n_runs):
            logger.info("run:"+str(run))
            agent_fn = SAC
            for steps in [5]:

                #
                # agent_kwargs = {"n_steps": steps, "use_model": False}
                # single_run(env, agent_fn, logdata, run, agent_kwargs=agent_kwargs, n_steps=50000)
                # p.dump(logdata, open(logfile, "wb"))


                agent_kwargs = {"n_steps": steps, "use_model": True, "use_model_stochastic": True}
                single_run(env, agent_fn, logdata, run, agent_kwargs=agent_kwargs, n_steps=999)
                p.dump(logdata, open(logfile, "wb"))



                # agent_kwargs = {"n_steps": steps, "use_model": True, "diverse": False}
                # single_run(env, agent_fn, logdata, run, agent_kwargs=agent_kwargs, n_episodes=250)
                # p.dump(logdata, open(logfile, "wb"))
                #
                # agent_kwargs = {"n_steps": steps, "use_model": True}
                # single_run(env, agent_fn, logdata, run, agent_kwargs=agent_kwargs, n_episodes=50)
                # p.dump(logdata, open(logfile, "wb"))

    # except Exception:
    #     logger.exception("Fatal error.")