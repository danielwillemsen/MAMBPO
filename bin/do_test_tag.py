import time
import os, sys

sys.path.insert(1, os.path.join(sys.path[0], '../decentralizedlearning/submodules/multi-agent-particle-envs'))
sys.path.insert(1, os.path.join(sys.path[0], '../decentralizedlearning'))
sys.path.insert(1, os.path.join(sys.path[0], '../'))

import numpy as np
from envwrapper import EnvWrapper
import itertools
from decentralizedlearning.algs.sac import SAC
from decentralizedlearning.algs.configs.config import get_hyperpar
from decentralizedlearning.data_log import DataLog

import logging
from decentralizedlearning.algs.masac import MASAC
from gym import spaces

def scale_action(env, agent_id, action):
    if not isinstance(env.action_space[agent_id], spaces.Discrete):
        return (env.action_space[agent_id].high - env.action_space[agent_id].low) * action * 0.5
    else:
        return action


def run_episode(env, agents, eval=False, render=False, generate_val_data=False, greedy_eval=True, steps=25, store_data=False, trainer=None):
    """ Runs a single training or evaluation episode"""
    obs_n = env.reset()
    reward_tot = [0.0 for i in range(len(agents))]
    reward_n = [0.0 for i in range(len(agents))]
    done_n = [False for i in range(len(agents))]

    if store_data:
        observations = []
        rewards = []
        rewards_details = {"rewards_target": [], "rewards_collision": []}
        actions = []

    # Start env
    for i in range(steps):

        # query for action from each agent's policy
        act_n = []
        if store_data or True:
            observations.append(obs_n)
            actions.append([])
            rewards.append(reward_n)
        # Take actions
        action_list = []
        for j, agent in enumerate(agents):
            action_unscaled = agent.step(obs_n[j], reward_n[j], done=done_n[0], eval=eval,
                                                     generate_val_data=generate_val_data, greedy_eval=greedy_eval)
            action_list.append(action_unscaled)
            if store_data:
                actions[-1].append(action_unscaled)
            action = scale_action(env, j, action_unscaled)
            act_n.append(action)
        if trainer is not None and not eval:
            trainer.step(obs_n, reward_n, action_list, done=done_n)
        # step environment
        obs_n, reward_n, done_n, info_n = env.step(act_n)

        for j, r in enumerate(reward_n):
            reward_tot[j] += r
        if done_n[0]:
            for j, agent in enumerate(agents):
                action = scale_action(env, j, agent.step(obs_n[j], reward_n[j], done=done_n[0], eval=eval,
                                                         generate_val_data=generate_val_data, greedy_eval=greedy_eval))
                act_n.append(action)
                agent.reset()
            break

        # render all agent views
        if render:
            env.render()
    for j, agent in enumerate(agents):
        agent.reset()
    if trainer is not None:
        trainer.reset()
    if not store_data:
        return reward_tot, i + 1
    else:
        extra_data = {"observations": observations, "actions": actions, "rewards": rewards, "rewards_details": rewards_details}
        return reward_tot, i + 1, extra_data


def train(env, agents, data_log, n_episodes=10000, n_steps=None, generate_val_data=False, record_env=None, trainer=None):
    """ Performs a single training run"""
    # Setup logging and start code
    logger = logging.getLogger('root')
    step_tot = 0
    logger.info(env.observation_space[0].high)
    alphas = [agent.alpha for agent in trainer.agents]
    data_log.log_var("alphas", alphas)

    ep_generator = range(n_episodes) if n_episodes else itertools.count()
    # Start training
    for i in ep_generator:
        # Do some logging
        logger.info("episode:" + str(i))
        data_log.set_episode(i)

        # Periodically store networks
        if i % 250 == 0: #was 25
            store_networks(trainer, agents, data_log)

        # Run a single episode
        score, step, extra_data = run_episode(env, agents, render=False, store_data=True, trainer=trainer)

        # Do more logging
        logger.info("Score: " + str(score))
        step_tot += step
        data_log.set_step(step_tot)
        data_log.log_var("score", score)
        alphas = [agent.alpha for agent in trainer.agents]
        data_log.log_var("alphas", alphas)

        # Break training loop
        if n_steps and step_tot > n_steps:
            break

        #Periodically save logs
        if i % 50 == 0: #was 5
            logger.info("Saving log...")
            data_log.save()
            logger.info("Saved log")

    # Save logs one last time
    logger.info("Saving log...")
    data_log.save()
    logger.info("Saved log")
    return


def store_networks(trainer, agents, data_log):
    """ Store networks of all agents in data log

    :param trainer: Trainer with model
    :param agents: Agents
    :param data_log: Data log to store model in
    :return:
    """
    networks = []
    for agent in agents:
        if agent.model:
            networks.append({"actor": agent.actor.state_dict(),
                             "critics": [critic.state_dict() for critic in agent.critics],
                             "model": agent.model.state_dict()})
        else:
            networks.append({"actor": agent.actor.state_dict(),
                             "critics": [critic.state_dict() for critic in agent.critics],
                             "model": None})
    if trainer.model:
        data_log.log_var("networks_model", trainer.model.state_dict())
    data_log.log_var("networks_agents", networks)



def single_run(env, trainer_fn, data_log, seed, agent_kwargs=dict(), n_steps=None, record_env=None, name=None):
    """ Setup a single run and perform the run"""

    # Setup logging
    logger = logging.getLogger(__name__)
    if not name:
        name = trainer_fn.__name__ + str(agent_kwargs)
    logger.info("agent:" + name)
    data_log.init_run(name)

    # Initialize Agents and environments
    env.reset()
    trainer = trainer_fn(env.n_agents, env.observation_space, env.action_space, **agent_kwargs)
    agents = trainer.agents
    if agents[0].par.use_shared_replay_buffer:
        for agent in agents:
            agent.set_replay_buffer(agents[0].real_buffer)

    # Perform training
    train(env, agents, data_log, n_steps=n_steps, generate_val_data=True, record_env=record_env, trainer=trainer)


if __name__ == '__main__':
    """Main script to run experiments"""
    name = "simple_tag_coop"     # Environment name, select from: "HalfCheetah-v2", "simple_tag_coop", "simple_spread"
    n_runs = 5                 # Amount of runs to do
    logpath = "./logs/"         # Logging directory
    logname = "tag_masac_long"  # Name of log file
    config_name = "default"
    n_steps = 25*20001
    # Setup logging (to .log file)
    logfile = logpath + logname
    logging.basicConfig(filename=logpath + logname + ".log", filemode='w', level=logging.DEBUG)
    logger = logging.getLogger('root')
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    # Setup logging (to .p file)
    data_log = DataLog(logpath, logname)
    logdata = dict()

    # Further environment Setup
    suite = "gym" if name == "HalfCheetah-v2" else "particle"

    for run in range(n_runs):
        logger.info("run:" + str(run))
        agent_fn = SAC
        for n_agent in [4]:
            env = EnvWrapper(suite, name, n_agents=n_agent, randomized=False)
            algname = "SAC"
            par = get_hyperpar(config_name)
            record_env = EnvWrapper(suite, name, n_agents=n_agent, randomized=True)
            single_run(env, MASAC, data_log, run, agent_kwargs={"hyperpar": par}, n_steps=n_steps,
                       record_env=record_env, name=name+config_name+logname)