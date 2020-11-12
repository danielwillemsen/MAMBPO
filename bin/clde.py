import pickle as p
import time
import os, sys

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
from decentralizedlearning.algs.configs.config_cheetah import get_hyperpar
from decentralizedlearning.data_log import DataLog

import logging
import matplotlib.pyplot as plt
from decentralizedlearning.algs.masac import MASAC
from decentralizedlearning.algs.maddpg import MADDPG
from gym import spaces

def scale_action(env, agent_id, action):
    if not isinstance(env.action_space[agent_id], spaces.Discrete):
        return (env.action_space[agent_id].high - env.action_space[agent_id].low) * action * 0.5
    else:
        return action


def run_episode(env, agents, eval=False, render=False, generate_val_data=False, greedy_eval=True, steps=25, store_data=False, trainer=None):
    obs_n = env.reset()
    reward_tot = [0.0 for i in range(len(agents))]
    reward_n = [0.0 for i in range(len(agents))]
    done_n = [False for i in range(len(agents))]
    rewards_target = [0.0 for i in range(len(agents))]
    rewards_collision = [0.0 for i in range(len(agents))]

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
            rewards_details["rewards_target"].append(rewards_target)
            rewards_details["rewards_collision"].append(rewards_collision)

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
        obs_n, reward_n, done_n, _, _ = env.step(act_n)
        rewards_collision = []#reward_details["rewards_collision"]
        rewards_target = []#reward_details["rewards_target"]
        for j, r in enumerate(reward_n):
            reward_tot[j] += r
        if done_n[0]:
            for j, agent in enumerate(agents):
                action = scale_action(env, j, agent.step(obs_n[j], reward_n[j], done=done_n[0], eval=eval,
                                                         generate_val_data=generate_val_data, greedy_eval=greedy_eval))
                act_n.append(action)
                agent.reset()
            # print("Episode finished after {} timesteps".format(i + 1))
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
    logger = logging.getLogger('root')
    scores = []
    scores_eval = []
    times = []
    time_start = time.time()
    step_tot = 0
    steps = []
    ep_generator = range(n_episodes) if n_episodes else itertools.count()
    logger.info(env.env.observation_space[0].high)

    # if generate_val_data:
    #     logger.info("Generating val data")
    #     while len(agents[0].val_buffer) < agents[0].val_buffer.n_samples:
    #         score, _ = run_episode(env, agents, eval=False, generate_val_data=True, trainer=trainer)
    scores_list = []

    for i in ep_generator:
        data_log.set_episode(i)
        # logger.info("episode:" + str(i))
        # if i % 20 == 0:
        #     generate_statistics(agents, record_env, data_log)
        # Sometimes generate val data
        # if i % 2 == 0:
        #     run_episode(env, agents, eval=False, generate_val_data=True)
        score, step, extra_data = run_episode(env, agents, render=False, store_data=True, trainer=trainer)
        step_tot += step
        data_log.set_step(step_tot)
        data_log.log_var("score", score)
        rewards_collision = [sum(dat) for dat in zip(*extra_data["rewards_details"]["rewards_collision"])]
        rewards_target = [sum(dat) for dat in zip(*extra_data["rewards_details"]["rewards_target"])]
        #
        # data_log.log_var("score_collision", rewards_collision)
        # data_log.log_var("score_target", rewards_target)
        #
        # score2, _ = run_episode(env, agents, eval=True, render=False)
        # data_log.log_var("score_greedy", score2)

        # score_eval, _ = run_episode(env, agents, eval=True)
        scores.append(score)
        scores_list.append(np.mean(score))
        # scores_eval.append(score_eval)
        t = time.time() - time_start
        times.append(t)
        if i%50 == 49:
            logger.info("episode:" + str(i))

            score, step, extra_data = run_episode(env, agents, render=False, store_data=True, trainer=trainer)
            scores_greedy = []
            for e in range(10):
                score_greedy, _, _ = run_episode(env, agents, render=False, eval=True, store_data=True, trainer=trainer)
                scores_greedy.append(np.mean(score_greedy))

            print("Mean score: " + str(np.mean(scores_list)))
            print("Mean score (greedy): " + str(np.mean(scores_greedy)))


            scores_list = []
            logger.info("time_elapsed:" + str(t))
            logger.info("score:" + str(score))
            logger.info("score_collision: " + str(rewards_collision))
            logger.info("score_target: " + str(rewards_target))

            # logger.info("score_greedy:" + str(score2))
            logger.info("step_tot:" + str(step_tot))

        steps.append(step_tot)
        if n_steps and step_tot > n_steps:
            break
        # logger.info("score_eval:"+str(score_eval))
        if i % 1000 == 0:
            logger.info("Saving log...")
            data_log.save()
            logger.info("Saved log")
    logger.info("Saving log...")
    data_log.save()
    logger.info("Saved log")
    return {"scores": scores, "steps": steps, "scores_eval": scores_eval, "times": times}


def generate_statistics(agents, record_env, data_log):
    if record_env is not None:
        score, _, statistics = run_episode(record_env, agents, eval=True, render=False, greedy_eval=False, store_data=True)

    # Test Model
    # if agents[0].model:
    #     rewards = statistics["rewards"]
    #     observations = statistics["observations"]
    #     actions = statistics["actions"]
    #     observation = observations[0][0]
    #     rews_real = []
    #     rews_pred = []
    #     for step in range(1000):
    #         rews_real.append(rewards[step][0])
    #         logger.info("stat_rew_real:" + str(rewards[step][0]))
    #         action = actions[step][0]
    #         observation, rew_predict = agents[0].model.step_single(observation, action)
    #         logger.info("stat_rew_predict:" + str(rew_predict[0]))
    #         rews_pred.append(rew_predict[0])
    #     data_log.log_var("model_vis", {"real": rews_real, "pred": rews_pred})

    # Store model, policy and actor
    networks = []
    # for agent in agents:
    #     if agent.model:
    #         networks.append({"actor": agent.actor.state_dict(),
    #                          "critics": [critic.state_dict() for critic in agent.critics],
    #                          "model": agent.model.state_dict()})
    #     else:
    #         networks.append({"actor": agent.actor.state_dict(),
    #                          "critics": [critic.state_dict() for critic in agent.critics],
    #                          "model": None})
    data_log.log_var("networks", networks)



def single_run(env, agent_fn, logdata, data_log, seed, agent_kwargs=dict(), n_episodes=None, n_steps=None, record_env=None, name=None, trainer_fn=None):
    logger = logging.getLogger(__name__)
    if not name:
        name = agent_fn.__name__ + str(agent_kwargs)
    logger.info("agent:" + name)


    env.env.seed(seed=seed)
    if name not in logdata:
        logdata[name] = []
    # env.render()
    obs_n = env.reset()

    if trainer_fn:
        trainer = trainer_fn(env.n_agents, env.observation_space[0].shape[0], env.action_space[0].shape[0], **agent_kwargs)
        agents = trainer.agents
    else:
        trainer = None
        agents = []
        for i in range(env.n_agents):
            agents.append(agent_fn(env.observation_space[i].shape[0], env.action_space[i].shape[0], **agent_kwargs))

    data_log.init_run(name)

    logdata[name].append(
        train(env, agents, data_log, n_episodes=n_episodes, n_steps=n_steps, generate_val_data=True, record_env=record_env, trainer=trainer))
    # env.close()


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
    name = "waypoints.py"

    # env = EnvWrapper("custom", name)
    # env.env.render()
    # execution loop
    n_runs = 3
    logdata = dict()
    logpath = "../logs/"
    logname = "clde_test2"
    logfile = logpath + logname

    logging.basicConfig(filename=logpath + logname + ".log", filemode='w', level=logging.DEBUG)
    logger = logging.getLogger('root')
    handler = logging.StreamHandler(sys.stdout)

    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    # name2 = "HalfCheetah-v2"

    data_log = DataLog(logpath, logname)

    if True:
        for run in range(n_runs):
            logger.info("run:" + str(run))
            agent_fn = SAC
            for n_agent in [4]:
                # logger.info("n_agents:" + str(n_agent))
                # env = EnvWrapper("custom", name, n_agents=n_agent)
                # agent_fn = SAC
                # algname = "model"
                #
                # par = get_hyperpar("custom", alg=algname)
                # agent_kwargs = {"hyperpar": par}
                # record_env = EnvWrapper("custom", name, n_agents=n_agent)
                # single_run(env, agent_fn, logdata, data_log, run, agent_kwargs=agent_kwargs, n_steps=21000,
                #            record_env=record_env, name=algname+str(n_agent))

                env = EnvWrapper("custom", name, n_agents=n_agent, randomized=False)
                agent_fn = SAC
                algname = "SAC"

                par = get_hyperpar("MA2", alg=algname)
                agent_kwargs = {"hyperpar": par, "discrete": True if isinstance(env.action_space[0], spaces.Discrete) else False}
                record_env = EnvWrapper("custom", name, n_agents=n_agent, randomized=True)
                single_run(env, agent_fn, logdata, data_log, run, agent_kwargs=agent_kwargs, n_steps=2100000,
                           record_env=record_env, name=algname+str(n_agent), trainer_fn=None)

            # for steps in [40]:
            #     #
            #     algname = "model_regulated"
            #     par = get_hyperpar(name2, alg=algname)
            #     agent_kwargs = {"hyperpar": par}
            #     record_env = EnvWrapper("gym-record", name,
            #                             video_dir_name=logpath + "videos/" + logname + "/" + str(run) + algname)
            #     single_run(env, agent_fn, logdata, data_log, run, agent_kwargs=agent_kwargs, n_steps=50000,
            #                record_env=record_env, name=algname)
            #     record_env.close()
            #
            #     p.dump(logdata, open(logfile, "wb"))
            #
            #
            # algname = "SAC"
            # par = get_hyperpar(name2, alg=algname)
            # agent_kwargs = {"hyperpar": par}
            # record_env = EnvWrapper("gym-record", name,
            #                         video_dir_name=logpath + "videos/" + logname + "/" + str(run) + algname)
            # single_run(env, agent_fn, logdata,data_log, run, agent_kwargs=agent_kwargs, n_steps=50000,
            #            record_env=record_env, name=algname)
            # record_env.close()
            # p.dump(logdata, open(logfile, "wb"))

            # par = get_hyperpar(name, alg="model")
            # agent_kwargs = {"hyperpar": par}
            # single_run(env, agent_fn, logdata, run, agent_kwargs=agent_kwargs, n_steps=100000)
            # p.dump(logdata, open(logfile, "wb"))
            #

            #
            # agent_kwargs = {"n_steps": steps, "use_model": False}
            # single_run(env, agent_fn, logdata, run, agent_kwargs=agent_kwargs, n_steps=15000)
            # p.dump(logdata, open(logfile, "wb"))

            # agent_kwargs = {"n_steps": steps, "use_model": True, "diverse": False}
            # single_run(env, agent_fn, logdata, run, agent_kwargs=agent_kwargs, n_episodes=250)
            # p.dump(logdata, open(logfile, "wb"))
            #
            # agent_kwargs = {"n_steps": steps, "use_model": True}
            # single_run(env, agent_fn, logdata, run, agent_kwargs=agent_kwargs, n_episodes=50)
            # p.dump(logdata, open(logfile, "wb"))

    # except Exception:
    #     logger.exception("Fatal error.")