from decentralizedlearning.algs.configs.config import get_hyperpar
from decentralizedlearning.data_log import DataLog
import itertools
import pickle as p
import time
import os
import logging
from gym import spaces
import pyglet

def scale_action(env, agent_id, action):
    if not isinstance(env.action_space[agent_id], spaces.Discrete):
        return (env.action_space[agent_id].high - env.action_space[agent_id].low) * action * 0.5
    else:
        return action


def run_episode(env, agents, eval=False, render=False, generate_val_data=False, greedy_eval=True, steps=25, store_data=False, trainer=None, benchmark=False, save_all=False, name=None):
    obs_n = env.reset()
    reward_tot = [0.0 for i in range(len(agents))]
    reward_n = [0.0 for i in range(len(agents))]
    done_n = [False for i in range(len(agents))]
    rewards_target = [0.0 for i in range(len(agents))]
    rewards_collision = [0.0 for i in range(len(agents))]
    info_n_list = []
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
        info_n_list.append(info_n)
        rewards_target = [0.0 for i in range(len(agents))]
        rewards_collision = [0.0 for i in range(len(agents))]
        # for j, tup in enumerate(info_n["n"]):
        #     rewards_collision[j] = tup[1]
        #     rewards_target[j] = tup[2]
        # rewards_collision = []#reward_details["rewards_collision"]
        # rewards_target = []#reward_details["rewards_target"]
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
        # if store_data or True:
        #     rewards_details["rewards_target"].append(rewards_target)
        #     rewards_details["rewards_collision"].append(rewards_collision)

        # render all agent views
        if render:
            env.render()
        if save_all and i<10:
            pyglet.image.get_buffer_manager().get_color_buffer().save("../figures/videos/" + name + "0" + str(i) + ".png")
        elif save_all:
            pyglet.image.get_buffer_manager().get_color_buffer().save("../figures/videos/" + name + str(i) + ".png")
    for j, agent in enumerate(agents):
        agent.reset()
    if trainer is not None:
        trainer.reset()
    if benchmark:
        return reward_tot, i+1, info_n_list
    if not store_data:
        return reward_tot, i + 1
    else:
        extra_data = {"observations": observations, "actions": actions, "rewards": rewards, "rewards_details": rewards_details}
        return reward_tot, i + 1, extra_data

def train(env, agents, data_log, n_episodes=10000, n_steps=None, generate_val_data=False, record_env=None):
    logger = logging.getLogger('root')
    scores = []
    scores_eval = []
    times = []
    time_start = time.time()
    step_tot = 0
    steps = []
    ep_generator = range(n_episodes) if n_episodes else itertools.count()
    logger.info(env.env.observation_space.high)

    if generate_val_data:
        logger.info("Generating val data")
        while len(agents[0].val_buffer) < agents[0].val_buffer.n_samples:
            score, _ = run_episode(env, agents, eval=False, generate_val_data=True)

    for i in ep_generator:
        data_log.set_episode(i)
        logger.info("episode:" + str(i))
        if i % 10 == 0:
            generate_statistics(agents, record_env, data_log)
        # Sometimes generate val data
        if i % 2 == 0:
            run_episode(env, agents, eval=False, generate_val_data=True)
        score, step = run_episode(env, agents, render=False)
        step_tot += step
        data_log.set_step(step_tot)
        data_log.log_var("score", score)

        score2, _ = run_episode(env, agents, eval=True, render=False)
        data_log.log_var("score_greedy", score2)

        # score_eval, _ = run_episode(env, agents, eval=True)
        scores.append(score)
        # scores_eval.append(score_eval)
        t = time.time() - time_start
        times.append(t)
        logger.info("time_elapsed:" + str(t))
        logger.info("score:" + str(score))
        logger.info("score_greedy:" + str(score2))

        steps.append(step_tot)
        logger.info("step_tot:" + str(step_tot))
        if n_steps and step_tot > n_steps:
            break
        # logger.info("score_eval:"+str(score_eval))
        logger.info("Saving log...")
        data_log.save()
        logger.info("Saved log")
    return {"scores": scores, "steps": steps, "scores_eval": scores_eval, "times": times}


def generate_statistics(agents, record_env, data_log):
    if record_env is not None:
        score, _, statistics = run_episode(record_env, agents, eval=True, render=False, greedy_eval=False, store_data=True)

    # Test Model
    if agents[0].model:
        rewards = statistics["rewards"]
        observations = statistics["observations"]
        actions = statistics["actions"]
        observation = observations[0][0]
        rews_real = []
        rews_pred = []
        for step in range(1000):
            rews_real.append(rewards[step][0])
            action = actions[step][0]
            observation, rew_predict = agents[0].model.step_single(observation, action)
            rews_pred.append(rew_predict[0])
        data_log.log_var("model_vis", {"real": rews_real, "pred": rews_pred})

    # Store model, policy and actor
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
    data_log.log_var("networks", networks)



def single_run(env, agent_fn, logdata, data_log, seed, agent_kwargs=dict(), n_episodes=None, n_steps=None, record_env=None, name=None):
    logger = logging.getLogger(__name__)
    if not name:
        name = agent_fn.__name__ + str(agent_kwargs)
    logger.info("agent:" + name)


    env.env.seed(seed=seed)
    if name not in logdata:
        logdata[name] = []
    # env.render()
    obs_n = env.reset()
    agents = []
    for i in range(env.n_agents):
        agents.append(agent_fn(env.observation_space[i].shape[0], env.action_space[i].shape[0], **agent_kwargs))

    data_log.init_run(name)

    logdata[name].append(
        train(env, agents, data_log, n_episodes=n_episodes, n_steps=n_steps, generate_val_data=True, record_env=record_env))
