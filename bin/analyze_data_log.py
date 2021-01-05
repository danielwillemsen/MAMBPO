import pickle
import matplotlib.pyplot as plt
import numpy as np
from decentralizedlearning.algs.sac import SAC
from decentralizedlearning.envwrapper import EnvWrapper
from decentralizedlearning.run_utils import run_episode
from decentralizedlearning.algs.configs import config

import torch
import random
from scipy.stats import pearsonr

name = "3_agent_model"

name_run = "3_agent_model"
data = torch.load("../logs/" + name + ".p", map_location="cpu")

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def analyze_model(data, it, run=0):
    agents, env = setup_agent_env(data, it, run)

    score, _, statistics = run_episode(env, agents, eval=True, render=False, greedy_eval=False, store_data=True)
    print("Score: " + str(score))
    n_steps = 100
    if agents[0].model:
        rewards = statistics["rewards"]
        observations = statistics["observations"]
        actions = statistics["actions"]
        observation = observations[0][0]
        rews_pred_mult = []
        start = 000
        for rollout in range(5):
            observation = observations[start][0]
            rews_real = []
            rews_pred = []
            for step in range(n_steps):
                rews_real.append(rewards[start+step][0])
                action = actions[start + step][0]
                observation, rew_predict = agents[0].model.step_single(observation, action)
                rews_pred.append(rew_predict[0])
            rews_pred_mult.append(rews_pred)

    rews_real_cum = np.cumsum(rews_real)
    rews_pred_cum_mult = [np.cumsum(rews) for rews in rews_pred_mult]
    #print(rews_pred_cum_mult)
    # # Plot single rollout:
    # plt.plot(rews_pred_cum_mult[0][:n_steps], label="Predicted")
    # plt.plot(rews_real_cum[:n_steps], label="Real")
    #
    # plt.xlabel("Timestep")
    # plt.ylabel("Cumulative reward")
    # plt.title("Rollout after training for " + str(it*10) + " episodes")
    # plt.ylim(-100,250)
    # plt.legend()
    # plt.show()

    # Plot means and std of rollouts
    mean_scores = np.mean(rews_pred_cum_mult, axis=0)[:n_steps]
    std_scores = np.std(rews_pred_cum_mult, axis=0)[:n_steps]
    # Do interpolation
    plt.plot(mean_scores, label="Predicted")
    plt.fill_between([i for i in range(n_steps)], (mean_scores - std_scores), (mean_scores + std_scores), alpha=0.25)

    plt.plot(rews_real_cum[:n_steps], label="Real")

    plt.xlabel("Timestep")
    plt.ylabel("Cumulative reward")
    plt.title("Rollout after training for " + str(it*10) + " episodes")
    plt.ylim(-100,250)
    plt.legend()
    plt.show()

def analyze_model_statistics(data, it, run=0):
    agents, env = setup_agent_env(data, it, run)
    rollout_mod = {1: [], 10:[], 50:[],100:[]}
    rollout_real = {1: [], 10:[], 50:[],100:[]}

    for ep in range(1):
        score, _, statistics = run_episode(env, agents, eval=True, render=True, greedy_eval=False, store_data=True)
        print("Score: " + str(score))
        n_steps = 101
        start = random.randint(0, 500)
        if agents[0].model:
            rewards = statistics["rewards"]
            observations = statistics["observations"]
            actions = statistics["actions"]
            rews_pred_mult = []
            for rollout in range(5):
                observation = observations[start][0]
                rews_real = []
                rews_pred = []
                for step in range(n_steps):
                    rews_real.append(rewards[start+step][0])
                    action = actions[start + step][0]
                    observation, rew_predict = agents[0].model.step_single(observation, action)
                    rews_pred.append(rew_predict[0])
                rews_pred_mult.append(rews_pred)
        rews_real_cum = np.cumsum(rews_real)
        rews_pred_cum_mult = [np.cumsum(rews) for rews in rews_pred_mult]
        for item in rews_pred_cum_mult:
            for key in rollout_mod.keys():
                rollout_mod[key].append(item[key])
                rollout_real[key].append(rews_real_cum[key])
    # Calculate stuff
    for key in rollout_mod.keys():
        pred = np.array(rollout_mod[key])
        target = np.array(rollout_real[key])
        error = (pred-target)
        RMSE = np.sqrt(np.mean(error**2))
        print("RMSE-" + "Ep." + str(it*10) + "Rollout." + str(key) + "---" + str(RMSE))
        print("Mean Target-" + "Ep." + str(it*10) + "Rollout." + str(key) + "---" + str(np.mean(target)))
        print("Mean Model-" + "Ep." + str(it*10) + "Rollout." + str(key) + "---" + str(np.mean(pred)))
        print("Overshoot-" + "Ep." + str(it*10) + "Rollout." + str(key) + "---" + str((np.mean(pred)-np.mean(target))/(np.mean(target))*100) + "%")

        print("RMSE/Mean Target-" + "Ep." + str(it*10) + "Rollout." + str(key) + "---" + str(RMSE/np.mean(target)*100) + "%")

def analyze_model_obs_statistics(data, it, run=0, agent_list=None):
    agents, env = setup_agent_env(data, it, run)
    n_steps = 101

    rollout_mod = [[[] for i in range(n_steps)] for j in range(env.observation_space[0].shape[0])]
    rollout_real =[[[] for i in range(n_steps)] for j in range(env.observation_space[0].shape[0])]
    # rollout_mod = {1: [], 10:[], 50:[],100:[]}
    # rollout_real = {1: [], 10:[], 50:[],100:[]}
    if agent_list is None:
        agent_list = [i for i in range(len(agents))]
    for ep in range(50):
        score, _, statistics = run_episode(env, agents, eval=True, render=False, greedy_eval=False, store_data=True)
        print("Score: " + str(score))
        start = random.randint(0, 50)
        for agent in agent_list:
            if agents[0].model:
                rewards = statistics["rewards"]
                observations = statistics["observations"]
                actions = statistics["actions"]
                rews_pred_mult = []
                for rollout in range(5):
                    observation = observations[start][agent]
                    rews_real = []
                    rews_pred = []
                    for step in range(n_steps):
                        rews_real.append(rewards[start+step][agent])
                        action = actions[start + step][agent]
                        observation, rew_predict = agents[agent].model.step_single(observation, action)
                        rews_pred.append(rew_predict[0])
                        for j in range(len(observation)):
                            rollout_mod[j][step].append(observation[j])
                            rollout_real[j][step].append(observations[start+step+1][agent][j])
    #plt.scatter([rollout_mod[1][2][k] for k in range(len(rollout_mod[0][2]))], [rollout_real[1][2][k] for k in range(len(rollout_mod[0][2]))])
    #plt.show()
    n_corr = 0
    for rol_mod, rol_real in zip(rollout_mod, rollout_real):
        corrs = [pearsonr(single_len_mod, single_len_real)[0] for single_len_mod, single_len_real in zip(rol_mod, rol_real)]
        plt.plot(corrs, label=str(n_corr))
        n_corr += 1

    plt.xlabel("Rollout length")
    plt.ylabel("Correlation Coefficient")
    plt.ylim(-0.2, 1.0)
    plt.grid()
    # plt.legend()
    plt.title("Correlations of observations after training for " + str(it*10) + " episodes")
    plt.show()


def analyze_model_obs_naive(data, it, run=0):
    agents, env = setup_agent_env(data, it, run)
    n_steps = 101

    rollout_mod = [[[] for i in range(n_steps)] for j in range(env.observation_space[0].shape[0])]
    rollout_real =[[[] for i in range(n_steps)] for j in range(env.observation_space[0].shape[0])]
    # rollout_mod = {1: [], 10:[], 50:[],100:[]}
    # rollout_real = {1: [], 10:[], 50:[],100:[]}

    for ep in range(100):
        score, _, statistics = run_episode(env, agents, eval=True, render=False, greedy_eval=False, store_data=True)
        print("Score: " + str(score))
        start = random.randint(0, 50)
        if agents[0].model:
            rewards = statistics["rewards"]
            observations = statistics["observations"]
            actions = statistics["actions"]
            rews_pred_mult = []
            for rollout in range(5):
                observation = observations[start][0]
                rews_real = []
                rews_pred = []
                for step in range(n_steps):
                    rews_real.append(rewards[start+step][0])
                    action = actions[start + step][0]
                    # observation, rew_predict = agents[0].model.step_single(observation, action)
                    # rews_pred.append(rew_predict[0])
                    for j in range(len(observation)):
                        rollout_mod[j][step].append(observations[start][0][j])
                        rollout_real[j][step].append(observations[start+step+1][0][j])
    #plt.scatter([rollout_mod[1][2][k] for k in range(len(rollout_mod[0][2]))], [rollout_real[1][2][k] for k in range(len(rollout_mod[0][2]))])
    #plt.show()
    n_corr = 0
    for rol_mod, rol_real in zip(rollout_mod, rollout_real):
        corrs = [pearsonr(single_len_mod, single_len_real)[0] for single_len_mod, single_len_real in zip(rol_mod, rol_real)]
        plt.plot(corrs, label=str(n_corr))
        n_corr += 1

    plt.xlabel("Rollout length")
    plt.ylabel("Correlation Coefficient")
    plt.ylim(-0.2, 1.0)
    plt.grid()
    plt.legend()
    plt.title("Correlations of observations (Naive Model)")
    plt.show()

def analyze_mean_model_obs_corr(data, its, run=0):
    for it in its:
        agents, env = setup_agent_env(data, it, run, actor_it=8)
        n_steps = 101

        rollout_mod = [[[] for i in range(n_steps)] for j in range(17)]
        rollout_real =[[[] for i in range(n_steps)] for j in range(17)]
        # rollout_mod = {1: [], 10:[], 50:[],100:[]}
        # rollout_real = {1: [], 10:[], 50:[],100:[]}corrs

        for ep in range(25):
            score, _, statistics = run_episode(env, agents, eval=True, render=False, greedy_eval=False, store_data=True)
            print("Score: " + str(score))
            start = random.randint(0, 500)
            if agents[0].model:
                rewards = statistics["rewards"]
                observations = statistics["observations"]
                actions = statistics["actions"]
                rews_pred_mult = []
                for rollout in range(5):
                    observation = observations[start][0]
                    rews_real = []
                    rews_pred = []
                    for step in range(n_steps):
                        rews_real.append(rewards[start+step][0])
                        action = actions[start + step][0]
                        observation, rew_predict = agents[0].model.step_single(observation, action)
                        rews_pred.append(rew_predict[0])
                        for j in range(len(observation)):
                            rollout_mod[j][step].append(observation[j])
                            rollout_real[j][step].append(observations[start+step+1][0][j])
        # plt.scatter(rollout_mod[0][10], rollout_real[0][10])
        # plt.show()
        corrs_tot = []
        for rol_mod, rol_real in zip(rollout_mod, rollout_real):
            corrs = [pearsonr(single_len_mod, single_len_real)[0] for single_len_mod, single_len_real in zip(rol_mod, rol_real)]
            corrs_tot.append(corrs)

        plt.plot(np.mean(corrs_tot, axis=0), label=str(it*10) + " Episodes")

    plt.xlabel("Rollout length")
    plt.ylabel("Correlation Coefficient")
    # plt.legend()
    plt.title("Correlations of observations")
    plt.show()

def setup_agent_env(data, it, run, actor_it=None):
    env = EnvWrapper("custom", "waypoints.py", n_agents=4)
    if not actor_it:
        actor_it = it
    agents = [SAC(env.observation_space[0].shape[0], env.action_space[0].shape[0], use_model=True,
                  hidden_dims_model=(200, 200, 200)) for _ in range(env.n_agents)]
    for key, agent in enumerate(agents):
        agent.actor.load_state_dict(data[name_run]["runs"][run]["networks"][actor_it][3][key]["actor"])
        for critic, state_dict in zip(agent.critics, data[name_run]["runs"][run]["networks"][it][3][key]["critics"]):
            critic.load_state_dict(state_dict)
        agent.model.load_state_dict(data[name_run]["runs"][run]["networks"][it][3][key]["model"])
    return agents, env


def plot_all_run(data, var="score", plot_janner=False):
    if plot_janner:
        tasks = ["cheetah"]
        algorithms = ['mbpo']

        colors = {
            'mbpo': '#3a22b4',
        }
        import pickle
        for task in tasks:
            for alg in algorithms:
                print(task, alg)

                ## load results
                fname = '../logs/mbpo_v1.5_results/{}_{}.pkl'.format(task, alg)
                data_temp = pickle.load(open(fname, 'rb'))

                ## plot trial mean
                plt.plot(data_temp['x'] * 1000, data_temp['y'], linewidth=1.5, label=alg + " (Janner et al.)",
                         c=colors[alg])
                ## plot error bars
                plt.fill_between(data_temp['x'] * 1000, data_temp['y'] - data_temp['std'],
                                 data_temp['y'] + data_temp['std'], color=colors[alg],
                                 alpha=0.25)
    steps = np.arange(0,20001,100)
    for key in data.keys():
        values = []
        for run in data[key]["runs"]:
            values.append([])
            steps_var = [val[1] for val in run[var]]
            for step in steps:
                idx = find_nearest(steps_var, step)
                values[-1].append(np.mean(run[var][idx][3]))
            values[-1] = np.array(values[-1])
        mean_scores = np.mean(values, axis=0)
        std_scores = np.std(values, axis=0)

        plt.plot(steps, mean_scores, label=key[:-1] + ' - ' + str(key[-1]) + " Agents")
        plt.fill_between(steps, mean_scores - std_scores, mean_scores + std_scores, alpha=0.25)

    plt.xlim(0, steps[-1])
    plt.xlabel("Timestep")
    plt.ylabel("Score")
    plt.legend()
    plt.show()


def plot_single_run(data, run=None, var="score", plot_janner=True, agents=None):
    if plot_janner:
        tasks = ["cheetah"]
        algorithms = ['mbpo']

        colors = {
            'mbpo': '#3a22b4',
        }
        import pickle
        for task in tasks:
            for alg in algorithms:
                print(task, alg)

                ## load results
                fname = '../logs/mbpo_v1.5_results/{}_{}.pkl'.format(task, alg)
                data_temp = pickle.load(open(fname, 'rb'))

                ## plot trial mean
                plt.plot(data_temp['x'] * 1000, data_temp['y'], linewidth=1.5, label=alg + " (Janner et al.)",
                         c=colors[alg])
                ## plot error bars
                plt.fill_between(data_temp['x'] * 1000, data_temp['y'] - data_temp['std'],
                                 data_temp['y'] + data_temp['std'], color=colors[alg],
                                 alpha=0.25)

    if run is not None:
        for key in data.keys():
            vals = [item[3][0] for item in data[key]["runs"][run][var]]
            steps =[item[1] for item in data[key]["runs"][run][var]]
            plt.plot(steps,vals, label=key)
    else:
        for key in data.keys():
            for run in range(len(data[key]["runs"])):
                if agents is None:
                    vals = [np.mean(item[3]) for item in data[key]["runs"][run][var]]
                    steps = [item[1] for item in data[key]["runs"][run][var]]
                    plt.plot(steps, vals, label=key)

                elif type(agents) == list:
                    for agent in agents:
                        vals = [item[3][agent] for item in data[key]["runs"][run][var]]
                        steps =[item[1] for item in data[key]["runs"][run][var]]
                        plt.plot(steps,vals, label=key+"-"+str(agent))

    plt.xlim(0, steps[-1])
    plt.xlabel("Timestep")
    plt.ylabel("Score")
    plt.legend()
    plt.show()


def plot_model_vis(data, it, run=0):
    plt.plot(np.cumsum(data[name_run]["runs"][run]["model_vis"][it][3]["real"][:50]), label="Real")
    plt.plot(np.cumsum(data[name_run]["runs"][run]["model_vis"][it][3]["pred"][:50]), label="Predicted")
    plt.xlabel("Timestep")
    plt.ylabel("Cumulative reward")
    plt.title("Rollout after training for " + str(data[name_run]["runs"][run]["model_vis"][it][0]) + " episodes")
    plt.ylim(-100,250)
    plt.legend()
    plt.show()

plot_single_run(data, plot_janner=False, var="score_greedy")
plot_all_run(data, plot_janner=False, var="score_greedy")

# analyze_model(data, 0)
# analyze_model(data, 1)
# # plot_model_vis(data, 1)
# # analyze_model(data, 0)
# analyze_model(data, 2)
# analyze_model_obs_statistics(data, 1)
# analyze_model_obs_statistics(data, 2)
#
# analyze_model_obs_statistics(data, 0)

# #
# # analyze_model(data, 1)
# # # analyze_model(data, 10)
# analyze_model_obs_naive(data, 2)

# analyze_model_obs_statistics(data, 0)
# analyze_model_obs_statistics(data, 2)

# #
# # analyze_model(data, 2)
# # # analyze_model(data, 10)
# analyze_model_obs_statistics(data, 2)
# #
# # analyze_model(data, 5)
# # # analyze_model(data, 10)
# analyze_model_obs_statistics(data, 5)
#
# # analyze_model(data, 10)
# # # analyze_model(data, 10)
# analyze_model_obs_statistics(data, 10)

#analyze_mean_model_obs_corr(data, [0,1,2,5,8])
#analyze_model_statistics(data, 0)
#analyze_model_statistics(data, 2)
#analyze_model_statistics(data, 5)
#
# plot_single_run(data, var="score_greedy")
# plot_all_run(data, var="score_greedy")
# #
#plot_model_vis(data, 0)
#plot_model_vis(data, 1)
