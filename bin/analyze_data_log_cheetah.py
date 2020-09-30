import pickle
import matplotlib.pyplot as plt
import numpy as np
from decentralizedlearning.algs.sac import SAC
from decentralizedlearning.envwrapper import EnvWrapper
from decentralizedlearning.run_utils import run_episode
from decentralizedlearning.algs.configs import config_cheetah
from decentralizedlearning.algs.models import DegradedSim

import torch
import random
from scipy.stats import pearsonr

name = "cheetah_plotmodel_4layers_3"
#cheetah_plotmodel_4layers_regulated_2
name_run = "model"
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

    for ep in range(50):
        score, _, statistics = run_episode(env, agents, eval=True, render=False, greedy_eval=False, store_data=True)
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

def analyze_model_obs_statistics_naive(data, it, runs=[0,1,2]):
    corrs_tot = []
    for run in runs:
        n_steps = 101
        agents, env = setup_agent_env(data, it, run)

        rollout_mod = [[[] for i in range(n_steps)] for j in range(env.observation_space[0].shape[0])]
        rollout_real =[[[] for i in range(n_steps)] for j in range(env.observation_space[0].shape[0])]
        rollout_mod_naive = [[[] for i in range(n_steps)] for j in range(env.observation_space[0].shape[0])]

        # rollout_mod = {1: [], 10:[], 50:[],100:[]}
        # rollout_real = {1: [], 10:[], 50:[],100:[]}

        for ep in range(10):
            score, _, statistics = run_episode(env, agents, eval=True, render=False, greedy_eval=False, store_data=True)
            print("Score: " + str(score))
            start = random.randint(0, 1)
            if agents[0].model:
                rewards = statistics["rewards"]
                observations = statistics["observations"]
                actions = statistics["actions"]
                rews_pred_mult = []
                for rollout in range(1):
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
                            rollout_mod_naive[j][step].append(observations[start][0][j])

        # plt.scatter([rollout_mod[1][2][k] for k in range(len(rollout_mod[0][2]))], [rollout_real[1][2][k] for k in range(len(rollout_mod[0][2]))])
        plt.show()
        num_cor = 0
        corrs_tot.append([])
        for rol_mod, rol_real, rol_mod_naive in zip(rollout_mod, rollout_real, rollout_mod_naive):
            corrs = [pearsonr(single_len_mod, single_len_real)[0] for single_len_mod, single_len_real in zip(rol_mod, rol_real)]
            corrs_naive = [pearsonr(single_len_mod, single_len_real)[0] for single_len_mod, single_len_real in zip(rol_mod_naive, rol_real)]
            corrs_tot[-1].append(corrs_naive)


    for corrs in zip(*corrs_tot):
        plt.plot(np.mean(corrs, axis=0))#, label=str(num_cor))
    num_cor += 1

    plt.xlabel("Rollout length")
    plt.ylabel("Correlation with real observations")
    plt.legend()
    plt.ylim(-0.2, 1.0)
    plt.grid()
    plt.title("Model trained for " + str(it*10) + " episodes")
    plt.show()

def analyze_model_obs_statistics(data, it, runs=[0,1,2]):
    corrs_tot = []
    for run in runs:
        n_steps = 101
        agents, env = setup_agent_env(data, it, run)

        rollout_mod = [[[] for i in range(n_steps)] for j in range(env.observation_space[0].shape[0])]
        rollout_real =[[[] for i in range(n_steps)] for j in range(env.observation_space[0].shape[0])]
        rollout_mod_naive = [[[] for i in range(n_steps)] for j in range(env.observation_space[0].shape[0])]

        # rollout_mod = {1: [], 10:[], 50:[],100:[]}
        # rollout_real = {1: [], 10:[], 50:[],100:[]}

        for ep in range(100):
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
                            rollout_mod_naive[j][step].append(observations[start][0][j])

        # plt.scatter([rollout_mod[1][2][k] for k in range(len(rollout_mod[0][2]))], [rollout_real[1][2][k] for k in range(len(rollout_mod[0][2]))])
        plt.show()
        num_cor = 0
        corrs_tot.append([])
        for rol_mod, rol_real, rol_mod_naive in zip(rollout_mod, rollout_real, rollout_mod_naive):
            corrs = [pearsonr(single_len_mod, single_len_real)[0] for single_len_mod, single_len_real in zip(rol_mod, rol_real)]
            corrs_naive = [pearsonr(single_len_mod, single_len_real)[0] for single_len_mod, single_len_real in zip(rol_mod_naive, rol_real)]
            corrs_tot[-1].append(corrs)

    for corrs in zip(*corrs_tot):
        plt.plot(np.mean(corrs, axis=0))#, label=str(num_cor))
    num_cor += 1

    plt.xlabel("Rollout length")
    plt.ylabel("Correlation with real observations")
    plt.legend()
    plt.ylim(-0.2, 1.0)
    plt.grid()
    plt.title("Model trained for " + str(it*10) + " episodes")
    plt.show()


def analyze_model_obs_statistics_degraded(data, it, runs=[0]):
    corrs_tot = []
    for run in runs:
        n_steps = 101
        agents, env = setup_agent_env(data, it, run)
        _, env_deg = setup_agent_env(data, it, run)
        env_deg.reset()
        model = DegradedSim(env_deg)

        rollout_mod = [[[] for i in range(n_steps)] for j in range(env.observation_space[0].shape[0])]
        rollout_real =[[[] for i in range(n_steps)] for j in range(env.observation_space[0].shape[0])]
        rollout_mod_naive = [[[] for i in range(n_steps)] for j in range(env.observation_space[0].shape[0])]

        # rollout_mod = {1: [], 10:[], 50:[],100:[]}
        # rollout_real = {1: [], 10:[], 50:[],100:[]}

        for ep in range(100):
            score, _, statistics = run_episode(env, agents, eval=True, render=False, greedy_eval=False, store_data=True, store_states=True)
            print("Score: " + str(score))
            start = random.randint(0, 500)
            if model:
                rewards = statistics["rewards"]
                observations = statistics["observations"]
                actions = statistics["actions"]
                states = statistics["states"]
                rews_pred_mult = []
                for rollout in range(5):
                    observation = observations[start][0]
                    state = states[start]
                    rews_real = []
                    rews_pred = []
                    for step in range(n_steps):
                        rews_real.append(rewards[start+step][0])
                        action = actions[start + step][0]
                        observation, rew_predict = model.step_single(observation, action, state=state)
                        state = model.env.get_state()
                        rews_pred.append(rew_predict[0])
                        for j in range(len(observation)):
                            rollout_mod[j][step].append(observation[j])
                            rollout_real[j][step].append(observations[start+step+1][0][j])
                            rollout_mod_naive[j][step].append(observations[start][0][j])

        # plt.scatter([rollout_mod[1][2][k] for k in range(len(rollout_mod[0][2]))], [rollout_real[1][2][k] for k in range(len(rollout_mod[0][2]))])
        plt.show()
        num_cor = 0
        corrs_tot.append([])
        for rol_mod, rol_real, rol_mod_naive in zip(rollout_mod, rollout_real, rollout_mod_naive):
            corrs = [pearsonr(single_len_mod, single_len_real)[0] for single_len_mod, single_len_real in zip(rol_mod, rol_real)]
            corrs_naive = [pearsonr(single_len_mod, single_len_real)[0] for single_len_mod, single_len_real in zip(rol_mod_naive, rol_real)]
            corrs_tot[-1].append(corrs)

    for corrs in zip(*corrs_tot):
        plt.plot(np.mean(corrs, axis=0))#, label=str(num_cor))
    num_cor += 1

    plt.xlabel("Rollout length")
    plt.ylabel("Correlation with real observations")
    plt.legend()
    plt.ylim(-0.2, 1.0)
    plt.grid()
    plt.title("Model trained for " + str(it*10) + " episodes")
    plt.show()

def analyze_mean_model_obs_corr(data, its, run=0):
    for it in its:
        agents, env = setup_agent_env(data, it, run, actor_it=8)
        n_steps = 101

        rollout_mod = [[[] for i in range(n_steps)] for j in range(17)]
        rollout_real =[[[] for i in range(n_steps)] for j in range(17)]
        # rollout_mod = {1: [], 10:[], 50:[],100:[]}
        # rollout_real = {1: [], 10:[], 50:[],100:[]}corrs

        for ep in range(5):
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
    plt.ylim(-0.1, 1.0)
    plt.grid()
    plt.legend()
    plt.title("Correlations of observations")
    plt.show()

def setup_agent_env(data, it, run, actor_it=None):
    env = EnvWrapper("gym", "HalfCheetah-v2")
    if not actor_it:
        actor_it = it
    agents = [SAC(env.observation_space[0].shape[0], env.action_space[0].shape[0], use_model=True,
                  hidden_dims_model=(200, 200, 200, 200))]
    for agent in agents:
        agent.actor.load_state_dict(data[name_run]["runs"][run]["networks"][actor_it][3][0]["actor"])
        for critic, state_dict in zip(agent.critics, data[name_run]["runs"][run]["networks"][it][3][0]["critics"]):
            critic.load_state_dict(state_dict)
        agent.model.load_state_dict(data[name_run]["runs"][run]["networks"][it][3][0]["model"])
    return agents, env


def plot_all_run(data, var="score", plot_janner=True, baseline=None, baseline_name=None):
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
    steps = np.arange(0,50001,1000)
    for key in data.keys():
        plot_name(data, key, steps, var)
    if baseline:
        data2 = torch.load("../logs/" + baseline + ".p", map_location="cpu")
        key=baseline_name
        plot_name(data2, key, steps, var)

    plt.xlim(0, steps[-1])
    plt.xlabel("Timestep")
    plt.ylabel("Score")
    plt.legend()
    plt.show()


def plot_name(data, key, steps, var):
    values = []
    for run in data[key]["runs"]:
        values.append([])
        steps_var = [val[1] for val in run[var]]
        for step in steps:
            idx = find_nearest(steps_var, step)
            values[-1].append(run[var][idx][3][0])
        values[-1] = np.array(values[-1])
    mean_scores = np.mean(values, axis=0)
    std_scores = np.std(values, axis=0)
    plt.plot(steps, mean_scores, label=key + " (ours)")
    plt.fill_between(steps, mean_scores - std_scores, mean_scores + std_scores, alpha=0.25)


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

# plot_single_run(data, plot_janner=True, var="score_greedy")
# plot_all_run(data, plot_janner=True, var="score_greedy")
# plot_all_run(data, plot_janner=True, var="score_greedy", baseline="cheetah_plotmodel_4layers_regulated_2", baseline_name="model_regulated")
analyze_model_obs_statistics_degraded(data, 5)

#analyze_model(data, 0)
#analyze_model(data, 1)
# plot_model_vis(data, 1)
# analyze_model(data, 0)
# # analyze_model(data, 10)
# analyze_model_obs_statistics(data, 0)
# #
# analyze_model_obs_statistics(data, 0)
# analyze_model_obs_statistics(data, 1)
# analyze_model_obs_statistics(data, 2)
# analyze_model_obs_statistics_naive(data, 0)
# analyze_model_obs_statistics(data, 9)

# # analyze_model(data, 1)
# # # analyze_model(data, 10)
# analyze_model_obs_statistics(data, 10)
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