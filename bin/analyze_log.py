import pickle
import matplotlib.pyplot as plt
import numpy as np
from decentralizedlearning.algs.sac import SAC
from decentralizedlearning.envwrapper import EnvWrapper
from decentralizedlearning.run_utils import run_episode
from decentralizedlearning.algs.configs import config
from decentralizedlearning.algs.models import DegradedSim
from decentralizedlearning.algs.models import EnsembleModel
from decentralizedlearning.algs.mambpo import MAMBPO, MAMBPOAgent
import torch
import random
from scipy.stats import pearsonr
import pyglet
import pickle as p

# plt.rcParams.update({'font.size': 18})
plt.style.use('seaborn-paper')

name = "tag_masac_1_auto_2"
#cheetah_plotmodel_4layers_regulated_2
name_run = "SAC4"
data = torch.load("../logs/" + name + ".p", map_location="cpu")
a =2

def moving_average(a, n=50):
    b = np.zeros(a.size)
    for i in range(len(a)):
        if i>=n:
            b[i] = np.mean(a[i-n:i+1])
        else:
            b[i] = np.mean(a[0:i+1])
    return b

def moving_average_2d(a, n=200):
    b = np.zeros(a.shape)
    for i in range(a.shape[1]):
        if i>=n:
            b[:,i] = np.mean(a[:,i-n:i+1], axis=1)
        else:
            b[:,i] = np.mean(a[:,0:i+1], axis=1)
    return b

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def find_nearest_mult(array, values):
    array = np.asarray(array)
    idx = (np.abs(np.expand_dims(array, 1) - np.expand_dims(values, 0))).argmin(axis=0)
    return idx

def play_game(data, it, run=0, env_name=None, steps=25, name=None):
    if not name:
        name = env_name
    trainer, agents, env, _ = setup_agent_env(data, it, run, env_name=env_name)
    env.reset()
    score, _, statistics = run_episode(env, agents, eval=True, render=True, greedy_eval=False, store_data=True, steps=steps)
    pyglet.image.get_buffer_manager().get_color_buffer().save(name+"test.png")
    print(score)

def analyze_model(data, it, run=0, env_name=None):
    trainer, agents, env, ep = setup_agent_env(data, it, run, env_name=env_name)

    score, _, statistics = run_episode(env, agents, eval=True, render=False, greedy_eval=False, store_data=True)
    print("Score: " + str(score))
    n_steps = 24
    if trainer.model:
        rewards = statistics["rewards"]
        observations = statistics["observations"]
        actions = statistics["actions"]
        observation = observations[0]
        rews_pred_mult = [0]
        start = 000
        for rollout in range(5):
            observation = np.concatenate(observations[start],axis=0)
            rews_real = []
            rews_pred = []
            for step in range(n_steps):
                rews_real.append(rewards[start+step+1][0]) #We predict the next reward, thus +1 is needed.
                action = actions[start + step]
                observation, rew_predict = trainer.model.step_single(observation, action)
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
    plt.title("Rollout after training for " + ep + " episodes")
    # plt.ylim(-100,250)
    plt.legend()
    plt.show()

def analyze_model_statistics(data, it, run=0, env_name=None):
    trainer, agents, env, ep = setup_agent_env(data, it, run, env_name=env_name)
    rollout_mod = {1: [], 2:[], 5:[]}
    rollout_real = {1: [], 2:[], 5:[]}

    for ep in range(5):
        score, _, statistics = run_episode(env, agents, eval=True, render=False, greedy_eval=False, store_data=True)
        print("Score: " + str(score))
        n_steps = 6
        start = random.randint(0, 18)
        if trainer.model:
            rewards = statistics["rewards"]
            observations = statistics["observations"]
            actions = statistics["actions"]
            rews_pred_mult = []
            for rollout in range(5):
                observation = np.concatenate(observations[start], axis=0)
                rews_real = []
                rews_pred = []
                for step in range(n_steps):
                    rews_real.append(rewards[start+step+1][0])
                    action = actions[start + step]
                    observation, rew_predict = trainer.model.step_single(observation, action)
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

        print("RMSE/Mean Target-" + "Ep." + str(it*10) + "Rollout." + str(key) + "---" + str(RMSE/np.mean(np.abs(target)*100)) + "%")


def plot_model_statistics(data, its, run=0, env_name=None):
    errors = []
    errors_observations = None
    episodes = []
    for it in its:
        trainer, agents, env, ep = setup_agent_env(data, it, run, env_name=env_name)
        rollout_mod = {1: []}
        rollout_real = {1: []}
        observation = np.concatenate(env.reset())
        print(len(observation))
        if errors_observations is None:
            errors_observations = [[] for obs in observation]
        obs_real ={1: [[] for obs in observation]}
        obs_mod = {1: [[] for obs in observation]}
        episodes.append(int(ep))
        for ep in range(100):
            score, _, statistics = run_episode(env, agents, eval=True, render=False, greedy_eval=False, store_data=True)
            print("Score: " + str(score))
            n_steps = 1
            start = random.randint(0, 23)
            if trainer.model:
                rewards = statistics["rewards"]
                observations = statistics["observations"]
                actions = statistics["actions"]
                rews_pred_mult = []
                for rollout in range(5):
                    observation = np.concatenate(observations[start], axis=0)
                    rews_real = []
                    rews_pred = []
                    for step in range(n_steps):
                        rews_real.append(rewards[start+step+1][0])
                        obs_real_i = np.concatenate(observations[start+step+1]).tolist()
                        action = actions[start + step]
                        observation, rew_predict = trainer.model.step_single(observation, action)
                        for n,obs_real_i_i in enumerate(obs_real_i):
                            obs_real[1][n].append(obs_real_i_i)
                            obs_mod[1][n].append(observation[n])
                        rews_pred.append(rew_predict[0])
                    rews_pred_mult.append(rews_pred)
            rews_real_cum = np.cumsum(rews_real)
            rews_pred_cum_mult = [np.cumsum(rews) for rews in rews_pred_mult]
            for item in rews_pred_cum_mult:
                for key in rollout_mod.keys():
                    rollout_mod[key].append(item[key-1])
                    rollout_real[key].append(rews_real_cum[key-1])
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
        errors.append(RMSE/np.mean(target))

        for i, _ in enumerate(observation):
            pred = np.array(obs_mod[1][i])
            target = np.array(obs_real[1][i])
            error = pred-target
            RMSE = np.sqrt(np.mean(error**2))
            errors_observations[i].append(RMSE/np.mean(np.abs(target)))
    plt.scatter(episodes, errors)

    for i, errors_observation in enumerate(errors_observations):
        plt.scatter(episodes, errors_observation, label=str(i), c="k")
    plt.xlabel("Episodes trained")
    plt.ylabel("Model error (Prediction RMSE / Mean Absolute Target Value)")
    plt.ylim(0.,2.0)
    plt.grid()
    plt.show()



def plot_model_statistics_corr(data, its, run=0, env_name=None):
    plt.figure(figsize=(5,4))
    plt.grid()

    errors = []
    errors_observations = None
    episodes = []
    for it in its:
        trainer, agents, env, ep = setup_agent_env(data, it, run, env_name=env_name)
        rollout_mod = {1: []}
        rollout_real = {1: []}
        observation = np.concatenate(env.reset())
        print(len(observation))
        if errors_observations is None:
            errors_observations = [[] for obs in observation]
        obs_real ={1: [[] for obs in observation]}
        obs_mod = {1: [[] for obs in observation]}
        episodes.append(int(ep))
        for ep in range(250):
            score, _, statistics = run_episode(env, agents, eval=True, render=False, greedy_eval=False, store_data=True)
            print("Score: " + str(score))
            n_steps = 1
            start = random.randint(0, 23)
            if trainer.model:
                rewards = statistics["rewards"]
                observations = statistics["observations"]
                actions = statistics["actions"]
                rews_pred_mult = []
                for rollout in range(5):
                    observation = np.concatenate(observations[start], axis=0)
                    rews_real = []
                    rews_pred = []
                    for step in range(n_steps):
                        rews_real.append(rewards[start+step+1][0])
                        obs_real_i = np.concatenate(observations[start+step+1]).tolist()
                        action = actions[start + step]
                        observation, rew_predict = trainer.model.step_single(observation, action)
                        for n,obs_real_i_i in enumerate(obs_real_i):
                            obs_real[1][n].append(obs_real_i_i)
                            obs_mod[1][n].append(observation[n])
                        rews_pred.append(rew_predict[0])
                    rews_pred_mult.append(rews_pred)
            rews_real_cum = np.cumsum(rews_real)
            rews_pred_cum_mult = [np.cumsum(rews) for rews in rews_pred_mult]
            for item in rews_pred_cum_mult:
                for key in rollout_mod.keys():
                    rollout_mod[key].append(item[key-1])
                    rollout_real[key].append(rews_real_cum[key-1])
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
        errors.append(pearsonr(pred, target)[0])

        for i, _ in enumerate(observation):
            pred = np.array(obs_mod[1][i])
            target = np.array(obs_real[1][i])
            error = pred-target
            RMSE = np.sqrt(np.mean(error**2))
            errors_observations[i].append(pearsonr(pred, target)[0])
    plt.scatter(episodes, errors, label="Reward", c="k")

    for i, errors_observation in enumerate(errors_observations):
        if i==0:
            plt.scatter(episodes, errors_observation, marker="x", label="Observations", c="k")
        else:
            plt.scatter(episodes, errors_observation, marker="x", c="k")

    plt.xlabel("Episodes trained")
    plt.ylabel("Correlation")
    plt.ylim(0.0,1.0)
    plt.legend()
    plt.show()


def plot_model_statistics_r2(data, its, n_runs=5, env_name=None):
    plt.figure(figsize=(5,4))
    plt.grid()

    errors = []
    errors_observations = None
    episodes = []
    for it in its:
        trainer, agents, env, ep = setup_agent_env(data, it, 0, env_name=env_name)
        rollout_mod = {1: []}
        rollout_real = {1: []}
        observation = np.concatenate(env.reset())
        # print(len(observation))
        if errors_observations is None:
            errors_observations = [[] for obs in observation]
        obs_real ={1: [[] for obs in observation]}
        obs_mod = {1: [[] for obs in observation]}
        episodes.append(int(ep))
        for run in range(n_runs):
            trainer, agents, env, ep = setup_agent_env(data, it, run, env_name=env_name)
            for ep in range(5):
                score, _, statistics = run_episode(env, agents, eval=True, render=False, greedy_eval=False, store_data=True)
                # print("Score: " + str(score))
                n_steps = 1
                start = random.randint(0, 23)
                if trainer.model:
                    rewards = statistics["rewards"]
                    observations = statistics["observations"]
                    actions = statistics["actions"]
                    rews_pred_mult = []
                    for rollout in range(1):
                        observation = np.concatenate(observations[start], axis=0)
                        rews_real = []
                        rews_pred = []
                        for step in range(n_steps):
                            rews_real.append(rewards[start+step+1][0])
                            obs_real_i = np.concatenate(observations[start+step+1]).tolist()
                            action = actions[start + step]
                            observation, rew_predict = trainer.model.step_single(observation, action)
                            for n,obs_real_i_i in enumerate(obs_real_i):
                                obs_real[1][n].append(obs_real_i_i)
                                obs_mod[1][n].append(observation[n])
                            rews_pred.append(rew_predict[0])
                        rews_pred_mult.append(rews_pred)
                rews_real_cum = np.cumsum(rews_real)
                rews_pred_cum_mult = [np.cumsum(rews) for rews in rews_pred_mult]
                for item in rews_pred_cum_mult:
                    for key in rollout_mod.keys():
                        rollout_mod[key].append(item[key-1])
                        rollout_real[key].append(rews_real_cum[key-1])
        # Calculate stuff
        for key in rollout_mod.keys():
            pred = np.array(rollout_mod[key])
            target = np.array(rollout_real[key])
            error = (pred-target)
            SS_tot = np.sum((target - np.mean(target))**2)
            SS_res = np.sum(error**2)
            # RMSE = np.sqrt(np.mean(error**2))
            # print("RMSE-" + "Ep." + str(it*10) + "Rollout." + str(key) + "---" + str(RMSE))
            # print("Mean Target-" + "Ep." + str(it*10) + "Rollout." + str(key) + "---" + str(np.mean(target)))
            # print("Mean Model-" + "Ep." + str(it*10) + "Rollout." + str(key) + "---" + str(np.mean(pred)))
            # print("Overshoot-" + "Ep." + str(it*10) + "Rollout." + str(key) + "---" + str((np.mean(pred)-np.mean(target))/(np.mean(target))*100) + "%")
            #
            # print("RMSE/Mean Target-" + "Ep." + str(it*10) + "Rollout." + str(key) + "---" + str(RMSE/np.mean(target)*100) + "%")
        errors.append(1-SS_res/SS_tot)

        for i, _ in enumerate(observation):
            pred = np.array(obs_mod[1][i])
            target = np.array(obs_real[1][i])
            error = pred - target
            SS_tot = np.sum((target - np.mean(target))**2)
            SS_res = np.sum(error**2)
            errors_observations[i].append(1-SS_res/SS_tot)
    plt.scatter(episodes, errors, label="Reward", c="k")

    for i, errors_observation in enumerate(errors_observations):
        if i==0:
            plt.scatter(episodes, errors_observation, marker="x", label="Observations", c="k")
        else:
            plt.scatter(episodes, errors_observation, marker="x", c="k")

    plt.xlabel("Episodes trained")
    plt.ylabel(r"R$^2$")
    plt.ylim(0.0,1.0)
    plt.legend()
    plt.show()


def plot_model_statistics_r22(data, its, n_runs=5, env_name=None, pick=False, add_mean_line=False,ylim=None):
    plt.figure(figsize=(4,3), dpi=300)
    plt.grid()

    if not pick:
        errors = []
        errors_observations = None
        episodes = []
        for it in its:
            trainer, agents, env, ep = setup_agent_env(data, it, 0, env_name=env_name)
            rollout_mod = {1: []}
            rollout_real = {1: []}
            observation = np.concatenate(env.reset())
            # print(len(observation))
            if errors_observations is None:
                errors_observations = [[] for obs in observation]
            obs_real ={1: [[] for obs in observation]}
            obs_mod = {1: [[] for obs in observation]}
            episodes.append(int(ep))
            for run in range(n_runs):
                trainer, agents, env, ep = setup_agent_env(data, it, run, env_name=env_name)
                for ep in range(250):
                    score, _, statistics = run_episode(env, agents, eval=True, render=False, greedy_eval=False, store_data=True)
                    # print("Score: " + str(score))
                    n_steps = 1
                    start = random.randint(0, 23)
                    if trainer.model:
                        rewards = statistics["rewards"]
                        observations = statistics["observations"]
                        actions = statistics["actions"]
                        rews_pred_mult = []
                        for rollout in range(1):
                            observation = np.concatenate(observations[start], axis=0)
                            rews_real = []
                            rews_pred = []
                            for step in range(n_steps):
                                rews_real.append(rewards[start+step+1][0])
                                obs_real_i = np.concatenate(observations[start+step+1]).tolist()
                                action = actions[start + step]
                                observation, rew_predict = trainer.model.step_single(observation, action)
                                for n,obs_real_i_i in enumerate(obs_real_i):
                                    obs_real[1][n].append(obs_real_i_i)
                                    obs_mod[1][n].append(observation[n])
                                rews_pred.append(rew_predict[0])
                            rews_pred_mult.append(rews_pred)
                    rews_real_cum = np.cumsum(rews_real)
                    rews_pred_cum_mult = [np.cumsum(rews) for rews in rews_pred_mult]
                    for item in rews_pred_cum_mult:
                        for key in rollout_mod.keys():
                            rollout_mod[key].append(item[key-1])
                            rollout_real[key].append(rews_real_cum[key-1])
            # Calculate stuff
            for key in rollout_mod.keys():
                pred = np.array(rollout_mod[key])
                target = np.array(rollout_real[key])
                error = (pred-target)
                SS_tot = np.sum((target - np.mean(target))**2)
                SS_res = np.sum(error**2)
                # RMSE = np.sqrt(np.mean(error**2))
                # print("RMSE-" + "Ep." + str(it*10) + "Rollout." + str(key) + "---" + str(RMSE))
                # print("Mean Target-" + "Ep." + str(it*10) + "Rollout." + str(key) + "---" + str(np.mean(target)))
                # print("Mean Model-" + "Ep." + str(it*10) + "Rollout." + str(key) + "---" + str(np.mean(pred)))
                # print("Overshoot-" + "Ep." + str(it*10) + "Rollout." + str(key) + "---" + str((np.mean(pred)-np.mean(target))/(np.mean(target))*100) + "%")
                #
                # print("RMSE/Mean Target-" + "Ep." + str(it*10) + "Rollout." + str(key) + "---" + str(RMSE/np.mean(target)*100) + "%")
            errors.append(1-SS_res/SS_tot)

            for i, _ in enumerate(observation):
                pred = np.array(obs_mod[1][i])
                target = np.array(obs_real[1][i])
                error = pred - target
                SS_tot = np.sum((target - np.mean(target))**2)
                SS_res = np.sum(error**2)
                errors_observations[i].append(1-SS_res/SS_tot)
        p.dump({"errors_observations":errors_observations, "errors":errors, "episodes":episodes}, open(env_name+"corr2.p", "wb"))
    else:
        dat = p.load(open(env_name+"corr2.p", "rb"))
        errors = dat["errors"]
        episodes = dat["episodes"]
        errors_observations = dat["errors_observations"]
    # Delete communication observations (unused)
    del_idx = []
    for idx, err in enumerate(errors_observations):
        if not np.any(np.array(err)>-99.):
            del_idx.append(idx)
    errors_observations = [item for j, item in enumerate(errors_observations) if j not in del_idx]
    if not add_mean_line:
        for i, errors_observation in enumerate(errors_observations):
            if i==0:
                plt.plot(episodes, errors_observation, c='tab:orange', linewidth=0.25)
            else:
                plt.plot(episodes, errors_observation, c='tab:orange', linewidth=0.25)
        plt.plot(episodes, errors, marker="o", label="Reward", c='tab:blue')
        errors_observation_mean = [np.mean(nums) for nums in zip(*errors_observations)]

        plt.plot(episodes, errors_observation_mean, marker="o", label="Observations", c='tab:orange')

        plt.xlabel("Episodes trained")
        plt.ylabel(r"R$^2$")
        plt.ylim(0.0,1.0)
        plt.xlim(0,5000)
        plt.legend()
        plt.tight_layout()
        plt.savefig("../figures/" + env_name +"_model" + ".png")
    else:
        fig, ax1 = plt.subplots(figsize=(4,2), dpi=300)

        # for i, errors_observation in enumerate(errors_observations):
        #     if i == 0:
        #         ax1.plot(episodes, errors_observation, c='tab:orange', linewidth=0.25)
        #     else:
        #         ax1.plot(episodes, errors_observation, c='tab:orange', linewidth=0.25)
        plt.plot(episodes, errors, marker="o", label="Reward ", c='tab:blue')
        errors_observation_mean = [np.mean(nums) for nums in zip(*errors_observations)]

        # plt.plot(episodes, errors_observation_mean, marker="o", label="Observations", c='tab:orange')

        ax1.set_xlabel("Episodes trained")
        ax1.set_ylabel(r"R$^2$ (Reward)")
        ax1.set_ylim(0.0, 1.0)
        ax1.set_xlim(0, 5000)
        ax1.yaxis.label.set_color('tab:blue')
        ax1.tick_params(axis='y', colors='tab:blue')
        ax2 = ax1.twinx()
        ax2.yaxis.label.set_color('tab:red')
        ax2.tick_params(axis='y', colors='tab:red')
        ax2.set_ylabel("Cumulative reward")
        ax2.set_ylim(ylim)

        plot_name_ep(data, list(data.keys())[0], 5000, "score", name="Cumulative Reward", use_moving_average=True, n_runs=5, ax=ax2, color="tab:red")

        # fig.legend()
        fig.tight_layout()
        fig.savefig("../figures/" + env_name + "_model" + ".png")

    # plt.show()


def print_model_bias(data, it, n_runs=5, env_name=None, pick=False):
    plt.figure(figsize=(4,3), dpi=300)
    plt.grid()

    errors = []
    errors_observations = None
    episodes = []
    trainer, agents, env, ep = setup_agent_env(data, it, 0, env_name=env_name)
    rollout_mod = {1: []}
    rollout_real = {1: []}
    observation = np.concatenate(env.reset())
    # print(len(observation))
    if errors_observations is None:
        errors_observations = [[] for obs in observation]
    obs_real ={1: [[] for obs in observation]}
    obs_mod = {1: [[] for obs in observation]}
    episodes.append(int(ep))
    for run in range(n_runs):
        trainer, agents, env, ep = setup_agent_env(data, it, run, env_name=env_name)
        for ep in range(250):
            score, _, statistics = run_episode(env, agents, eval=True, render=False, greedy_eval=False, store_data=True)
            # print("Score: " + str(score))
            n_steps = 1
            start = random.randint(0, 23)
            if trainer.model:
                rewards = statistics["rewards"]
                observations = statistics["observations"]
                actions = statistics["actions"]
                rews_pred_mult = []
                for rollout in range(1):
                    observation = np.concatenate(observations[start], axis=0)
                    rews_real = []
                    rews_pred = []
                    for step in range(n_steps):
                        rews_real.append(rewards[start+step+1][0])
                        obs_real_i = np.concatenate(observations[start+step+1]).tolist()
                        action = actions[start + step]
                        observation, rew_predict = trainer.model.step_single(observation, action)
                        for n,obs_real_i_i in enumerate(obs_real_i):
                            obs_real[1][n].append(obs_real_i_i)
                            obs_mod[1][n].append(observation[n])
                        rews_pred.append(rew_predict[0])
                    rews_pred_mult.append(rews_pred)
            rews_real_cum = np.cumsum(rews_real)
            rews_pred_cum_mult = [np.cumsum(rews) for rews in rews_pred_mult]
            for item in rews_pred_cum_mult:
                for key in rollout_mod.keys():
                    rollout_mod[key].append(item[key-1])
                    rollout_real[key].append(rews_real_cum[key-1])
    # Calculate stuff
    for key in rollout_mod.keys():
        pred = np.array(rollout_mod[key])
        target = np.array(rollout_real[key])
        print("Mean reward real (" + env_name + "): " , np.mean(target), "(+- )", 2*np.std(target)/np.sqrt(len(target)))
        print("Mean reward model (" + env_name + "): " , np.mean(pred), "(+- )", 2*np.std(pred)/np.sqrt(len(pred)))
        print("Bias", (np.mean(pred) - np.mean(target))/np.mean(target))

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

def setup_agent_env(data, it, run, actor_it=None, env_name="simple_spread", benchmark=False):
    env = EnvWrapper("particle", env_name, benchmark=benchmark)
    if not actor_it:
        actor_it = it
    trainer = MAMBPO(env.n_agents, env.observation_space, env.action_space,
                     use_model=True, hidden_dims_actor=(128,128),
                     hidden_dims_model=(200,200,200,200))
    agents = trainer.agents
    name_run = [k for k in data.keys()][0]
    ep = data[name_run]["runs"][run]["networks_agents"][actor_it][0]
    try:
        trainer.model.load_state_dict(data[name_run]["runs"][run]["networks_model"][it][3])
    except:
        print("Loaded no model")
    for agent_n, agent in enumerate(agents):
        agent.actor.load_state_dict(data[name_run]["runs"][run]["networks_agents"][actor_it][3][agent_n]["actor"])
        for critic, state_dict in zip(agent.critics, data[name_run]["runs"][run]["networks_agents"][it][3][agent_n]["critics"]):
            critic.load_state_dict(state_dict)
    #     if type(agent.model) == EnsembleModel and data[name_run]["runs"][run]["networks"][it][3][0]["model"] is not None:
    #         agent.model.load_state_dict(data[name_run]["runs"][run]["networks"][it][3][0]["model"])
    return trainer, agents, env, str(ep)

def setup_agent_env2(run, it, actor_it=None):
    env = EnvWrapper("gym", "HalfCheetah-v2")
    if not actor_it:
        actor_it = it
    agents = [SAC(env.observation_space[0].shape[0], env.action_space[0].shape[0], use_model=True,
                  hidden_dims_model=(200, 200, 200, 200))]
    for agent in agents:
        agent.actor.load_state_dict(run["networks"][actor_it][3][0]["actor"])
        for critic, state_dict in zip(agent.critics, run["networks"][it][3][0]["critics"]):
            critic.load_state_dict(state_dict)
        agent.model.load_state_dict(run["networks"][it][3][0]["model"])
    return agents, env

def setup_agent_env2_no_model(run, it, actor_it=None, deg=0.0):
    env = EnvWrapper("gym", "HalfCheetah-v2")
    if not actor_it:
        actor_it = it
    agents = [SAC(env.observation_space[0].shape[0], env.action_space[0].shape[0], use_model=True,
                  hidden_dims_model=(200, 200, 200, 200))]
    for agent in agents:
        agent.actor.load_state_dict(run["networks"][actor_it][3][0]["actor"])
        for critic, state_dict in zip(agent.critics, run["networks"][it][3][0]["critics"]):
            critic.load_state_dict(state_dict)
    return agents, env

def plot_all_run(data, var="score", plot_janner=True, baseline=None, baseline_name=None, var_name="Score"):
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
                plt.plot(data_temp['x'] * 1000, data_temp['y'], linewidth=1.5, label="MBPO (data from Janner et al.)",
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
    plt.ylabel(var_name)
    plt.legend()
    plt.show()


def plot_all_run_logs(logs, var="score", plot_janner=True, baseline=None, baseline_name=None,
                      use_moving_average=False,
                      var_name="Score", names=None, steps_max=5000, ylim=(0.,250.), n_runs=None, name_fig="test"):
    plt.figure(figsize=(4,3), dpi=300)
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
                plt.plot(data_temp['x'], data_temp['y'], linewidth=1.5, label="MBPO (data from Janner et al.)",
                         c=colors[alg])
                ## plot error bars
                plt.fill_between(data_temp['x'], data_temp['y'] - data_temp['std'],
                                 data_temp['y'] + data_temp['std'], color=colors[alg],
                                 alpha=0.25)
    steps_max = steps_max
    for i_log, log in enumerate(logs):
        if names:
            name = names[i_log]
        else:
            name = log
        data = torch.load("../logs/" + log + ".p", map_location="cpu")
        for key in data.keys():
                plot_name_ep(data, key, steps_max, var, name=name, use_moving_average=use_moving_average, n_runs=n_runs)
    # plot_data_csv("../logs/MAMBPO.csv", steps_max, "MAMBPO (Data by Gupta et al.)")
    plt.xlim(1., steps_max)
    plt.xlabel("Episodes trained")
    #plt.ylim(-200,-120)
    plt.ylim(*ylim)
    plt.ylabel(var_name)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("../figures/" + name_fig + ".png")


def plot_data_csv(file, steps_max, name):
    data = np.genfromtxt(file, delimiter=",", names=["x", "y"])
    plt.plot(data["x"], data["y"], label=name)

def plot_name_ep(data, key, steps_max, var, name=None, use_moving_average=False, n_runs=None, ax=None, color=None):
    steps_max = min(steps_max, data[key]["runs"][0][var][-1][0])
    steps = np.arange(0,steps_max+1,1)
    if not name:
        name=key + "(ours)"
    values = []
    if not n_runs:
        n_runs = len(data[key]["runs"])
    for run in data[key]["runs"][:n_runs]:
        # run = data[key]["runs"][0] #Only take first run
        values.append([])
        steps_var = [val[0] for val in run[var]]
        idxs = find_nearest_mult(steps_var, steps)
        for idx, step in enumerate(steps):
            # idx = find_nearest(steps_var, step)
            values[-1].append(np.mean(run[var][idxs[idx]][3]))
        values[-1] = np.array(values[-1])
    if use_moving_average:
        values = moving_average_2d(np.array(values))
    else:
        values= np.array(values)
    mean_scores = np.mean(values, axis=0)
    print(mean_scores[-1])
    std_scores = np.std(values, axis=0)/np.sqrt(values.shape[0])
    if not ax:
        plt.plot(steps, mean_scores, label=name)
        plt.fill_between(steps, mean_scores - std_scores, mean_scores + std_scores, alpha=0.25)
    else:
        ax.plot(steps, mean_scores, label=name, c=color)
        ax.fill_between(steps, mean_scores - std_scores, mean_scores + std_scores, alpha=0.25, color=color)

def plot_name(data, key, steps, var, name=None, use_moving_average=False):
    if not name:
        name=key + "(ours)"
    values = []
    for run in data[key]["runs"]:
        values.append([])
        steps_var = [val[1] for val in run[var]]
        for step in steps:
            idx = find_nearest(steps_var, step)
            values[-1].append(np.mean(run[var][idx][3][0]))
        values[-1] = np.array(values[-1])
    mean_scores = np.mean(values, axis=0)
    std_scores = np.std(values, axis=0)
    if use_moving_average:
        mean_scores = moving_average(mean_scores)
        std_scores = moving_average(std_scores)
    plt.plot(steps, mean_scores, label=name + " (ours)")
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

def calc_mse_score(env, agents):
    scores = []
    for ep in range(5):
        score, _, statistics = run_episode(env, agents, eval=True, render=False, greedy_eval=True, store_data=True,
                                           store_states=True)
        scores.append(score)
    mean_score = np.mean(scores)

    ses = []
    for ep in range(5):
        score, _, statistics = run_episode(env, agents, eval=True, render=False, greedy_eval=False, store_data=True,
                                           store_states=True)

        rewards = statistics["rewards"]
        observations = statistics["observations"]
        actions = statistics["actions"]
        states = statistics["states"]
        for i in range(20):
            start = random.randint(0, 500)
            observation = observations[start][0]
            action = actions[start][0]
            observation_next_pred, _ = agents[0].model.step_single(observation, action, state=states[start])
            observation_next = observations[start+1][0]
            ses.append(np.mean((observation_next-observation_next_pred)**2))
    mean_se = np.mean(ses)
    return mean_se, mean_score

def plot_mse_perf(logs, its):
    colors = ["red", "green", "blue", "orange", "yellow", "cyan"]
    for log in logs:
        data = torch.load("../logs/" + log + ".p", map_location="cpu")
        for name, val in data.items():
            for i, it in enumerate(its):
                for run in val["runs"]:
                    if "model" in run["networks"][it][3][0].keys() and run["networks"][it][3][0]["model"] is not None:
                        agents, env = setup_agent_env2(run, it)
                        mse, score = calc_mse_score(env, agents)
                        plt.scatter(mse, score, label=str(it), c=colors[i])
    plt.legend()
    plt.show()

def plot_mse_perf_all(logs_deg, logs, its=[1,2,4]):
    colors = ["red", "green", "blue", "orange", "yellow", "cyan"]
    for log_id, log in enumerate(logs_deg):
        print(log)
        data = torch.load("../logs/" + log + ".p", map_location="cpu")
        if "_b" in log:
            bias = float(log[-3:])
            deg = 0.0
        else:
            deg = float(log[-3:]) if "." in log[-3:] else 0.0
            bias = 0.0
        for i, it in enumerate(its):
            for name, val in data.items():
                for j, run in enumerate(val["runs"]):
                    if len(run["networks"])>it:
                        agents, env = setup_agent_env2_no_model(run, it)
                        _, env_deg = setup_agent_env2_no_model(run, it)
                        env_deg.reset()
                        agents[0].model = DegradedSim(env_deg, degradation=deg, bias=bias)
                        mse, score = calc_mse_score(env, agents)
                        if log_id == 0 and j == 0:
                            plt.scatter(mse, score, marker="x", c=colors[i], label="Fixed model, agent trained for " + str(10*it) + "ep.")
                        else:
                            plt.scatter(mse, score, marker="x", c=colors[i])

    for log in logs:
        print(log)
        data = torch.load("../logs/" + log + ".p", map_location="cpu")
        deg = float(log[-3:]) if "." in log[-3:] else 0.0
        for i, it in enumerate(its):
            for name, val in data.items():
                for j, run in enumerate(val["runs"]):
                    if len(run["networks"])>it and run["networks"][0][3][0]["model"] is not None:
                        agents, env = setup_agent_env2(run, it)
                        mse, score = calc_mse_score(env, agents)
                        if j == 0:
                            plt.scatter(mse, score, c=colors[i],
                                        label="Trained model, agent trained for " + str(10 * it) + "ep.")
                        else:
                            plt.scatter(mse, score, c=colors[i])
    plt.legend()
    plt.xlabel("1-step model prediction MSE")
    plt.ylabel("Score on Cheetah")
    plt.grid()
    plt.show()


def plot_mse_perf_all_mean(logs_deg, logs, its=[1,2,4]):
    colors = ["red", "green", "blue", "orange", "yellow", "cyan"]
    for log_id, log in enumerate(logs_deg):
        print(log)
        data = torch.load("../logs/" + log + ".p", map_location="cpu")
        if "_b" in log:
            bias = float(log[-3:])
            deg = 0.0
        else:
            deg = float(log[-3:]) if "." in log[-3:] else 0.0
            bias = 0.0
        for i, it in enumerate(its):
            for name, val in data.items():
                mses, scores = [], []
                for j, run in enumerate(val["runs"]):
                    if len(run["networks"])>it:
                        agents, env = setup_agent_env2_no_model(run, it)
                        _, env_deg = setup_agent_env2_no_model(run, it)
                        env_deg.reset()
                        agents[0].model = DegradedSim(env_deg, degradation=deg, bias=bias)
                        mse, score = calc_mse_score(env, agents)
                        mses.append(mse)
                        scores.append(score)
                if log_id == 0:
                    plt.scatter(np.mean(mses), np.mean(scores), marker="x", c=colors[i], label="Fixed model, agent trained for " + str(10*it) + "ep.")
                else:
                    plt.scatter(np.mean(mses), np.mean(scores), marker="x", c=colors[i])

    for log in logs:
        print(log)
        data = torch.load("../logs/" + log + ".p", map_location="cpu")
        deg = float(log[-3:]) if "." in log[-3:] else 0.0
        for i, it in enumerate(its):
            for name, val in data.items():
                mses, scores = [], []
                for j, run in enumerate(val["runs"]):
                    if len(run["networks"])>it and run["networks"][0][3][0]["model"] is not None:
                        agents, env = setup_agent_env2(run, it)
                        mse, score = calc_mse_score(env, agents)
                        mses.append(mse)
                        scores.append(score)
                if "model" in name:
                    plt.scatter(np.mean(mses), np.mean(scores), c=colors[i],
                                label="Trained model, agent trained for " + str(10 * it) + "ep.")
    plt.legend()
    plt.xlabel("1-step model prediction MSE")
    plt.ylabel("Score on Cheetah")
    plt.grid()
    plt.show()

def plot_mse_noise(logs, data, its):
    degs = [0.0, 0.1, 0.2, 0.3]
    it = 5
    for degradation in degs:
        for bias in [0.0,0.1]:
            agents, env = setup_agent_env(data, it, 0)
            _, env_deg = setup_agent_env(data, it, 0)
            env_deg.reset()
            model = DegradedSim(env_deg, degradation=degradation, bias=bias)
            agents[0].model = model
            mse, _ = calc_mse_score(env, agents)
            plt.scatter(degradation, mse)

    colors = ["red", "green", "blue", "orange", "yellow", "cyan"]
    for log in logs:
        data = torch.load("../logs/" + log + ".p", map_location="cpu")
        for name, val in data.items():
            for i, it in enumerate(its):
                for j, run in enumerate(val["runs"]):
                    if "model" in run["networks"][it][3][0].keys() and run["networks"][it][3][0]["model"] is not None:
                        agents, env = setup_agent_env2(run, it)
                        mse, score = calc_mse_score(env, agents)
                        if j == 0:
                            plt.hlines(mse, label=str(it), colors=colors[i], xmin=degs[0], xmax=degs[-1])
                        else:
                            plt.hlines(mse, colors=colors[i], xmin=degs[0], xmax=degs[-1])
    plt.legend()
    plt.ylabel("MSE")
    plt.xlabel("Action noise")

    plt.show()

def count_rew_greater_0(data, its, n_runs=5, env_name=None):
    for it in its:
        count_rew = 0
        count_tot = 0
        count_col = 0
        for run in range(n_runs):
            trainer, agents, env, ep = setup_agent_env(data, it, run, env_name=env_name, benchmark=True)
            for i in range(250):
                score, _, statistics = run_episode(env, agents, eval=True, render=False, greedy_eval=False, store_data=True, benchmark=True)
                if env_name == "simple_tag_coop":
                    if score[0] > 0:
                        count_rew += 1
                if env_name == "simple_spread":
                    covered = False
                    collision = False
                    for step_data in statistics:
                        if step_data["n"][0][3]==3:
                            covered = True
                        if step_data["n"][0][1] > 1 or step_data["n"][1][1] > 1 or step_data["n"][1][1] > 1:
                            collision = True
                    if covered and not collision:
                        count_rew += 1
                    if collision:
                        count_col += 1
                count_tot += 1
        print("Episode: ", ep)
        print(count_tot)
        print(count_rew)
        print("Chance of success:", float(count_rew)/float(count_tot))
        print("Chance of collision", float(count_col)/float(count_tot))

        # plot_name_ep(data, key, steps_max, var, name=log, use_moving_average=use_moving_average)

# Plot cheetah results
# logs_nav = ["cheetah_verify5"]
# names = ["MAMBPO (ours)"]
# plot_all_run_logs(logs_nav, var="score", plot_janner=True, var_name="Cumulative reward per episode",
#                   names=names,
#                   steps_max=50,
#                   ylim=(0, 12000), n_runs=3, name_fig="cheetah_results")
# plt.savefig("../figures/cheetah_results.png")

# First plot nav results
# logs_nav = ["nav_mambpo_10step", "nav_masac_5run", "nav_masac_10step"] #, "nav_masac_1_auto_2", "nav_mambpo_cheetah2"]
# names = ["MAMBPO (10 steps)", "MAMBPO (1 step)", "MAMBPO (10 steps)"]
# plot_all_run_logs(logs_nav, var="score", plot_janner=False, use_moving_average=True, var_name="Cumulative reward per episode",
#                   names=names,
#                   steps_max=5000,
#                   ylim=(-190,-120), n_runs=5, name_fig="nav_results")
# plt.savefig("../figures/nav_results.png")
#
#Plot tag results
logs_nav = ["tag_masac_5run"]#, "tag_masac_1_auto_2"]

#names = ["MAMBPO", "MAMBPO"]
plot_all_run_logs(logs_nav, var="score", plot_janner=False, use_moving_average=True, var_name="Cumulative reward per episode",
                  names=None,
                  steps_max=10000,
                  ylim=(0, 250), n_runs=5, name_fig="tag_results")
plt.savefig("../figures/tag_results.png")
plt.show()
# #
#
# name = "nav_mambpo_10step"
# # # name = "nav_masac_5run"
# #
# # name_run = "SAC4"
# env_name="simple_spread"
# data = torch.load("../logs/" + name + ".p", map_location="cpu")
# # play_game(data, 20, 0, env_name=env_name, steps=5, name="nav_env")
# #
# # # play_game(data, 20, 0, env_name=env_name)
# # count_rew_greater_0(data, [i for i in [0,5,10,20]], n_runs=5, env_name=env_name)
# plot_model_statistics_r22(data, [i for i in range(0,21,1)], n_runs=5, env_name=env_name, pick=True, add_mean_line=True, ylim=(-190,-120))
# #print_model_bias(data, 20, n_runs=5, env_name=env_name)
#
# # plt.show()
# #
# # name = "nav_masac_5run"
# # name_run = "SAC4"
# # env_name="simple_spread"
# # data = torch.load("../logs/" + name + ".p", map_location="cpu")
# # # count_rew_greater_0(data, [i for i in  [0,5,10,20]], n_runs=2, env_name=env_name)
# #
# name = "tag_mambpo_10step"
# # name = "tag_masac_5run"
#
# # name_run = "SAC4"
# env_name="simple_tag_coop"
# data = torch.load("../logs/" + name + ".p", map_location="cpu")
# # play_game(data, 20, 0, env_name=env_name, steps=5, name="tag_env")
# # play_game(data, 20, 0, env_name=env_name, steps=5, name="tag_env2")
# # play_game(data, 20, 0, env_name=env_name, steps=5, name="tag_env3")
# # play_game(data, 20, 0, env_name=env_name, steps=5, name="tag_env4")
# # play_game(data, 20, 0, env_name=env_name)
# plot_model_statistics_r22(data, [i for i in range(0,21,1)], n_runs=5, env_name=env_name, pick=True, add_mean_line=True, ylim=(0,250))
# # count_rew_greater_0(data, [i for i in  [0,5,10,20]], n_runs=5, env_name=env_name)

# print_model_bias(data, 20, n_runs=5, env_name=env_name)

# plt.show()
#
# count_rew_greater_0(data, [i for i in [0,5,10,20]], n_runs=2, env_name=env_name)
#
# name = "tag_masac_5run"
# name_run = "SAC4"
# env_name="simple_tag_coop"
# data = torch.load("../logs/" + name + ".p", map_location="cpu")
# count_rew_greater_0(data, [i for i in  [0,5,10,20]], n_runs=2, env_name=env_name)
#
# plot_model_statistics_r2(data, [i for i in range(0,21,2)], n_runs=5, env_name=env_name)
