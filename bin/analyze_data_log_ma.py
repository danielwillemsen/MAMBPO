import pickle
import matplotlib.pyplot as plt
import numpy as np
from decentralizedlearning.algs.sac import SAC
from decentralizedlearning.envwrapper import EnvWrapper
from decentralizedlearning.run_utils import run_episode
from decentralizedlearning.algs.configs import config
from decentralizedlearning.algs.models import DegradedSim
from decentralizedlearning.algs.models import EnsembleModel
from decentralizedlearning.algs.masac import MASAC, MASACAgent
import torch
import random
from scipy.stats import pearsonr
import pyglet

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

def play_game(data, it, run=0):
    trainer, agents, env, _ = setup_agent_env(data, it, run)

    score, _, statistics = run_episode(env, agents, eval=True, render=True, greedy_eval=False, store_data=True)
    pyglet.image.get_buffer_manager().get_color_buffer().save("test.png")
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
    plt.ylim(0.5,1.0)
    plt.legend()
    plt.show()

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

def setup_agent_env(data, it, run, actor_it=None, env_name="simple_spread"):
    env = EnvWrapper("particle", env_name)
    if not actor_it:
        actor_it = it
    trainer = MASAC(env.n_agents, env.observation_space, env.action_space,
                    use_model=True, hidden_dims_actor=(128,128),
                    hidden_dims_model=(200,200,200,200))
    agents = trainer.agents
    ep = data[name_run]["runs"][run]["networks_agents"][actor_it][0]
    trainer.model.load_state_dict(data[name_run]["runs"][run]["networks_model"][it][3])
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
    plt.ylabel(var_name)
    plt.legend()
    plt.show()


def plot_all_run_logs(logs, var="score", plot_janner=True, baseline=None, baseline_name=None,
                      use_moving_average=False,
                      var_name="Score", names=None, steps_max=5000, ylim=(0.,250.)):
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
    steps_max = steps_max
    plt.figure(figsize=(5,4))
    for i_log, log in enumerate(logs):
        if names:
            name = names[i_log]
        else:
            name = log
        data = torch.load("../logs/" + log + ".p", map_location="cpu")
        for key in data.keys():
                plot_name_ep(data, key, steps_max, var, name=name, use_moving_average=use_moving_average)
    # plot_data_csv("../logs/MASAC.csv", steps_max, "MASAC (Data by Gupta et al.)")
    plt.xlim(1., steps_max)
    plt.xlabel("Episode")
    #plt.ylim(-200,-120)
    plt.ylim(*ylim)
    plt.ylabel(var_name)
    plt.legend()
    plt.grid()
    plt.show()

def plot_data_csv(file, steps_max, name):
    data = np.genfromtxt(file, delimiter=",", names=["x", "y"])
    plt.plot(data["x"], data["y"], label=name)

def plot_name_ep(data, key, steps_max, var, name=None, use_moving_average=False):
    steps_max = min(steps_max, data[key]["runs"][0][var][-1][0])
    steps = np.arange(0,steps_max,1)
    if not name:
        name=key + "(ours)"
    values = []
    for run in data[key]["runs"]:
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
    std_scores = np.std(values, axis=0)/np.sqrt(values.shape[0])
    # if use_moving_average:
    #     mean_scores = moving_average(mean_scores)
    #     std_scores = moving_average(std_scores)
    if "actor" in name:
        name = "MASAC (Shared Critic and Actor)"
    # elif "critic" in name:
    #     name = "MASAC (Shared Critic)"
    # elif "sac" in name:
    #     name = "SAC"
    # elif "navigation_sharebuffer" in name:
    #     name = "SAC (Shared Buffer)"
    elif "no_model" in name:
        name = "Multi-Agent Actor-Critic"
    elif "model" in name:
        name = "Multi-Agent Model-Based Actor-Critic"
    elif "tag_masac" in name:
        name = "Multi-Agent Actor-Critic"

    # else:
    #     name = "MASAC"# (Model-Based)"
    plt.plot(steps, mean_scores, label=name)
    plt.fill_between(steps, mean_scores - std_scores, mean_scores + std_scores, alpha=0.25)

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

def count_rew_greater_0(logs):
    for log in logs:
        data = torch.load("../logs/" + log + ".p", map_location="cpu")
        for key in data.keys():
            count_tot = 0
            count_rew = 0
            for run in data[key]["runs"]:
                for (ep, _, _, scores) in run["score"]:
                    if ep>1000 and ep<=2000:
                        if scores[0] > 0:
                            count_rew += 1
                        count_tot += 1
            print(log, count_tot)
            print(log, count_rew)
            print("Chance of catching prey:", float(count_rew)/float(count_tot))

        # plot_name_ep(data, key, steps_max, var, name=log, use_moving_average=use_moving_average)

#First plot nav results
logs_nav = ["cheetah_verify4"]
plot_all_run_logs(logs_nav, var="score", plot_janner=True, var_name="Return",
                  names=None,
                  steps_max=50,
                  ylim=(0, 6000))

logs_nav = ["nav_mambpo_5step_5run_lr", "nav_masac_5run"] #, "nav_masac_1_auto_2", "nav_mambpo_cheetah2"]
names = ["MAMBPO", "MASAC"]
plot_all_run_logs(logs_nav, var="score", plot_janner=False, use_moving_average=True, var_name="Reward",
                  names=None,
                  steps_max=5000,
                  ylim=(-180,-120))
plt.savefig("../figures/nav_results.png")

#Plot tag results
logs_nav = ["tag_mambpo_5step_5run_lr", "tag_mambpo_5step_5run", "tag_mambpo_5step_5run_lr2", "tag_mambpo_5step_5run_4","tag_masac_model", "tag_masac_1_auto_2"]
logs_nav = ["tag_mambpo_5step_5run_lr", "tag_masac_5run"]#, "tag_masac_1_auto_2"]

names = ["MAMBPO", "MASAC"]
plot_all_run_logs(logs_nav, var="score", plot_janner=False, use_moving_average=True, var_name="Reward",
                  names=None,
                  steps_max=10000,
                  ylim=(0, 250))
plt.savefig("../figures/tag_results.png")



name = "tag_mambpo_5step_5run_lr"
#cheetah_plotmodel_4layers_regulated_2
name_run = "SAC4"
env_name="simple_tag_coop"
data = torch.load("../logs/" + name + ".p", map_location="cpu")
analyze_model(data, 0, run=0, env_name=env_name)
analyze_model_statistics(data, 20, run=0, env_name=env_name)
plot_model_statistics_corr(data, [i*4 for i in range(11)], run=0, env_name=env_name)

# logs = ["3_agent_model", "3_agent", "4_agent", "4_agent_lowlr"]
# logs = [ "4_agent", "4_agent_lowlr"]
# logs = [ "4_agent_new", "4_agent_lowlr"]
#
# logs = ["particle_long", "particle_long_common_critic", "particle_long_common_critic_common_actor", "particle_long_sac", "navigation_sharebuffer", "navigation_5_step_no_model", "navigation_5_step_model","navigation_20_step_no_model"]#, "4_agent_5", "4_agent_grad1", "4_agent_grad0.5", "4_agent_5_2"]
# # logs = ["particle_comm", "particle_comm_sac"]#, "particle_long_common_critic_common_actor", "particle_long_sac"]#, "4_agent_5", "4_agent_grad1", "4_agent_grad0.5", "4_agent_5_2"]
# # logs = ["particle_long", "particle_long_common_critic", "particle_long_common_critic_common_actor", "particle_long_sac", "navigation_sharebuffer", "navigation_5_step_no_model", "navigation_5_step_model","navigation_20_step_no_model"]#, "4_agent_5", "4_agent_grad1", "4_agent_grad0.5", "4_agent_5_2"]
# logs = ["particle_long", "particle_long_sac", "navigation_sharebuffer"]
# plot_all_run_logs(logs, var="mean_score_greedy", plot_janner=False, use_moving_average=True, var_name="Reward")
#
# logs = ["navigation_no_model_improved_4", "navigation_model_improved_4","navigation_no_model_improved_4_sac", "navigation_model_improved_4_nonorm_lowlr", "navigation_no_model_improved_4_nonorm_lowlr", "navigation_no_model_improved", "navigation_model_improved"]
# plot_all_run_logs(logs, var="mean_score_greedy", plot_janner=False, use_moving_average=True, var_name="Reward")
#

#play_game(data, 40, 0)
# logs = ["navigation_model_improved_4_nonorm_lowlr", "navigation_no_model_improved_4_nonorm_lowlr", "navigation_model_improved_20_nonorm_lowlr", "navigation_model_cheetah"]
logs = ["navigation_no_model_improved_4_nonorm_lowlr","navigation_model_cheetah_simple",
        # "navigation_model_cheetah_correct", , "navigation_model_cheetah",
        # "navigation_model_cheetah_simple_2",
        # "navigation_model_cheetah_simple_200",
        # "navigation_model_cheetah_simple_512",
        # "navigation_model_cheetah_simple_lowreg",
        "navigation_model_cheetah_improved",
        "navigation_no_model_cheetah_improved",
        "particle_long"]
logs = ["tag_masac_model", "tag_masac", "tag_masac_model_10", "tag_masac_1_auto_2"]
# count_rew_greater_0(logs)

#plot_all_run_logs(logs, var="score", plot_janner=False, use_moving_average=True, var_name="Reward", steps_max=10000)
logs = ["navigation_model_cheetah_improved", "navigation_no_model_cheetah_improved", "nav_masac_model", "nav_masac_model_5", "nav_masac_model_5_500","nav_masac"]
#plot_all_run_logs(logs, var="score", plot_janner=False, use_moving_average=True, var_name="Reward")
logs = [ "nav_masac_model_5_500", "nav_masac_model_5_1000", "nav_masac_5_500", "nav_masac_model_10_1000", "nav_masac"]
logs = ["nav_masac_model_5_500", "nav_masac_5_500", "nav_masac", "nav_masac_model_10_500", "nav_masac_1_noauto_0.05", "nav_masac_1_noauto_0.1", "nav_masac_1_noauto_0.02", "nav_masac_model_10_500_noauto", "nav_masac_1_auto_2"]
logs = ["nav_masac_model_5_500", "nav_masac", "nav_masac_1_auto_2", "nav_masac_1_noauto_0.02"]

#plot_all_run_logs(logs, var="score", plot_janner=False, use_moving_average=True, var_name="Reward", steps_max=10000)
