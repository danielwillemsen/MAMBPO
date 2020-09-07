import matplotlib.pyplot as plt
import numpy as np

#files = ["test.log", "test_smallactor.log", "test_ddpg.log", "test_lr.log", "test_alpha.log"]
#files = ["test_steps2.log", "test_steps3.log","test_steps4.log","test_steps5.log","test_steps6.log"]
files = ["blabla.log"]

def update_var(line, var, name, type):
    if name+":" in line:
        if not "greedy" in line:
            var = type(line[line.find(name+":")+len(name+":"):])
    return var

def moving_average(a, n=50) :
    b = np.zeros(a.size)
    for i in range(len(a)):
        if i>=n:
            b[i] = np.mean(a[i-n:i+1])
        else:
            b[i] = np.mean(a[0:i+1])
    return b

data = dict()
for file in files:
    with open('../logs/' + file) as f:
        n_agents = None
        run = None
        agent = None
        ep = None
        time = None
        score = "[0]"
        test_loss_o = "tensorrrr"
        test_loss_r = "tensorrrr"
        train_loss_o = "tensorrrr"
        train_loss_r = "tensorrrr"
        score_agents = ["" for agent in range(1)]
        steps = ""
        for line in f:
            line = line[0:-1]
            if True:
                if "run" in line:
                    run = int(line[-1])
                ep = update_var(line, ep, "episode", int)
                n_agents = update_var(line, n_agents, "n_agents", int)
                agent = update_var(line, agent, "agent", str)
                no_model_str = "'use_model': False"
                model_str = "'use_model': True"
                if no_model_str in line:
                    print(line)
                    agent = no_model_str+str(n_agents)
                    if agent in data.keys() and False:
                        if type(data[agent]) is dict:
                            data[agent] = [data[agent]]
                        data[agent].append({"ep": [], "time": [], "score":[], "test_loss_r":[], "train_loss_r":[],
                                       "test_loss_o":[], "train_loss_o":[], "steps":[], "score_agent":[[] for agent in range(n_agents)]})
                        score_agents = ["" for agent in range(n_agents)]

                    else:
                        data[agent] = {"ep": [], "time": [], "score":[], "test_loss_r":[], "train_loss_r":[],
                                       "test_loss_o":[], "train_loss_o":[], "steps":[], "score_agent":[[] for agent in range(n_agents)]}
                        score_agents = ["" for agent in range(n_agents)]

                if model_str in line:
                    print(line)
                    agent = model_str+str(n_agents)
                    if agent in data.keys():
                        if type(data[agent]) is dict:
                            data[agent] = [data[agent]]
                        data[agent].append({"ep": [], "time": [], "score":[], "test_loss_r":[], "train_loss_r":[],
                                       "test_loss_o":[], "train_loss_o":[], "steps":[], "score_agent":[[] for agent in range(n_agents)]})
                    else:
                        data[agent] = {"ep": [], "time": [], "score":[], "test_loss_r":[], "train_loss_r":[],
                                       "test_loss_o":[], "train_loss_o":[], "steps":[], "score_agent":[[] for agent in range(n_agents)]}
                time = update_var(line, time, "time_elapsed", float)
                score = update_var(line, score, "score", str)
                test_loss_r = update_var(line, test_loss_r, "test_loss_r", str)
                test_loss_o = update_var(line, test_loss_o, "test_loss_o", str)
                train_loss_r = update_var(line, train_loss_r, "train_loss_r", str)
                train_loss_o = update_var(line, train_loss_o, "train_loss_o", str)
                for i in range(len(score_agents)):
                    score_agents[i] = update_var(line, score_agents[i], "score_agent"+str(i), str)
                steps = update_var(line, steps, "step_tot", str)
                if "step_tot" in line and type(data[agent]) is list:
                    data[agent][run]["ep"].append(ep)
                    #data[agent][run]["score"].append(float(score[1:-1]))
                    #data[agent+file]["test_loss_o"].append(float(test_loss_o[7:-18]))
                    #data[agent+file]["train_loss_o"].append(float(train_loss_o[7:-18]))
                    #data[agent+file]["test_loss_r"].append(float(test_loss_r[7:-18]))
                    #data[agent+file]["train_loss_r"].append(float(train_loss_r[7:-18]))
                    data[agent][run]["steps"].append(float(steps))
                    data[agent][run]["time"].append(time)
                    for i in range(len(score_agents)):
                        data[agent][run]["score_agent"][i].append(float(score_agents[i]))


                elif "step_tot" in line: #not train_loss_r == "tensorrrr" :
                    data[agent]["ep"].append(ep)
                    #data[agent]["score"].append(float(score[1:-1]))
                    #data[agent+file]["test_loss_o"].append(float(test_loss_o[7:-18]))
                    #data[agent+file]["train_loss_o"].append(float(train_loss_o[7:-18]))
                    #data[agent+file]["test_loss_r"].append(float(test_loss_r[7:-18]))
                    #data[agent+file]["train_loss_r"].append(float(train_loss_r[7:-18]))
                    data[agent]["steps"].append(int(steps))

                    data[agent]["time"].append(time)
                    for i in range(len(score_agents)):
                        data[agent]["score_agent"][i].append(float(score_agents[i]))


# for name, agent in data.items():
#     if len(agent["ep"])>0:
#         plt.plot(agent["ep"], agent["score"], label=name)
# plt.legend()
# plt.show()
# plt.show()

# for name, agent in data.items():
#     if "ep" in agent.keys() and len(agent["ep"])>0:
#         plt.plot(agent["steps"], moving_average(np.asarray(agent["score"]),1), label=name[0:25])
# plt.xlim([0.,50000.])
#plot original reults
tasks = ["cheetah"]
algorithms = ['mbpo']

colors = {
	'mbpo': '#3a22b4',
}
data2 = data
# import pickle
# for task in tasks:
#     for alg in algorithms:
#         print(task, alg)
#
#         ## load results
#         fname = '../logs/mbpo_v1.5_results/{}_{}.pkl'.format(task, alg)
#         data = pickle.load(open(fname, 'rb'))
#
#         ## plot trial mean
#         plt.plot(data['x']*1000, data['y'], linewidth=1.5, label=alg+" (Janner et al.)", c=colors[alg])
#         ## plot error bars
#         plt.fill_between(data['x']*1000, data['y']-data['std'], data['y']+data['std'], color=colors[alg], alpha=0.25)
# import pickle
# for task in tasks:
#     for alg in algorithms:
#         print(task, alg)
#
#         ## load results
#         fname = '../logs/mbpo_v1.5_results/{}_{}.pkl'.format(task, alg)
#         data = pickle.load(open(fname, 'rb'))
#
#         ## plot trial mean
#         plt.plot(data['x']*1000, data['y'], linewidth=1.5, label=alg+" (Janner et al.)", c=colors[alg])
#         ## plot error bars
#         plt.fill_between(data['x']*1000, data['y']-data['std'], data['y']+data['std'], color=colors[alg], alpha=0.25)

for name, agent in data.items():
    # Calculate mean
    dat = [np.mean(items) for items in zip(*data[name]["score_agent"])]
    plt.plot(dat, label=name)
    # for i in range(len(data[name]["score_agent"])):
    #     plt.plot(data[name]["score_agent"][i], label=name + str(i))
plt.legend()
plt.show()

# for name, agent in data2.items():
#     #if len(agent[-1]["ep"]) < 50:
#     #    del agent[-1]
#     if type(agent) == list:
#         score_lists = [np.array(run["score"]) for run in agent]
#         mean_scores = np.mean(score_lists, axis=0)
#         std_scores = np.std(score_lists, axis=0)
#         if "True" in name:
#             label = "mbpo"
#         else:
#             label = "SAC"
#         plt.plot(agent[0]["steps"], mean_scores, label=label + " (ours)")
#         plt.fill_between(agent[0]["steps"], mean_scores-std_scores, mean_scores+std_scores, alpha=0.25)
#
# plt.legend()
# plt.xlabel("Real environment steps")
# plt.ylabel("Score")
# plt.show()
#
# for name, agent in data.items():
#     if len(agent["ep"])>0:
#         plt.plot(agent["steps"], agent["time"], label=name)
#
#
# plt.legend()
# plt.show()
# for name, agent in data.items():
#     if len(agent["ep"])>0:
#         plt.semilogy(agent["ep"], agent["train_loss_r"], label="train-r")
#         plt.semilogy(agent["ep"], agent["test_loss_r"], label="test-r")
#         plt.semilogy(agent["ep"], agent["train_loss_o"], label="train-o")
#         plt.semilogy(agent["ep"], agent["test_loss_o"], label="test-o")
# plt.ylim(0.01, 1.0)
# plt.legend()
# plt.show()