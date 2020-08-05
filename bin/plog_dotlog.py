import matplotlib.pyplot as plt
import numpy as np

#files = ["test.log", "test_smallactor.log", "test_ddpg.log", "test_lr.log", "test_alpha.log"]
#files = ["test_steps2.log", "test_steps3.log","test_steps4.log","test_steps5.log","test_steps6.log"]
files = ["test_steps8.log"]

def update_var(line, var, name, type):
    if name in line:
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
        run = None
        agent = None
        ep = None
        time = None
        score = "[0]"
        test_loss_o = "tensorrrr"
        test_loss_r = "tensorrrr"
        train_loss_o = "tensorrrr"
        train_loss_r = "tensorrrr"
        steps = ""
        for line in f:
            line = line[0:-1]
            if not run:
                if "run" in line:
                    run = int(line[-1])
                ep = update_var(line, ep, "episode", int)
                agent = update_var(line, agent, "agent", str)
                if "agent" in line:
                    data[agent+file] = {"ep": [], "time": [], "score":[], "test_loss_r":[], "train_loss_r":[],
                                   "test_loss_o":[], "train_loss_o":[], "steps":[]}
                time = update_var(line, time, "time_elapsed", float)
                score = update_var(line, score, "score", str)
                test_loss_r = update_var(line, test_loss_r, "test_loss_r", str)
                test_loss_o = update_var(line, test_loss_o, "test_loss_o", str)
                train_loss_r = update_var(line, train_loss_r, "train_loss_r", str)
                train_loss_o = update_var(line, train_loss_o, "train_loss_o", str)
                steps = update_var(line, steps, "step_tot", str)

                if "step_tot" in line: #not train_loss_r == "tensorrrr" :
                    data[agent+file]["ep"].append(ep)
                    data[agent+file]["score"].append(float(score[1:-1]))
                    #data[agent+file]["test_loss_o"].append(float(test_loss_o[7:-18]))
                    #data[agent+file]["train_loss_o"].append(float(train_loss_o[7:-18]))
                    #data[agent+file]["test_loss_r"].append(float(test_loss_r[7:-18]))
                    #data[agent+file]["train_loss_r"].append(float(train_loss_r[7:-18]))
                    data[agent+file]["steps"].append(float(steps))

                    data[agent+file]["time"].append(time)

for name, agent in data.items():
    if len(agent["ep"])>0:
        plt.plot(agent["ep"], agent["score"], label=name)
plt.legend()
plt.show()

for name, agent in data.items():
    if len(agent["ep"])>0:
        plt.plot(agent["steps"], moving_average(np.asarray(agent["score"]),10), label=name)
plt.legend()
plt.show()
for name, agent in data.items():
    if len(agent["ep"])>0:
        plt.plot(agent["steps"], agent["time"], label=name)
plt.legend()
plt.show()
for name, agent in data.items():
    if len(agent["ep"])>0:
        plt.semilogy(agent["ep"], agent["train_loss_r"], label="train-r")
        plt.semilogy(agent["ep"], agent["test_loss_r"], label="test-r")
        plt.semilogy(agent["ep"], agent["train_loss_o"], label="train-o")
        plt.semilogy(agent["ep"], agent["test_loss_o"], label="test-o")
plt.ylim(0.01, 1.0)
plt.legend()
plt.show()