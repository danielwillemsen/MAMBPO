import seaborn as sns
import pandas as pd
import numpy as np
import pickle as p
import matplotlib.pyplot as plt

sns.set(style="darkgrid")

data = dict()
files = []
for dat in ["../logs/mb_bigtest_cheetah3"]:
    if data:
        data.update(p.load(open(dat, "rb")))
    else:
        data = p.load(open(dat,"rb"))

frames = dict()
for key, dat in data.items():
    frame = pd.DataFrame(columns=["episode", "time", "steps", "score"])
    for run in dat:
        run_score = run["scores"]
        run_step = run["steps"]
        run_time = run["times"]
        for ep, perf in enumerate(run_score):
            frame.loc[len(frame)] = [ep, run_time[ep], run_step[ep], np.mean(perf)]
    frames[key] = frame

for name, frame in frames.items():
    # if "HDDPGAgent" in name:
    #     name = "DDPG"
    # elif "Model" in name:
    #     name = "Model-Based TD3"
    # else:
    #     name = "TD3"
    ax = sns.lineplot(x="episode", y="score", data=frame, label=name)
    #ax.set(ylim=(,1)
    #ax.set(yscale="log")
plt.show()


for name, frame in frames.items():
    ax = sns.lineplot(x="episode", y="time", data=frame, label=name)
    #ax.set(ylim=(,1)
    #ax.set(yscale="log")

plt.show()

#fmri = sns.load_dataset("fmri")
##data = pd.DataFrame(data=data)

#sns
