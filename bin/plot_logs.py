import seaborn as sns
import pandas as pd
import numpy as np
import pickle as p
import matplotlib.pyplot as plt

sns.set(style="darkgrid")

data = p.load(open("./logs/delay", "rb"))

frames = dict()
for key, dat in data.items():
    frame = pd.DataFrame(columns=["episode", "time", "score"])
    for run in dat:
        run_score = run["scores"]
        run_time = run["times"]
        for ep, perf in enumerate(run_score):
            frame.loc[len(frame)] = [ep, run_time[ep], perf[0]]
    frames[key] = frame

for name, frame in frames.items():
    if name == "HDDPGAgent":
        name = "DDPG"
    elif name == "ModelAgent":
        name = "Model-Based TD3"
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
