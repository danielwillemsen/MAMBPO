import seaborn as sns
import pandas as pd
import numpy as np
import pickle as p
import matplotlib.pyplot as plt

sns.set(style="darkgrid")

data = p.load(open("../logs/exp_mult5", "rb"))

frames = dict()
for key, dat in data.items():
    frame = pd.DataFrame(columns=["episode", "score"])
    for run in dat:
        for ep, perf in enumerate(run):
            frame.loc[len(frame)] = [ep, perf]
    frames[key] = frame

for name, frame in frames.items():
    ax = sns.lineplot(x="episode", y="score", data=frame, label=name)
    #ax.set(ylim=(,1)
    #ax.set(yscale="log")

plt.show()
#fmri = sns.load_dataset("fmri")
##data = pd.DataFrame(data=data)

#sns