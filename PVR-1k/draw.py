import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

types = ['Regret', 'Reward']
methods = ['ConUCB', 'LinUCB', 'SemiUCB']
x = np.arange(100)
inter = 5
ft=14
w=2
data = dict()
plt.figure(figsize=(5, 4))
for m in methods:
    with open(f'result/Regret_{m}.pkl', 'rb') as f:
        data[m] = dict()
        file = np.asarray(pkl.load(f))
        data[m]['mean'] = np.mean(file, axis=0)
        data[m]['ste'] = np.std(file, axis=0) / np.sqrt(file.shape[-1])
    plt.errorbar(x[::inter], data[m]['mean'][::inter], yerr=data[m]['ste'][::inter], label=m,linewidth=w)

plt.xlabel('Rounds',fontsize=ft)
plt.ylabel('Cumulative Regret',fontsize=ft)
plt.xticks(fontsize=ft)
plt.yticks(fontsize=ft)
plt.subplots_adjust(left=0.2,right=0.95,bottom=0.2,top=0.98)
plt.legend(fontsize=ft)
plt.savefig('FineResult/regret.pdf')


data = dict()
plt.figure(figsize=(5, 4))
for m in methods:
    with open(f'result/Reward_{m}.pkl', 'rb') as f:
        data[m] = dict()
        file = np.asarray(pkl.load(f))
        file = np.cumsum(file, axis=1) / (x.reshape(1, -1) + 1)
        data[m]['mean'] = np.mean(file, axis=0)
        data[m]['ste'] = np.std(file, axis=0) / np.sqrt(file.shape[-1])
    plt.errorbar(x[::inter], data[m]['mean'][::inter], yerr=data[m]['ste'][::inter], label=m,linewidth=w)

plt.xlabel('Rounds',fontsize=ft)
plt.ylabel('Cumulative Reward',fontsize=ft)
plt.xticks(fontsize=ft)
plt.yticks(fontsize=ft)
plt.subplots_adjust(left=0.2,right=0.95,bottom=0.2,top=0.98)
plt.legend(fontsize=ft)
plt.savefig('FineResult/reward.pdf')

