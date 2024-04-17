import numpy as np
import pickle
import pandas as pd
from numpy.linalg import inv
import matplotlib.pyplot as plt
import datetime
from algorithms import MAB,LinUCB,ConUCB,SemiUCB
from tqdm import tqdm

def offlineEvaluate(mab, user, n_arms, rewards, rewards_ATTR,rewards_CFG, contexts, nrounds, noise=0.1):
	# array to contain chosen arms in offline mode
	cum_regret = np.zeros(nrounds)
	# rewards of each chosen arm
	reward_arms = np.zeros(nrounds)
	# cumulative reward at each iteration
	note_reward = np.zeros(nrounds)
	# initialize overall cumulative reward to zero
	action_list=[]
	regret=0
	for i in range(nrounds):
		np.random.seed(i)
		c,a1,a2 = mab.play(contexts)
		vrew=rewards[c][a1][a2] and not (np.random.rand()<noise)
		arew=rewards_ATTR[a1][a2]
		crew=rewards_CFG[c]
		# print(vrew,arew,crew)
		# get the reward of chosen arm at round T
		# trew = vrew and not (np.random.rand()<noise)
		# reward_arms[i] = vrew

		mab.update(c,a1,a2,vrew,arew,crew,contexts)
		# update overall cumulative reward
		# update cumulative reward of round T 
		note_reward[i] = vrew
		regret+=1-vrew
		cum_regret[i]=(regret)
		action_list.append((user,i,c,a1,a2))

	return note_reward, cum_regret, action_list


with open('user2Vis.pkl','rb') as f:
	user2Vis=pickle.load(f)
with open('configAll.pkl','rb') as f:
	configAll=pickle.load(f)
with open('user2AttrAll.pkl','rb') as f:
	user2AttrAll=pickle.load(f)
with open('attr2vec.pkl','rb') as f:
	attr2vec=pickle.load(f)
with open('userLikeATTR.pkl','rb') as f:
	userLikeATTR=pickle.load(f)
with open('userLikeCFG.pkl','rb') as f:
	userLikeCFG=pickle.load(f)

# print(user2Vis)
n_cfg=len(configAll)

t=0
repeat=5
n_rounds=200
ConUCB_Reward_Iter=[]
ConUCB_Regret_Iter=[]
LinUCB_Reward_Iter=[]
LinUCB_Regret_Iter=[]
SemiUCB_Reward_Iter=[]
SemiUCB_Regret_Iter=[]
SemiNoBias_Reward_Iter=[]
SemiNoBias_Regret_Iter=[]
SemiNoHier_Reward_Iter=[]
SemiNoHier_Regret_Iter=[]
for rep_t in range(repeat):
	ALL_reward_ConUCB=[]
	ALL_reward_LinUCB=[]
	ALL_reward_SemiUCB=[]
	ALL_reward_SemiNoBias=[]
	ALL_reward_SemiNoHier=[]
	CUM_regret_ConUCB=[]
	CUM_regret_LinUCB=[]
	CUM_regret_SemiUCB=[]
	CUM_regret_SemiNoBias=[]
	CUM_regret_SemiNoHier=[]

	Action_ConUCB=[]
	Action_LinUCB=[]
	Action_SemiUCB=[]	
	for user,visList in tqdm(user2Vis.items()):
		if user in user2AttrAll.keys():
			attrList=user2AttrAll[user]
			if len(attrList)<20:
				arms = attrList
				# print(attrList)
				n_arms=len(arms)
				rewards=np.zeros((n_cfg,n_arms,n_arms))
				CFG_reward=np.zeros(n_cfg)
				ATTR_reward=np.zeros((n_arms,n_arms))
				for c in range(n_cfg):
					if (configAll[c] in userLikeCFG[user]):
						CFG_reward[c]=1
					for i in range(n_arms):
						for j in range(n_arms):
							if ((arms[i],arms[j]) in userLikeATTR[user]):
								ATTR_reward[i][j]=1							
							if ((arms[i],arms[j],configAll[c]) in visList):
								rewards[c][i][j]=1	
				
				contexts=[]
				for attr in attrList:
					contexts.append(attr2vec[attr])
				contexts=np.array(contexts)*500
				mab1 = ConUCB(narms=n_arms,nsups=n_cfg, ndims=10, alpha=0.5,rho=1)
				mab2 = LinUCB(narms=n_arms,nsups=n_cfg, ndims=10, alpha=0.5,rho=1)
				mab3 = SemiUCB(narms=n_arms,nsups=n_cfg, ndims=10, alpha=0.5,rho=1)
				mab4 = SemiUCB(narms=n_arms,nsups=n_cfg, ndims=10, alpha=0.5,rho=1,bias_k=0)
				mab5 = SemiUCB(narms=n_arms,nsups=n_cfg, ndims=10, alpha=0.5,rho=1,hier=False)
				note_reward_ConUCB, cum_regret_ConUCB, act_ConUCB = offlineEvaluate(mab1, user, arms, rewards, ATTR_reward, CFG_reward, contexts, n_rounds)
				note_reward_LinUCB, cum_regret_LinUCB, act_LinUCB = offlineEvaluate(mab2, user, arms, rewards, ATTR_reward, CFG_reward, contexts, n_rounds)
				note_reward_SemiUCB, cum_regret_SemiUCB, act_SemiUCB = offlineEvaluate(mab3, user, arms, rewards, ATTR_reward, CFG_reward, contexts, n_rounds,)	
				note_reward_SemiNoBias, cum_regret_SemiNoBias, _ = offlineEvaluate(mab4, user, arms, rewards, ATTR_reward, CFG_reward, contexts, n_rounds)	
				note_reward_SemiNoHier, cum_regret_SemiNoHier, _ = offlineEvaluate(mab5, user, arms, rewards, ATTR_reward, CFG_reward, contexts, n_rounds)	

				ALL_reward_ConUCB.append(note_reward_ConUCB)
				CUM_regret_ConUCB.append(cum_regret_ConUCB)
				ALL_reward_LinUCB.append(note_reward_LinUCB)
				CUM_regret_LinUCB.append(cum_regret_LinUCB)
				ALL_reward_SemiUCB.append(note_reward_SemiUCB)
				CUM_regret_SemiUCB.append(cum_regret_SemiUCB)
				ALL_reward_SemiNoBias.append(note_reward_SemiNoBias)
				CUM_regret_SemiNoBias.append(cum_regret_SemiNoBias)
				ALL_reward_SemiNoHier.append(note_reward_SemiNoHier)
				CUM_regret_SemiNoHier.append(cum_regret_SemiNoHier)				
				Action_ConUCB.append(act_ConUCB)
				Action_LinUCB.append(act_LinUCB)
				Action_SemiUCB.append(act_SemiUCB)
				# t+=1
				# if t>20*(rep_t+1):
				# 	break
	# print('Average reward', np.mean(results_ConUCB))
	# print(Action_ConUCB)
	ConUCB_Reward_Iter.append(np.mean(ALL_reward_ConUCB,axis=0))
	ConUCB_Regret_Iter.append(np.mean(CUM_regret_ConUCB,axis=0))
	LinUCB_Reward_Iter.append(np.mean(ALL_reward_LinUCB,axis=0))
	LinUCB_Regret_Iter.append(np.mean(CUM_regret_LinUCB,axis=0))
	SemiUCB_Reward_Iter.append(np.mean(ALL_reward_SemiUCB,axis=0))
	SemiUCB_Regret_Iter.append(np.mean(CUM_regret_SemiUCB,axis=0))
	SemiNoBias_Reward_Iter.append(np.mean(ALL_reward_SemiNoBias,axis=0))
	SemiNoBias_Regret_Iter.append(np.mean(CUM_regret_SemiNoBias,axis=0))
	SemiNoHier_Reward_Iter.append(np.mean(ALL_reward_SemiNoHier,axis=0))
	SemiNoHier_Regret_Iter.append(np.mean(CUM_regret_SemiNoHier,axis=0))
Mean_reward_ConUCB=np.mean(np.array(ConUCB_Reward_Iter),axis=0)
Mean_regret_ConUCB=np.mean(np.array(ConUCB_Regret_Iter),axis=0)
Mean_reward_LinUCB=np.mean(np.array(LinUCB_Reward_Iter),axis=0)
Mean_regret_LinUCB=np.mean(np.array(LinUCB_Regret_Iter),axis=0)
Mean_reward_SemiUCB=np.mean(np.array(SemiUCB_Reward_Iter),axis=0)
Mean_regret_SemiUCB=np.mean(np.array(SemiUCB_Regret_Iter),axis=0)
Mean_reward_SemiNoBias=np.mean(np.array(SemiNoBias_Reward_Iter),axis=0)
Mean_regret_SemiNoBias=np.mean(np.array(SemiNoBias_Regret_Iter),axis=0)
Mean_reward_SemiNoHier=np.mean(np.array(SemiNoHier_Reward_Iter),axis=0)
Mean_regret_SemiNoHier=np.mean(np.array(SemiNoHier_Regret_Iter),axis=0)

plt.figure(figsize=(12,8))
plt.plot(Mean_reward_SemiUCB, label = 'SemiUCB')
# plt.plot(Mean_reward_SemiNoBias, label = 'SemiUCB (No Bias)')
# plt.plot(Mean_reward_SemiNoHier, label = 'SemiUCB (No Hierachy)')
plt.plot(Mean_reward_ConUCB, label = 'ConUCB')
plt.plot(Mean_reward_LinUCB, label = 'LinUCB')

plt.legend()
plt.xlabel("Rounds")
plt.ylabel("Averaged reward per round", fontsize='large')
plt.title("Averaged reward after single simulation")
plt.savefig('result/reward.png')

plt.figure(figsize=(12,8))
plt.plot(Mean_regret_SemiUCB, label = 'SemiUCB')
# plt.plot(Mean_regret_SemiNoBias, label = 'SemiUCB (No Bias)')
# plt.plot(Mean_regret_SemiNoHier, label = 'SemiUCB (No Hierachy)')
plt.plot(Mean_regret_ConUCB, label = 'ConUCB')
plt.plot(Mean_regret_LinUCB, label = 'LinUCB')
plt.legend()
plt.xlabel("Rounds")
plt.ylabel("Cumulative regret per round", fontsize='large')
plt.title("Cumulative regret after single simulation")
plt.savefig('result/regret.png')

with open('result/Reward_ConUCB.pkl','wb') as f:
	pickle.dump(ConUCB_Reward_Iter,f)
with open('result/Reward_LinUCB.pkl','wb') as f:
	pickle.dump(LinUCB_Reward_Iter,f)
with open('result/Reward_SemiUCB.pkl','wb') as f:
	pickle.dump(SemiUCB_Reward_Iter,f)
with open('result/Reward_SemiNoBias.pkl','wb') as f:
	pickle.dump(SemiNoBias_Reward_Iter,f)
with open('result/Reward_SemiNoHier.pkl','wb') as f:
	pickle.dump(SemiNoHier_Reward_Iter,f)

with open('result/Regret_ConUCB.pkl','wb') as f:
	pickle.dump(ConUCB_Regret_Iter,f)
with open('result/Regret_LinUCB.pkl','wb') as f:
	pickle.dump(LinUCB_Regret_Iter,f)
with open('result/Regret_SemiUCB.pkl','wb') as f:
	pickle.dump(SemiUCB_Regret_Iter,f)
with open('result/Regret_SemiNoBias.pkl','wb') as f:
	pickle.dump(SemiNoBias_Regret_Iter,f)
with open('result/Regret_SemiNoHier.pkl','wb') as f:
	pickle.dump(SemiNoHier_Regret_Iter,f)

with open('result/Action_ConUCB.pkl','wb') as f:
	pickle.dump(Action_ConUCB,f)
with open('result/Action_LinUCB.pkl','wb') as f:
	pickle.dump(Action_LinUCB,f)
with open('result/Action_SemiUCB.pkl','wb') as f:
	pickle.dump(Action_SemiUCB,f)