import numpy as np
from numpy.linalg import inv
from abc import ABC, abstractmethod

class MAB(ABC):
	
	@abstractmethod
	def play(self, tround, context):
		# Current round of t (for my implementations average mean reward array 
		# at round t is passed to this function instead of tround itself)
		self.tround = tround
		# Context: features of contextual bandits
		self.context = context
		# choose an arm which yields maximum value of average mean reward, tie breaking randomly
		chosen_arm = np.random.choice(np.where(self.tround==max(self.tround))[0])
		return chosen_arm
		pass
		
	
	@abstractmethod
	def update(self, arm, reward, context):
		# get the chosen arm
		self.arm = arm
		# get the context (may be None)
		self.context = context
		# update the overall step of the model
		self.step_n += 1
		# update the step of individual arms
		self.step_arm[self.arm] += 1
		# update average mean reward of each arm
		self.AM_reward[self.arm] = ((self.step_arm[self.arm] - 1) / float(self.step_arm[self.arm]) 
		* self.AM_reward[self.arm] + (1 / float(self.step_arm[self.arm])) * reward)
		return
		pass

class LinUCB(MAB):
	
	def __init__(self, nsups, narms, ndims, alpha, rho):
		# Set number of arms
		self.nsups = nsups
		self.rho = rho
		self.step_n=1
		self.Q0=np.ones(self.nsups)*np.inf
		self.step_arm=np.ones(self.nsups)
		self.sup_reward=np.ones(self.nsups)

		self.narms = narms
		# Number of context features
		self.ndims = ndims
		# explore-exploit parameter
		self.alpha = alpha
		# Instantiate A as a ndims×ndims matrix for each arm

		self.A = np.zeros((narms, narms, ndims, ndims))
		# Instantiate b as a 0 vector of length ndims.
		self.b = np.zeros((narms, narms, ndims, 1))
		# set each A per arm as identity matrix of size ndims
		for i in range(narms):
			for j in range(narms):
				self.A[i][j] = np.eye(ndims)
		
		super().__init__()
		return
		
	def play(self, context):
		# gains per each arm
		p_t = np.zeros((self.nsups,self.narms,self.narms))
		
		#===============================
		#    MAIN LOOP ...
		#===============================
		for c in range(self.nsups):
			for i in range(self.narms):
				for j in range(self.narms):
				# initialize theta hat
					self.theta = inv(self.A[i][j]).dot(self.b[i][j])
					cntx = context[i]+context[j]
					U = self.sup_reward[c]+np.sqrt(self.rho *(np.log(self.step_n)) / self.step_arm[c])
					p_t[c][i][j] = min(self.theta.T.dot(cntx) + self.alpha * np.sqrt(cntx.dot(inv(self.A[i][j]).dot(cntx))),1)+min(U,1)
		# print(p_t)
		action = np.unravel_index(np.argmax(p_t),p_t.shape)
		#DEBUG
		# print(action)
		return action[0],action[1], action[2]
		
	
	def update(self, cfg, arm1,arm2,vrew,arew,crew, context):
		cont=context[arm1]+context[arm2]
		self.A[arm1][arm2] = self.A[arm1][arm2] + np.outer(cont,cont)
		self.b[arm1][arm2] = np.add(self.b[arm1][arm2].T, cont*(vrew)).reshape(self.ndims,1)
		self.step_n += 1
		self.step_arm[cfg] += 1
		self.sup_reward[cfg] = ((self.step_arm[cfg] - 1) / float(self.step_arm[cfg]) * self.sup_reward[cfg] + (1 / float(self.step_arm[cfg])) * vrew)

class ConUCB(MAB):
	
	def __init__(self, nsups, narms, ndims, alpha, rho):
		# Set number of arms
		self.nsups = nsups
		self.rho = rho
		self.step_n=1
		self.Q0=np.ones(self.nsups)*np.inf
		self.step_arm=np.ones(self.nsups)
		self.sup_reward=np.ones(self.nsups)

		self.narms = narms
		# Number of context features
		self.ndims = ndims
		# explore-exploit parameter
		self.alpha = alpha
		# Instantiate A as a ndims×ndims matrix for each arm

		self.A = np.zeros((narms, narms, ndims, ndims))
		# Instantiate b as a 0 vector of length ndims.
		self.b = np.zeros((narms, narms, ndims, 1))
		# set each A per arm as identity matrix of size ndims
		for i in range(narms):
			for j in range(narms):
				self.A[i][j] = np.eye(ndims)
		
		super().__init__()
		return
		
	def play(self, context):
		# gains per each arm
		p_t = np.zeros((self.nsups,self.narms,self.narms))
		# print(self.nsups,self.narms,self.narms)
		#===============================
		#    MAIN LOOP ...
		#===============================
		for c in range(self.nsups):
			for i in range(self.narms):
				for j in range(self.narms):
				# initialize theta hat
					self.theta = inv(self.A[i][j]).dot(self.b[i][j])
					cntx = context[i]+context[j]
					U = self.sup_reward[c]+np.sqrt(self.rho *(np.log(self.step_n)) / self.step_arm[c])
					p_t[c][i][j] = min(self.theta.T.dot(cntx) + self.alpha * np.sqrt(cntx.dot(inv(self.A[i][j]).dot(cntx))),1)+min(U,1)
		# print(p_t)
		action = np.unravel_index(np.argmax(p_t),p_t.shape)
		#DEBUG
		# print(action)
		return action[0],action[1], action[2]
		
	
	def update(self,cfg,arm1,arm2,vrew,arew,crew,context):
		arew+=vrew
		crew+=vrew
		cont=context[arm1]+context[arm2]
		self.A[arm1][arm2] = self.A[arm1][arm2] + np.outer(cont,cont)
		self.b[arm1][arm2] = np.add(self.b[arm1][arm2].T, cont*arew).reshape(self.ndims,1)
		self.step_n += 1
		self.step_arm[cfg] += 1
		self.sup_reward[cfg] = ((self.step_arm[cfg] - 1) / float(self.step_arm[cfg]) * self.sup_reward[cfg] + (1 / float(self.step_arm[cfg])) * crew)

class SemiUCB(MAB):
	
	def __init__(self, nsups, narms, ndims, alpha, rho, bias_k=0.5, hier=True):
		# Set number of arms
		self.hier=hier
		self.nsups = nsups
		self.rho = rho
		self.step_n=1
		self.Q0=np.ones(nsups)*np.inf
		self.step_arm=np.ones(nsups)
		self.sup_reward=np.ones(nsups)

		# self.nbias=nsups*narms*ndims
		self.B0=np.ones((nsups,narms,narms))*np.inf
		self.step_bias=np.ones((nsups,narms,narms))
		self.bias_reward=np.ones((nsups,narms,narms))
		self.bias_k=bias_k

		self.narms = narms
		# Number of context features
		self.ndims = ndims
		# explore-exploit parameter
		self.alpha = alpha
		# Instantiate A as a ndims×ndims matrix for each arm

		self.A = np.zeros((narms, narms, ndims, ndims))
		# Instantiate b as a 0 vector of length ndims.
		self.b = np.zeros((narms, narms, ndims, 1))
		# set each A per arm as identity matrix of size ndims
		for i in range(narms):
			for j in range(narms):
				self.A[i][j] = np.eye(ndims)
		
		super().__init__()
		return
		
	def play(self, context):
		# gains per each arm
		pc_t=np.zeros(self.nsups)
		p_t = np.zeros((self.nsups,self.narms,self.narms))
		#===============================
		#    MAIN LOOP ...
		#===============================
		if self.hier:
			for c in range(self.nsups):
				U = self.sup_reward[c]+np.sqrt(self.rho *(np.log(self.step_n)) / self.step_arm[c])
				bk=(np.sqrt(self.rho *(np.log(self.step_n)) / self.step_arm[c]))<self.bias_k
				pc_t[c] = min(U,1)+np.amax(self.bias_reward[c])*bk
			action_c=np.unravel_index(np.argmax(pc_t),pc_t.shape)
			bk=(np.sqrt(self.rho *(np.log(self.step_n)) / self.step_arm[action_c]))<self.bias_k
			for i in range(self.narms):
				for j in range(self.narms):
				# initialize theta hat
					self.theta = inv(self.A[i][j]).dot(self.b[i][j])
					cntx = context[i]+context[j]
					BU = self.bias_reward[action_c][i][j]+np.sqrt(self.rho *(np.log(self.step_n)) / self.step_bias[action_c][i][j])
					p_t[action_c][i][j] = min(self.theta.T.dot(cntx) + self.alpha * np.sqrt(cntx.dot(inv(self.A[i][j]).dot(cntx))),1)+min(BU,1)
			action = np.unravel_index(np.argmax(p_t[action_c]),p_t[action_c].shape)
			return action_c,action[0], action[1]
		else:
			for c in range(self.nsups):
				for i in range(self.narms):
					for j in range(self.narms):
					# initialize theta hat
						self.theta = inv(self.A[i][j]).dot(self.b[i][j])
						cntx = context[i]+context[j]
						BU = self.bias_reward[c][i][j]+np.sqrt(self.rho *(np.log(self.step_n)) / self.step_bias[c][i][j])
						# bk=(np.sqrt(self.rho *(np.log(self.step_n)) / self.step_arm[c]))<self.bias_k
						p_t[c][i][j] = min(self.theta.T.dot(cntx) + self.alpha * np.sqrt(cntx.dot(inv(self.A[i][j]).dot(cntx))),1)+min(BU,1)*self.bias_k
			action = np.unravel_index(np.argmax(p_t),p_t.shape)
			return action[0],action[1],action[2]
		
		
	
	def update(self, cfg, arm1,arm2,vrew,arew,crew, context):
		# bias= not((not vrew) and arew and crew)
		bias=vrew
		# if not bias:
		# 	print(bias)
		arew+=bias
		crew+=bias
		cont=context[arm1]+context[arm2]
		self.A[arm1][arm2] = self.A[arm1][arm2] + np.outer(cont,cont)
		self.b[arm1][arm2] = np.add(self.b[arm1][arm2].T, cont*arew).reshape(self.ndims,1)
		self.step_n += 1
		self.step_arm[cfg] += 1
		self.sup_reward[cfg] = ((self.step_arm[cfg] - 1) / float(self.step_arm[cfg]) * self.sup_reward[cfg] + (1 / float(self.step_arm[cfg])) * crew)
		self.step_bias[cfg][arm1][arm2]+=1
		self.bias_reward[cfg][arm1][arm2]= ((self.step_bias[cfg][arm1][arm2] - 1) / float(self.step_bias[cfg][arm1][arm2]) * self.bias_reward[cfg][arm1][arm2] + (1 / float(self.step_bias[cfg][arm1][arm2])) * bias)