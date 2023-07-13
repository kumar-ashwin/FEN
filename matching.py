import gurobipy as gp
from gurobipy import GRB
import numpy as np
import copy
from matthew_envt import get_distance

def get_assignment(Qvalues):
	n_agents = len(Qvalues)
	n_resources = len(Qvalues[0])
	#Create a model
	m = gp.Model("mip1")
	m.setParam('OutputFlag', 0)
	#Create variables
	x = m.addVars(n_agents, n_resources, vtype=GRB.BINARY, name="x")
	#Set objective
	m.setObjective(sum(sum(Qvalues[i][j]*x[i,j] for j in range(n_resources)) for i in range(n_agents)), GRB.MAXIMIZE)
	#Add constraints
	m.addConstrs(sum(x[i,j] for j in range(n_resources))==1 for i in range(n_agents)) # Each agent can only be assigned to exactly one resource
	m.addConstrs(sum(x[i,j] for i in range(n_agents))<=1 for j in range(1,n_resources)) # Each resource except the first one can only be assigned to one agent
	#Solve
	m.optimize()
	#Get solution
	assignment = []
	for i in range(n_agents):
		for j in range(n_resources):
			if x[i,j].x==1:
				assignment.append(j)
	return assignment

def compute_best_actions(model, obs, targets, n_agents, n_resources, su, epsilon=0.0, beta=0.0, direction='both'):

	Qvals = [[-1000000 for _ in range(n_resources+1)] for _ in range(n_agents)]
	occupied_resources = set([targets[j][0] for j in range(n_agents) if targets[j] is not None])

	#Get a random action with probability epsilon
	if np.random.rand()<epsilon:
		Qvals = [[np.random.rand()*min(2-ind, 1)*1.5 for ind in range(n_resources+1)] for _ in range(n_agents)] #Increase importance of doing nothing
		# print("random action")
		#occupied resources cant be taken, so set their Q values to -1000000
		Qvals = [[-1000000 if j-1 in occupied_resources else Qvals[i][j] for j in range(n_resources+1)] for i in range(n_agents)]
		# Qvals = [[np.random.rand() for _ in range(n_resources+1)] for _ in range(n_agents)]
	else:
		#First action is to do nothing. This is the default action.
		#Action indexing starts at -1. Shift by 1 to get the correct index
		resource = copy.deepcopy(obs[1])
		for i in range(n_agents):
			h = copy.deepcopy(obs[0][i])
			ant_loc = [h[0],h[1]]
			ant_speed = h[2]

			Qvals[i][0] = float(model.get(np.array([h])))

			if targets[i] is None:
				#If the agent can pick another action, get Q values for all actions
				for j in range(n_resources):
					if j not in occupied_resources:
						h = copy.deepcopy(obs[0][i])
						h[-3] = resource[j][0]
						h[-2] = resource[j][1]
						h[-1] = get_distance(ant_loc,resource[j])/ant_speed
						Qvals[i][j+1] = float(model.get(np.array([h])))
					
			#Fairness post processing
			if beta is not 0.0:
				# if direction=='both':
				mult = (su[i] - np.mean(su))/1000
				if direction=='adv':
					mult = min(0,(su[i] - np.mean(su)))/1000
				elif direction=='dis':
					mult = max(0,(su[i] - np.mean(su)))/1000
				# mult = (su[i] - np.mean(su))/1000
				for j in range(len(Qvals[i])):
					if j==0:
						Qvals[i][j] = Qvals[i][j] + beta * mult
					else:
						Qvals[i][j] = Qvals[i][j] - beta * mult
	
	# for i in range(n_agents):
	# 	print(Qvals[i])
	#For each agent, select the action using the central agent given Q values
	actions = get_assignment(Qvals)
	#shift actions by 1 to get the correct index
	actions = [a-1 for a in actions]
	# print(actions)
	return actions

def SI_reward(utils, direction='both'):
	#normalize
	avg = np.mean(utils)
	utils = [u/avg for u in utils]
	#Simple incentive/penalty for fairness
	if direction=='both':
		return [np.mean(utils) - utils[i] for i in range(len(utils))]
	elif direction=='adv':
		return [min(0, np.mean(utils) - utils[i]) for i in range(len(utils))]
	elif direction=='dis':
		return [max(0, np.mean(utils) - utils[i]) for i in range(len(utils))]
	else:
		print('Invalid direction')
		return None