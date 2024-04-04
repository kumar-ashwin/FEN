import gurobipy as gp
from gurobipy import GRB
import numpy as np
import copy
from utils import get_distance

def get_assignment_old(Qvalues):
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

def get_assignment(Qvalues, resource_counts, agent_constraints=None):
	# Qvalues is a list of lists, where each list is the Q values for each agent for each resource
	# resource_counts is a list of the number of agents that can be assigned to each resource
	# agent_constraints is a list of the resources an agent cannot be assigned to
	n_agents = len(Qvalues)
	n_resources = len(Qvalues[0])
	#Create a model
	m = gp.Model("mip1")
	m.setParam('OutputFlag', 0)
	#Create variables
	x = m.addVars(n_agents, n_resources, vtype=GRB.BINARY, name="x")
	#Set objective and add constraints
	m.setObjective(sum(sum(Qvalues[i][j]*x[i,j] for j in range(n_resources)) for i in range(n_agents)), GRB.MAXIMIZE)
	m.addConstrs(sum(x[i,j] for j in range(n_resources))==1 for i in range(n_agents)) # Each agent can only be assigned to exactly one resource (action)
	m.addConstrs(sum(x[i,j] for i in range(n_agents))<=resource_counts[j] for j in range(n_resources)) # Each resource can only be assigned to a certain number of agents
	if agent_constraints is not None:
		for i in range(n_agents):
			for j in agent_constraints[i]:
				m.addConstr(x[i,j]==0)
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
		# Qvals = [[np.random.rand()*min(2-ind, 1)*1.5 for ind in range(n_resources+1)] for _ in range(n_agents)] #Increase importance of doing nothing
		Qvals = [[np.random.rand()*max(2-ind,1) for ind in range(n_resources+1)] for _ in range(n_agents)] #Increase importance of doing nothing
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

def compute_best_joint_action(model, envt, obs, targets, n_agents, n_resources, su, epsilon=0.0, beta=0.0, direction='both'):
	#Enumerate and score all joint actions, then select the best one
	#For fairness, the modification will be made to each action as we can compute the change in variance because of each.

	occupied_resources = set([targets[j][0] for j in range(n_agents) if targets[j] is not None])
	# print('******************')
	# print("Occupied resources", occupied_resources)
	#Possible actions
	legal_actions = []
	#Enumerate all possible actions: Which resource goes to which agent
	#11P3 = 990 possible actions, max (1 null action)
	for i in range(n_agents+1):
		m1 = -1
		if i<n_agents:
			if targets[i] is not None: #If the agent already has a target, it cant be assigned another one, this action is invalid
				continue
			if 0 in occupied_resources: #If the resource is already occupied, doesn't make a difference. Only consider the null action
				continue
			m1 = i
		for j in range(n_agents+1):
			if i==j and i!=n_agents:
				continue
			m2 = -1
			if j<n_agents:
				if targets[j] is not None:
					continue
				if 1 in occupied_resources:
					continue
				m2 = j
			for k in range(n_agents+1):
				if i==k and i!=n_agents:
					continue
				if j==k and j!=n_agents:
					continue
				m3 = -1
				if k<n_agents:
					if targets[k] is not None:
						continue
					if 2 in occupied_resources:
						continue
					m3 = k
				
				legal_actions.append((m1,m2,m3))
		# #Do the above, but smarter, using permutations
		# from itertools import permutations
	# print("num valid actions", len(legal_actions))
	# print('******************')

	#Get a random action with probability epsilon
	if np.random.rand()<epsilon:
		sel_action_ind = np.random.choice(len(legal_actions))
		sel_action = legal_actions[sel_action_ind]
	else:
		Qvals = [-1000000 for _ in range(len(legal_actions))]
		for act_id, action in enumerate(legal_actions):
			#Get the Q values for this action
			agent_actions = [-1 for _ in range(n_agents)]
			for res, agent in enumerate(action):
				if agent!=-1:
					agent_actions[agent] = res
			pd_state = envt.get_post_decision_central_state(obs, agent_actions)
			Qvals[act_id] = float(model.get(np.array([pd_state])))
		
		#Fairness post processing
		#TODO
		# print(legal_actions[-10:])
		# print(Qvals[-10:])
		# exit()
		#Select the best action
		# pprint.pprint(list(zip(legal_actions, Qvals)))
		# print(Qvals)
		sel_action = legal_actions[np.argmax(Qvals)]
		# print(sel_action)
		# exit()
							
	#Map resource-agent assignment to per agent action
	actions = [-1 for _ in range(n_agents)]
	for res, agent in enumerate(sel_action):
		if agent!=-1:
			actions[agent] = res
	return actions