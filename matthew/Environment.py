import copy
import numpy as np
import matplotlib.pyplot as plt
from matching import get_assignment

np.random.seed(0)

def get_distance(a,b):
	#Get distance between two points
	return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

class MACEnvt:
	'''
	Meta class for multi-agent central decision-maker environments

	# self.su should be used to keep track of rewards.
	# self.discounted_su should be used for fairness
	'''
	def __init__(self):
		pass
	def reset(self):
		pass
	def step(self, actions):
		# actions[i] is the action taken by agent i
		pass
	def get_state(self):
		pass
	def set_state(self, state):
		pass
	def get_obs(self):
		pass
	def render(self):
		pass
	def get_post_decision_states(self, obs, actions):
		pass
	def compute_best_actions(self, model, envt, obs, epsilon=0.0, beta=0.0, direction='both', use_greedy=False):
		# targets, n_agents, n_resources, su, should be extracted from envt
		pass

class SimpleEnvt:
	def __init__(self, n_agents, warm_start=0, past_discount=0.995, **kwargs):
		"""
		Simple environment for fairness vs utility
		2 agents, 1 resource. 
		Agent 1 gets reward of 1 if it picks up the resource
		Agent 2 gets reward of 0.5 if it picks up the resource
		System util is the sum of the rewards
		"""
		self.n_agents = n_agents
		self.warm_start = warm_start
		self.past_discount = past_discount
		self.agent_scores = [1,0.5] # The reward each agent gets for picking up the resource
		self.observation_space = ['agent_score', 'su', 'relative_su']
		self.reset()

	def reset(self):
		self.su = np.zeros(self.n_agents)
		w = 5 #width of the warm start randomization
		self.discounted_su = np.array([
			self.warm_start + np.random.rand()*w - w/2 
			for _ in range(self.n_agents)])
		
	def step(self, actions):
		re = [0]*self.n_agents
		for i in range(self.n_agents):
			re[i] = actions[i]*self.agent_scores[i]

		self.su+=np.array(re)
		self.discounted_su = self.discounted_su*self.past_discount + np.array(re)
		return re

	def get_state(self):
		return self.su, self.discounted_su
	
	def set_state(self, state):
		self.su, self.discounted_su = state

	def get_obs(self):
		agents = []
		for i in range(self.n_agents):
			agent = [
				self.agent_scores[i],
				0, # Dummy, placeholder for whether agent gets a resource. Just for VF
				# self.su[i],
				self.discounted_su[i]/np.mean(self.discounted_su) - 1,
			]
			agents.append(agent)
		return [agents, []]
	
	def render(self):
		print("agent 1 \t agent 2")
		print(self.su[0], "\t", self.su[1])
	
	def get_post_decision_state_agent(self, obs, action, idx):
		s_i = copy.deepcopy(obs)
		if len(s_i)>1:
			s_i[1] += action*self.agent_scores[idx]
		return s_i

	def get_post_decision_states(self, obs, actions):
		states = []
		for i in range(self.n_agents):
			s_i = self.get_post_decision_state_agent(obs[0][i], actions[i], i)
			states.append(s_i)
		return states
	
	def compute_best_actions(self, model, envt, obs, epsilon=0.0, beta=0.0, direction='both', use_greedy=False, val=False):
		# Greedy strategy: Fastest agent gets the resource
		n_agents = envt.n_agents
		Qvals = [[-1000000 for _ in range(2)] for _ in range(n_agents)]
		#Get a random action with probability epsilon
		if np.random.rand()<epsilon:
			Qvals = [[np.random.rand() for act in range(2)] for _ in range(n_agents)]
		else:
			for i in range(n_agents):
				for j in range(2):
					s_i = self.get_post_decision_state_agent(obs[0][i], j, i)
					Qvals[i][j] = float(model.get(np.array([s_i])))
		if val:
			print(Qvals)
		actions = get_assignment(Qvals, [n_agents, 1])
		return actions
		# One way to account for agents being able to receive more than one resource is to flip the allocation. 
		# Each resource must receive exactly one agent

class MatthewEnvt:
	def __init__(self, 
	      n_agents, 
		  n_resources, 
		  max_size, 
		  min_size=0.01, 
		  size_update=0.005, 
		  base_speed=0.01, 
		  reallocate=False, 
		  simple_obs=False, 
		  warm_start=0,
		  past_discount=0.995,
		  GAMMA=None
		  ):
		self.n_agents = n_agents
		self.n_resources = n_resources
		self.max_size = max_size
		self.min_size = min_size
		self.size_update = size_update
		self.base_speed = base_speed
		self.warm_start = warm_start
		self.past_discount = past_discount
		self.GAMMA = GAMMA

		self.reset()
		
		self.reallocate = reallocate
		self.simple_obs = simple_obs
	
		self.observation_space = ['loc','size','speed', 'eaten','n_other_free_agents','relative_size', 'relative_su', 'other_agents', 'target_resource']
		if self.simple_obs:
			self.observation_space = ['loc','size','speed','n_other_free_agents','relative_size', 'target_resource']
	
	def reset(self):
		agent_types = [0,0,0,0,0,0,2,2,2,2]
		self.agent_types = agent_types
		ant = []
		size = []
		speed = []
		su = [0]*self.n_agents
		targets = []
		for i in range(self.n_agents):
			ant.append(np.random.rand(2))
			size.append(self.min_size + self.agent_types[i]/50)
			speed.append(self.base_speed + size[i])
			targets.append(None)
		su = np.array(su)

		resource=[]
		for i in range(self.n_resources):
			resource.append(np.random.rand(2))

		
		self.resource = resource
		self.ant = ant
		self.targets = targets
		self.size = size
		self.speed = speed
		self.set_speed_from_sizes()
		self.su = su
		
		w = 5 #width of the warm start randomization
		self.discounted_su = np.array([
			self.warm_start + np.random.rand()*w - w/2 
			for _ in range(self.n_agents)])
		
	
	def set_speed_from_sizes(self):
		for i in range(self.n_agents):
			self.speed[i] = self.base_speed + self.size[i]

	def step(self, actions):
		clear_targets = False
		# Actions just decide mapping of agents to resources
		re = [0]*self.n_agents  #rewards. If an agent picks up a resource, get reward of 1
		re_expected = [0]*self.n_agents  #rewards. If an agent is assigned a resource, get reward of 1*discount^T
		for i in range(self.n_agents):
			if actions[i]!=-1:
				self.targets[i] = [actions[i]]
				#Add the time to reach the resource to the targets vector
				time_to_reach = get_distance(self.ant[i],self.resource[actions[i]])/self.speed[i]
				self.targets[i].append(time_to_reach)
				assert self.GAMMA is not None, "GAMMA must be set for re_expected"
				re_expected[i]=1*self.GAMMA**time_to_reach 

			#Move each agent towards its target resource
			if self.targets[i] is not None:
				#Other agents can't pick up the resources if they are claimed
				#if target is overlapped by the agent, remove it
				if self.targets[i][1]<=1:
					self.ant[i][0] = self.resource[self.targets[i][0]][0]
					self.ant[i][1] = self.resource[self.targets[i][0]][1]
					re[i]=1 #Get reward

					#Reset target resource
					self.resource[self.targets[i][0]]=np.random.rand(2)
					self.size[i]=min(self.size[i]+self.size_update, self.max_size)
					# self.speed[i]=self.base_speed+self.size[i]
					self.targets[i]=None
					clear_targets = True
				else:
					#Move agent towards target resource. Each step, move 1/time_remaining of the way
					self.ant[i][0]+=(self.resource[self.targets[i][0]][0]-self.ant[i][0])/self.targets[i][1]
					self.ant[i][1]+=(self.resource[self.targets[i][0]][1]-self.ant[i][1])/self.targets[i][1]
					self.targets[i][1]-=1
					if get_distance(self.ant[i],self.resource[self.targets[i][0]])<self.size[i]:
						re[i]=1
						#Reset target resource
						self.resource[self.targets[i][0]]=np.random.rand(2)
						self.size[i]=min(self.size[i]+self.size_update, self.max_size)
						# self.speed[i]=self.base_speed+self.size[i]
						self.targets[i]=None
						clear_targets = True
			else:
				#Move randomly
				p_move = 0.8
				dr = np.random.rand()*2*np.pi
				if np.random.rand()<p_move:
					self.ant[i][0]+=np.cos(dr)*self.speed[i]
					self.ant[i][1]+=np.sin(dr)*self.speed[i]
			
			#Check for bounds
			if self.ant[i][0]<0:
				self.ant[i][0]=0
			if self.ant[i][0]>1:
				self.ant[i][0]=1
			if self.ant[i][1]<0:
				self.ant[i][1]=0
			if self.ant[i][1]>1:
				self.ant[i][1]=1
		
		#Update speeds
		self.set_speed_from_sizes()

		# If any resources were picked up, reset the targets
		if self.reallocate:
			if clear_targets:
				# print("Clearing Targets")
				for i in range(self.n_agents):
					self.targets[i]=None

		self.su+=np.array(re) # This always keeps track of what the agent has received exactlys
		
		#Update the discounted su
		re_return = np.array(re_expected)
		# re_return = np.array(re)
		self.discounted_su = self.discounted_su*self.past_discount + np.array(re_return)

		return re_return

	def get_state(self):
		return self.ant, self.resource, self.targets, self.size, self.speed, self.su, self.agent_types, self.discounted_su
	def set_state(self, state):
		self.ant, self.resource, self.targets, self.size, self.speed, self.su, self.agent_types, self.discounted_su = state
	
	def get_obs(self):
		#Gets the state of the environment (Vector of each agent states)
		#agent positions, resource positions, size vector, speed vector, number of agents
		state=[]
		agents = []
		for i in range(self.n_agents):
			h={}
			h['loc'] = [self.ant[i][0], self.ant[i][1]]
			h['size'] = self.size[i]
			h['speed'] = self.speed[i]
			h['eaten'] = self.su[i]
			
			#Get number of agents without a target resource
			n = sum([1 for j in range(self.n_agents) if self.targets[j] is None])
			h['n_other_free_agents'] = n
			h['relative_size'] = self.size[i]/np.mean(self.size) - 1
			h['relative_su'] = self.discounted_su[i]/np.mean(self.discounted_su) - 1

			#Get info about other agents
			others = []
			# append locations of all other agents, their sizes, and their distances to the target resource
			for j in range(self.n_agents):
				if j!=i:
					others.append([
						self.ant[j][0],self.ant[j][1],
						self.speed[j], 
						self.su[j],
						get_distance(self.ant[j], self.resource[self.targets[i][0]]) if self.targets[i] is not None else -1,
						1 if self.targets[j] is None else 0
						])
			#flatten
			others = [feature for other in others for feature in other]
			h['other_agents'] = others

			#Get info about target resource
			if self.targets[i] is not None:
				t = [self.resource[self.targets[i][0]][0], self.resource[self.targets[i][0]][1], self.targets[i][1]]
			else:
				t = [-1,-1,100]
			h['target_resource'] = t

			#feats
			feats = []
			flist = self.observation_space
			
			for f in flist:
				if f=='loc' or f=='other_agents' or f=='target_resource':
					feats.extend(h[f])
				else:
					feats.append(h[f])

			agents.append(feats)

		state.append(agents)
		state.append(copy.deepcopy(self.resource))

		return state

	def get_central_obs(self):
		#Gets the state of the environment
		#agent positions, resource positions, size vector, speed vector, number of agents
		global_state = []
		agent_states = []
		for i in range(self.n_agents):
			h={}
			h['loc'] = [self.ant[i][0], self.ant[i][1]]
			h['size'] = self.size[i]
			h['speed'] = self.speed[i]
			h['eaten'] = self.su[i]
			
			h['relative_size'] = self.size[i]/np.mean(self.size) - 1
			h['relative_su'] = self.discounted_su[i]/np.mean(self.discounted_su) - 1

			#Get info about target resource
			if self.targets[i] is not None:
				t = [self.resource[self.targets[i][0]][0], self.resource[self.targets[i][0]][1], self.targets[i][1]]
			else:
				t = [-1,-1,100]
			h['target_resource'] = t

			#feats
			feats = []
			flist = ['loc','size','speed', 'eaten','relative_size', 'relative_su', 'target_resource']
			if self.simple_obs:
				flist = ['loc','size','speed','n_other_free_agents','relative_size', 'target_resource']
			
			for f in flist:
				if f=='loc' or f=='other_agents' or f=='target_resource':
					feats.extend(h[f])
				else:
					feats.append(h[f])

			global_state.extend(feats)
			agent_states.append(feats)
		
		for i in range(self.n_resources):
			global_state.extend(self.resource[i])

		return [global_state, agent_states, copy.deepcopy(self.resource)]

	def get_post_decision_states(self, obs, actions):
		states = []
		for i in range(self.n_agents):
			ant_loc = self.ant[i]
			s_i = copy.deepcopy(obs[0][i])
			if actions[i]!=-1:
				j = actions[i]
				#apply action
				s_i[-3] = self.resource[j][0]
				s_i[-2] = self.resource[j][1]
				s_i[-1] = get_distance(ant_loc,self.resource[j])/self.speed[i]
			states.append(s_i)
		return states

	def get_post_decision_states(self, obs, actions):
		states = []

		for i in range(self.n_agents):
			ant_loc = self.ant[i]
			s_i = copy.deepcopy(obs[0][i])
			if actions[i]!=-1:
				j = actions[i]
				#apply action
				s_i[-3] = self.resource[j][0]
				s_i[-2] = self.resource[j][1]
				s_i[-1] = get_distance(ant_loc,self.resource[j])/self.speed[i]
			states.append(s_i)
		return states
	
	def get_post_decision_central_state(self, obs, actions):
		agents = copy.deepcopy(obs[1])
		state = []
		for i,agent in enumerate(agents):
			if actions[i]!=-1:
				j = actions[i]
				#apply action
				agent[-3] = self.resource[j][0]
				agent[-2] = self.resource[j][1]
				agent[-1] = get_distance(agent[0:2],self.resource[j])/self.speed[i]
			state.extend(agent)
		for i in range(self.n_resources):
			state.extend(self.resource[i])
		return state

	def render(self):
		for i in range(self.n_agents):
			theta = np.arange(0, 2*np.pi, 0.01)
			x = self.ant[i][0] + self.size[i] * np.cos(theta)
			y = self.ant[i][1] + self.size[i] * np.sin(theta)
			plt.plot(x, y)
			if self.targets[i] is not None:
				#plot a line from ant to target
				plt.plot([self.ant[i][0],self.resource[self.targets[i][0]][0]],[self.ant[i][1],self.resource[self.targets[i][0]][1]], color = 'red')
		for i in range(self.n_resources):
			plt.scatter(self.resource[i][0], self.resource[i][1], color = 'green')
		plt.axis("off")
		plt.axis("equal")
		plt.xlim(0 , 1)
		plt.ylim(0 , 1)
		plt.ion()
		plt.pause(0.1)
		plt.close()

	def compute_best_actions(self, model, envt, obs, epsilon=0.0, beta=0.0, direction='both', use_greedy=False):
		# Greedy strategy: Fastest agent gets the resource
		targets, n_agents, n_resources, su = envt.targets, envt.n_agents, envt.n_resources, envt.su

		Qvals = [[-1000000 for _ in range(n_resources+1)] for _ in range(n_agents)]
		occupied_resources = set([targets[j][0] for j in range(n_agents) if targets[j] is not None])

		#Get a random action with probability epsilon
		if np.random.rand()<epsilon:
			Qvals = [[np.random.rand()*max(2-ind,1) for ind in range(n_resources+1)] for _ in range(n_agents)] #Increase importance of doing nothing
			#occupied resources cant be taken, so set their Q values to -1000000
			Qvals = [[-1000000 if j-1 in occupied_resources else Qvals[i][j] for j in range(n_resources+1)] for i in range(n_agents)]
		else:
			#First action is to do nothing. This is the default action.
			#Action indexing starts at -1. Shift by 1 to get the correct index
			resource = copy.deepcopy(obs[1])
			for i in range(n_agents):
				h = copy.deepcopy(obs[0][i])
				ant_loc = [h[0],h[1]]
				ant_speed = h[2]

				Qvals[i][0] = float(model.get(np.array([h])))
				if use_greedy:
					Qvals[i][0] = 0

				if targets[i] is None:
					#If the agent can pick another action, get Q values for all actions
					for j in range(n_resources):
						if j not in occupied_resources:
							if use_greedy:
								Qvals[i][j+1] = ant_speed
							else:
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
		#For each agent, select the action using the central agent given Q values and target availability
		resource_counts = [n_agents] # First action does not have a resource restriction
		for i in range(n_resources):
			# Add available resources
			if i not in occupied_resources:
				resource_counts.append(1)
			else:
				resource_counts.append(0)
			
		actions = get_assignment(Qvals, resource_counts)
		#shift actions by 1 to get the correct index
		actions = [a-1 for a in actions]
		# print(actions)
		return actions
		pass

class JobSchedulingEnvt:
	'''
	Many workers that desire to work on a job. As long as worker occupies the job's location, they get a reward
	This is difficult to solve as a central agent. What does an allocation mean? How do other agents know how to get out of the way?

	Possible solution: have actions represent each grid location. Each agent can move to any location.
	Then, move agents to locations based on their past utility.
	'''
	def __init__(self, 
	      n_agents, 
		  gridsize=5,
		  reallocate=True, 
		  simple_obs=False, 
		  warm_start=0,
		  past_discount=0.995,
		  ):
		self.n_agents = n_agents
		self.gridsize = gridsize
		
		self.warm_start = warm_start
		self.past_discount = past_discount

		self.reset()

		self.reallocate = reallocate
		self.simple_obs = simple_obs
	
	def reset(self):
		# The grid is a 2D array of size gridsize x gridsize
		# Each agent is assigned a random location on the grid
		# Empty grid locations are 0
		# Agent locations are 1,2,3,4,5...
		# Job location is always fixed, so not tracked on the grid

		self.grid = np.zeros((self.gridsize, self.gridsize))
		ant = []
		#Random start locations
		for i in range(self.n_agents):
			loc = np.random.randint(0,self.gridsize,2)
			while self.grid[loc[0],loc[1]]!=0:
				loc = np.random.randint(0,self.gridsize,2)
			ant.append(np.random.randint(0,self.gridsize,2))
			self.grid[ant[i][0],ant[i][1]]=i+1
		ant = np.array(ant)

		self.job = np.random.randint(0,self.gridsize,2)
		self.job_reward = 1

		self.ant = ant
		self.targets = [None]*self.n_agents
		self.su = np.zeros(self.n_agents)

		w = 5 #width of the warm start distribution
		self.discounted_su = np.array([
			self.warm_start + np.random.rand()*w - w/2
			for _ in range(self.n_agents)])

	def pre_move(self, ant, dir):
		x, y = ant
		dir_map = {
			0: [0,0],
			1: [0,1],
			2: [0,-1],
			3: [-1,0],
			4: [1,0]
		}
		delta_x = dir_map[dir][0]
		delta_y = dir_map[dir][1]
		new_x = x + delta_x
		new_y = y + delta_y
		return new_x, new_y
		
	
	def move(self, ant, dir):
		#Move agent i in direction
		#dir is one of [stay, up, down, left, right]
		illegal=False
		x,y = ant
		new_x, new_y = self.pre_move(ant,dir)

		# Simple bounds checking. Later on, could use MAPF to make it smarter and allow simultaneous moves
		if 0<=new_x<self.gridsize and 0<=new_y<self.gridsize:
			if self.grid[new_x, new_y]==0:
				x, y = new_x, new_y
		else:
			illegal=True
		return x,y, illegal
		
	# Each action is a mapping of agent i to a grid location
	# OR
	# Each action is one of [stay, up, down, left, right]
	def step(self, actions):
		# actions[i] is the action taken by agent i
		# Doing 5 available actions first. Then will try to generalize to grid locations
		reset_job = False
		re = [0]*self.n_agents  #rewards. If an agent is on the job, get reward of 1
		penl = [0]*self.n_agents  #penalties. If an agent makes an illegal move, get penalty
		for i in range(self.n_agents):
			if actions[i]!=-1:
				new_x, new_y, illegal = self.move(self.ant[i], actions[i])			
				self.grid[self.ant[i][0],self.ant[i][1]]=0
				self.ant[i][0] = new_x
				self.ant[i][1] = new_y
				self.grid[new_x, new_y]=i+1
	

			#Check if agent is on the job
			if self.ant[i][0]==self.job[0] and self.ant[i][1]==self.job[1]:
				re[i]=self.job_reward
				# reset_job = True
			if illegal:
				penl[i] += -10000 #Large penalty for illegal move
				
		# if reset_job:
		# 	self.job = np.random.randint(0,self.gridsize,2)	
		self.su+=np.array(re)
		#Update the discounted utilities
		self.discounted_su = self.discounted_su*self.past_discount + np.array(re)

		# Add penalties for vf reward
		for i in range(self.n_agents):
			re[i] += penl[i]

		return re

	def get_state(self):
		return self.ant, self.job, self.su, self.discounted_su
		
	def set_state(self, state):
		self.ant, self.job, self.su, self.discounted_su = state

	def get_obs(self):
		#Gets the state of the environment (Vector of each agent states)
		state = []
		agents = []
		for i in range(self.n_agents):
			h={}
			h['loc'] = [self.ant[i][0], self.ant[i][1]]
			h['util'] = self.su[i]
			h['relative_util'] = self.discounted_su[i]/np.mean(self.discounted_su) - 1

			#Get info about other agents
			others = []
			# append locations of all other agents
			for j in range(self.n_agents):
				if j!=i:
					others.append([
						self.ant[j][0],self.ant[j][1],
						self.su[j],
					])
			#flatten
			others = [feature for other in others for feature in other]
			h['other_agents'] = others

			# alt: get the 3x3 grid around the agent
			h['grid'] = []
			for x in range(self.ant[i][0]-1, self.ant[i][0]+2):
				for y in range(self.ant[i][1]-1, self.ant[i][1]+2):
					if 0<=x<self.gridsize and 0<=y<self.gridsize:
						h['grid'].append(self.grid[x,y])
					else:
						h['grid'].append(-1)
			
			h['job'] = copy.deepcopy(self.job)
			#relative job location
			h['relative_job'] = [self.ant[i][0]-self.job[0], self.ant[i][1]-self.job[1]]
			# Relative job location does not work, would need to update it in the post_decision state as well

			#feats
			feats = []
			flist = ['loc','util','relative_util', 'other_agents', 'grid', 'job']
			flist = ['loc', 'relative_util', 'grid', 'job']
			flist = ['loc', 'job']
			flist = ['relative_job', 'grid']
			flist = ['relative_job', 'grid', 'relative_util'] # This seems to work with lr=0.0003, hidden_size=20
			# flist = ['relative_job', 'relative_job', 'grid', 'relative_util'] # second relative job because first is overwritten by post-decision loc: Probably not needed?
			# ^Nope. Worse. Even with 40 features
			if self.simple_obs:
				flist = ['loc','util','relative_util', 'grid', 'job']
			
			for f in flist:
				if f=='util' or f=='relative_util':
					feats.append(h[f])
				else:
					feats.extend(h[f])

			agents.append(feats)

		state.append(agents)
		state.append(copy.deepcopy(self.job))

		return state

	def render(self, VF=None):
		for i in range(self.n_agents):
			plt.scatter(self.ant[i][0], self.ant[i][1], color = 'blue')
			# Add text to show scores
			plt.text(self.ant[i][0], self.ant[i][1], str(self.su[i]))
		plt.scatter(self.job[0], self.job[1], color = 'red', marker='x')
		plt.axis("off")
		plt.axis("equal")
		# Make gridlines
		for i in range(self.gridsize+1):
			# Horizontal lines
			l = i - 0.5
			plt.plot([-0.5,self.gridsize-0.5],[l,l], color='black')
			# Vertical lines
			plt.plot([l,l],[-0.5,self.gridsize-0.5], color='black')
		
		if VF is not None:
			# value_grid = self.get_value_grid(VF)
			# for i in range(self.gridsize):
			# 	for j in range(self.gridsize):
			# 		plt.text(i,j, str(round(value_grid[i,j],2)))
			
			post_dec_value_grid = self.get_post_dec_grid(VF)
			for i in range(self.gridsize*2+1):
				for j in range(self.gridsize*2+1):
					if (i+j)%2!=0:
						plt.text(i/2-0.5,j/2-0.5, str(round(post_dec_value_grid[i,j],2)), fontsize=8)
			
			
		plt.xlim(-1 , self.gridsize)
		plt.ylim(-1 , self.gridsize)
		plt.ion()
		plt.pause(0.01)
		plt.close()
	

	def get_post_decision_state_agent(self, state, action, ind):
		# For a single agent
		s_i = copy.deepcopy(state)
		if action!=-1:
			x, y = s_i[:2]
			new_x, new_y = self.pre_move(s_i[:2], action)
			#apply action
			s_i[0] = (new_x + x)/2
			s_i[1] = (new_y + y)/2  #Show intent to move through a half step
		return s_i

	# relative cal: ant_loc - job_loc
	# ant     job    relative
	# 0,0     1,4	-1,-4
	#After move 1,1
	# 1,1     1,4	0,-3
	# So, the move can directly be applied to the relative location
	
	def get_post_decision_states(self, obs, actions):
		states = []
		for i in range(self.n_agents):
			s_i = self.get_post_decision_state_agent(obs[0][i], actions[i], i)
			states.append(s_i)
		return states

	def compute_best_actions(self, model, envt, obs, epsilon=0.0, beta=0.0, direction='both'):
		# envt is an environment class object with the state set to desired state
		# targets, n_agents, n_actions, su
		n_agents = envt.n_agents
		n_actions = 5
		su = envt.su

		Qvals = [[-1000000 for _ in range(n_actions)] for _ in range(n_agents)]

		#Get a random action with probability epsilon
		if np.random.rand()<epsilon:
			# Qvals = [[np.random.rand()*max(2-ind,1) for ind in range(n_actions)] for _ in range(n_agents)] #Increase importance of doing nothing
			actions = [np.random.randint(0,n_actions) for _ in range(n_agents)]
			return actions
		else:
			# First action is to do nothing.
			resource = copy.deepcopy(obs[1])
			for i in range(n_agents):
				h = copy.deepcopy(obs[0][i])
				for act in range(n_actions):
					h_post = self.get_post_decision_state_agent(h, act, i)
					# if act!=0:
					# 	if h_post[0]==h[0] and h_post[1]==h[1]:
					# 		#If the agent didn't move, set the Q value to -1000000
					# 		Qvals[i][act] = -1000000
						
					Qvals[i][act] = float(model.get(np.array([h_post])))

					# Custom calculation, naive greedy vf
					# pos_ant = np.array(h_post[:2])
					# pos_resource = np.array(h_post[-2:])
					# Qvals[i][act] = 5 - get_distance(pos_ant, pos_resource)
					
						
				#Fairness post processing
				if beta!=0.0:
					# print("Fairness post processing")
					# if direction=='both':
					mult = (su[i] - np.mean(su))/1000
					if direction=='adv':
						mult = min(0,(su[i] - np.mean(su)))/1000
					elif direction=='dis':
						mult = max(0,(su[i] - np.mean(su)))/1000
					meanQ = np.mean(Qvals[i])

					# Different way of doing SI in this case, as each agent is independent
					# Reduce good actions for agents with higher su and increase good actions for agents with lower su
					for act in range(n_actions):
						if Qvals[i][act]>meanQ:
							Qvals[i][act] = Qvals[i][act] - beta*mult
						else:
							Qvals[i][act] = Qvals[i][act] + beta*mult
		
		# for i in range(n_agents):
		# 	print("Qvals", i, Qvals[i])
		#For each agent, select the best greedy action for now
		actions = [np.argmax(Qvals[i]) for i in range(n_agents)]
		
		# print(actions)
		return actions
	
	def get_value_grid(self, VF):
		# Return a grid of values for each grid location, calling VF.get for each location
		# VF is a value function class object
		gridsize = self.gridsize
		grid = np.zeros((gridsize, gridsize))
		obs = self.get_obs()
		# print(obs)
		for i in range(gridsize):
			for j in range(gridsize):
				# # Get the observation for this grid location. Assume agent 0 is here
				# agent_obs = obs[0][0]
				# agent_obs[0] = i
				# agent_obs[1] = j
				# # print(agent_obs)
				# Get the observation for this grid location. Assume agent 0 is here
				job = obs[1]
				agent_obs = obs[0][0]
				# print(agent_obs)
				relative_job = [i-job[0], j-job[1]]
				agent_obs[0] = relative_job[0]
				agent_obs[1] = relative_job[1]

				# Get the value of this location
				val = VF.get(np.array([agent_obs]))
				grid[i,j] = val
		
		return grid

	def get_post_dec_grid(self, VF):
		# Return a grid of values for each grid location, calling VF.get for each location
		# VF is a value function class object
		gridsize = self.gridsize*2 + 1
		grid = np.zeros((gridsize, gridsize))
		obs = self.get_obs()
		# print(obs)
		for i in range(gridsize):
			for j in range(gridsize):
				# Get the observation for this grid location. Assume agent 0 is here
				agent_obs = obs[0][0]
				agent_obs[0] = i/2
				agent_obs[1] = j/2
				# print(agent_obs)

				# Get the value of this location
				val = VF.get(np.array([agent_obs]))
				grid[i,j] = val
		
		return grid


class NewJobSchedulingEnvt:
	'''
	Many workers that desire to work on a job. As long as worker occupies the job's location, they get a reward
	Central agent approach. The constraint is no worker can be in the same location at the same time. 
	The allowed actions are only the grid locations immediately adjacent to the agent.

	Actions are converted into integer based on new grid location. 
	Actions that would lead outside the grid: consider separately or ignore?
		Going with ignore, they would map back to the current location.
	'''
	def __init__(self, 
	      n_agents, 
		  gridsize=5,
		  reallocate=True, 
		  simple_obs=False, 
		  warm_start=0,
		  past_discount=0.995,
		  ):
		self.n_agents = n_agents
		self.gridsize = gridsize
		
		self.warm_start = warm_start
		self.past_discount = past_discount

		self.reset()

		self.reallocate = reallocate
		self.simple_obs = simple_obs
		self.observation_space = ['loc','util', 'relative_job', 'relative_util', 'other_agents', 'grid', 'job']
		if self.simple_obs:
			self.observation_space = ['relative_job', 'grid', 'relative_util']
	
	def reset(self):
		# The grid is a 2D array of size gridsize x gridsize
		# Each agent is assigned a random location on the grid
		# Empty grid locations are 0
		# Agent locations are 1,2,3,4,5...
		# Job location is always fixed, so not tracked on the grid

		self.grid = np.zeros((self.gridsize, self.gridsize))
		ant = []
		#Random start locations
		for i in range(self.n_agents):
			loc = np.random.randint(0,self.gridsize,2)
			while self.grid[loc[0],loc[1]]!=0:
				loc = np.random.randint(0,self.gridsize,2)
			ant.append(np.random.randint(0,self.gridsize,2)) # TODO: Check. This probably needs fixing.
			self.grid[ant[i][0],ant[i][1]]=i+1
		ant = np.array(ant)

		self.job = np.random.randint(1,self.gridsize-1,2)
		self.job_reward = 1

		self.ant = ant
		self.targets = [None]*self.n_agents
		self.su = np.zeros(self.n_agents)

		w = 5 #width of the warm start distribution
		self.discounted_su = np.array([
			self.warm_start + np.random.rand()*w - w/2
			for _ in range(self.n_agents)])
		
	def map_grid_to_idx(self, i,j):
		return i*self.gridsize+j

	def map_idx_to_grid(self, idx):
		return idx//self.gridsize, idx%self.gridsize

	def pre_move(self, ant, dir):
		x, y = ant
		dir_map = {
			0: [0,0],
			1: [0,1],
			2: [0,-1],
			3: [-1,0],
			4: [1,0]
		}
		delta_x = dir_map[dir][0]
		delta_y = dir_map[dir][1]
		new_x = x + delta_x
		new_y = y + delta_y
		return new_x, new_y
		
	
	def move(self, ant, dir):
		#Move agent i in direction
		#dir is one of [stay, up, down, left, right]
		illegal=False
		x,y = ant
		new_x, new_y = self.pre_move(ant,dir)

		# Simple bounds checking.
		if 0<=new_x<self.gridsize and 0<=new_y<self.gridsize:
			# No need to check if someone else is there. That is handled by the ILP.
			x, y = new_x, new_y
		else:
			illegal=True
		return x,y, illegal
		
	# Each action is one of [stay, up, down, left, right], and is mapped to a unique grid lcoation
	def step(self, actions):
		# actions[i] is the action taken by agent i
		reset_job = False
		re = [0]*self.n_agents  #rewards. If an agent is on the job, get reward of 1
		for i in range(self.n_agents):
			if actions[i]!=-1:
				new_x, new_y, _ = self.move(self.ant[i], actions[i])			
				if self.grid[self.ant[i][0],self.ant[i][1]]==i+1:
					# If the agent is still at the old location, update the grid
					# If another update has already been made, don't update the grid
					self.grid[self.ant[i][0],self.ant[i][1]]=0
				self.ant[i][0] = new_x
				self.ant[i][1] = new_y
				self.grid[new_x, new_y]=i+1
	

			#Check if agent is on the job
			if self.ant[i][0]==self.job[0] and self.ant[i][1]==self.job[1]:
				re[i]=self.job_reward
				# reset_job = True
				
		# if reset_job:
		# 	self.job = np.random.randint(0,self.gridsize,2)	
		self.su+=np.array(re)
		#Update the discounted utilities
		self.discounted_su = self.discounted_su*self.past_discount + np.array(re)

		# Add supplemental reward based on distance from job
		for i in range(self.n_agents):
			re[i] += -0.1*get_distance(self.ant[i], self.job)
		
		return re

	def get_state(self):
		return self.ant, self.job, self.su, self.discounted_su
		
	def set_state(self, state):
		self.ant, self.job, self.su, self.discounted_su = state

	def get_obs(self):
		#Gets the state of the environment (Vector of each agent states)
		state = []
		agents = []
		for i in range(self.n_agents):
			h={}
			h['loc'] = [self.ant[i][0], self.ant[i][1]]
			h['util'] = self.su[i]
			h['relative_util'] = self.discounted_su[i]/np.mean(self.discounted_su) - 1

			#Get info about other agents
			others = []
			# append locations of all other agents
			for j in range(self.n_agents):
				if j!=i:
					others.append([
						self.ant[j][0],self.ant[j][1],
						self.su[j],
					])
			#flatten
			others = [feature for other in others for feature in other]
			h['other_agents'] = others

			# alt: get the 3x3 grid around the agent
			h['grid'] = []
			for x in range(self.ant[i][0]-1, self.ant[i][0]+2):
				for y in range(self.ant[i][1]-1, self.ant[i][1]+2):
					if 0<=x<self.gridsize and 0<=y<self.gridsize:
						h['grid'].append(self.grid[x,y])
					else:
						h['grid'].append(-1)
			
			h['job'] = copy.deepcopy(self.job)
			#relative job location
			h['relative_job'] = [self.ant[i][0]-self.job[0], self.ant[i][1]-self.job[1]]
			# Relative job location does not work, would need to update it in the post_decision state as well

			#feats
			feats = []
			# flist = ['loc','util', 'relative_job', 'relative_util', 'other_agents', 'grid', 'job'] # this can work with distance penalty, but takes a long time to train >200 episodes
			# flist = ['loc', 'relative_util', 'grid', 'job']
			# flist = ['loc', 'job']
			# flist = ['relative_job', 'grid']
			# flist = ['relative_job', 'grid', 'relative_util'] # This seems to work with lr=0.0003, hidden_size=20
			# flist = ['relative_job', 'relative_job', 'grid', 'relative_util'] # second relative job because first is overwritten by post-decision loc: Probably not needed?
			# ^Nope. Worse. Even with 40 features
			# if self.simple_obs:
				# flist = ['relative_job', 'grid', 'relative_util']
			flist = self.observation_space
			
			for f in flist:
				# if it is an iterable, extend, else append
				if f=='util' or f=='relative_util':
					feats.append(h[f])
				else:
					feats.extend(h[f])

			agents.append(feats)

		state.append(agents)
		state.append(copy.deepcopy(self.job))

		return state

	def render(self, VF=None):
		text_render = True
		if not text_render:
			for i in range(self.n_agents):
				plt.scatter(self.ant[i][0], self.ant[i][1], color = 'blue')
				# Add text to show scores
				plt.text(self.ant[i][0], self.ant[i][1], str(self.su[i]))
			plt.scatter(self.job[0], self.job[1], color = 'red', marker='x')
			plt.axis("off")
			plt.axis("equal")
			# Make gridlines
			for i in range(self.gridsize+1):
				# Horizontal lines
				l = i - 0.5
				plt.plot([-0.5,self.gridsize-0.5],[l,l], color='black')
				# Vertical lines
				plt.plot([l,l],[-0.5,self.gridsize-0.5], color='black')	
				
			plt.xlim(-1 , self.gridsize)
			plt.ylim(-1 , self.gridsize)
			plt.ion()
			plt.pause(0.01)
			plt.close()
		else:
			## Text rendering
			# Print a * for each agent, and a # for the job. Each grid location is 3 spaces
			pstr = ""
			for i in range(self.gridsize):
				for j in range(self.gridsize):
					if self.grid[i,j]!=0:
						if self.job[0]==i and self.job[1]==j:
							# print("#*", end="  ")
							pstr += "#*  "
						else:
							# print("*", end="   ")
							pstr += "*   "
					elif self.job[0]==i and self.job[1]==j:
						# print("#", end="   ")
						pstr += "#   "
					else:
						# print(".", end="   ")
						pstr += ".   "
				# print()
				pstr += "\n"
			# print("Score", self.su)
			pstr += "Score: " + str(self.su) + "\n\n"
			print(pstr)


	

	def get_post_decision_state_agent(self, state, action, ind):
		# For a single agent
		s_i = copy.deepcopy(state)
		if action!=-1:
			new_x, new_y = self.pre_move(s_i[:2], action)
			#apply action
			s_i[0] = new_x 
			s_i[1] = new_y
		return s_i
	
	
	def get_post_decision_states(self, obs, actions):
		states = []
		for i in range(self.n_agents):
			s_i = self.get_post_decision_state_agent(obs[0][i], actions[i], i)
			states.append(s_i)
		return states

	def get_valid_locations(self, envt):
		# Get the valid locations for each agent
		# obs is the observation of the environment
		# Returns a mapping of each action to a valid location
		valid_locs = []
		for i in range(self.n_agents):
			h = envt.ant[i]
			valid = {}
			for act in range(5):
				new_x, new_y = self.pre_move(h[:2], act)
				if 0<=new_x<self.gridsize and 0<=new_y<self.gridsize:
					# If the move is legal, add it to the valid locations
					valid[act] = self.map_grid_to_idx(new_x, new_y)
				else:
					# valid[act] = -1
					valid[act] = self.map_grid_to_idx(h[0], h[1])
			valid_locs.append(valid)
		return valid_locs


	def compute_best_actions(self, model, envt, obs, epsilon=0.0, beta=0.0, direction='both', use_greedy=False):
		# envt is an environment class object with the state set to desired state
		# targets, n_agents, n_actions, su
		n_agents = envt.n_agents
		n_actions = 5
		n_locs = envt.gridsize**2
		su = envt.su

		Qvals = [[-1000000 for _ in range(n_actions)] for _ in range(n_agents)]

		valid_locs = envt.get_valid_locations(envt)

		#Get a random action with probability epsilon
		if np.random.rand()<epsilon:
			Qvals = [[np.random.rand() for ind in range(n_actions)] for _ in range(n_agents)]
		else:
			# First action is to do nothing.
			resource = copy.deepcopy(obs[1])
			for i in range(n_agents):
				h = copy.deepcopy(obs[0][i])
				for act in range(n_actions):
					h_post = self.get_post_decision_state_agent(h, act, i)

					if use_greedy:
						new_loc = self.pre_move(envt.ant[i], act)
						Qvals[i][act] = 5 - get_distance(new_loc, envt.job)
					else:
						Qvals[i][act] = float(model.get(np.array([h_post])))

					# Custom calculation, naive greedy vf
					# pos_ant = np.array(h_post[:2])
					# pos_resource = np.array(h_post[-2:])
					# Qvals[i][act] = 5 - get_distance(pos_ant, pos_resource)
					
						
				#Fairness post processing
				if beta!=0.0:
					# print("Fairness post processing")
					# if direction=='both':
					mult = (su[i] - np.mean(su))/1000
					if direction=='adv':
						mult = min(0,(su[i] - np.mean(su)))/1000
					elif direction=='dis':
						mult = max(0,(su[i] - np.mean(su)))/1000
					meanQ = np.mean(Qvals[i])

					# Different way of doing SI in this case, as each agent is independent
					# Reduce good actions for agents with higher su and increase good actions for agents with lower su
					for act in range(n_actions):
						if Qvals[i][act]>meanQ:
							Qvals[i][act] = Qvals[i][act] - beta*mult
						else:
							Qvals[i][act] = Qvals[i][act] + beta*mult
		

		# Convert Qvals to n_agens x n_locations for ILP matching
		Qvals_loc = [[-1000000 for _ in range(n_locs)] for _ in range(n_agents)]
		for i in range(n_agents):
			for act in range(n_actions):
				if act in valid_locs[i]:
					Qvals_loc[i][valid_locs[i][act]] = Qvals[i][act]
		# This has to be done after Qvals calculation to account for SI fairness
		# for i in range(n_agents):
		# 	print("Qvals", i, Qvals[i])
		#For each agent, select the best greedy action for now
		# actions = [np.argmax(Qvals[i]) for i in range(n_agents)]
		resource_counts = [1 for _ in range(n_locs)]
		# Locations with agents are not valid
		# Add agent constraints: list of locations of other agents
		agent_constraints = []
		for i in range(n_agents):
			illegal_locs = [self.map_grid_to_idx(envt.ant[j][0], envt.ant[j][1]) for j in range(n_agents) if j!=i]
			agent_constraints.append(illegal_locs)

		locations = get_assignment(Qvals_loc, resource_counts, agent_constraints)

		# Convert locations to actions
		actions = [-1]*n_agents
		for i in range(n_agents):
			for act in valid_locs[i]:
				if valid_locs[i][act]==locations[i]:
					actions[i] = act
					break 
		if -1 in actions:
			print("Invalid action")
			print(actions)
		# print(actions)
		return actions
	

class PlantEnvt:
	def __init__(self, 
			n_agents=5,
			gridsize=12,
			n_resources=8,
			reallocate=True, 
			simple_obs=False, 
			warm_start=0,
			past_discount=0.995,
			):
		"""
		3 types of resources
		Agents each have unique requirements for combinations of resources
		Once the requirements are satisfied, agents produce one 'unit' and receive a reward, consuming the resource.
		TODO: This can also have 2 modes: allocate grid locations, or allocate resources.
		"""
		self.n_agents = n_agents
		self.gridsize = gridsize
		self.n_resources = n_resources
		self.simple_obs = simple_obs
		self.warm_start = warm_start
		self.past_discount = past_discount

		self.observation_space = ['neighborhood', 'posessions', 'requirement' 'su', 'relative_su']
		self.reset()
	
	def reset(self):
		# The grid is a 2D array of size gridsize x gridsize
		# Each agent is assigned a random location on the grid
		# Empty grid locations are 0
		# Agent locations are 4.
		# Resource locations are 1+ resource type (1-3)

		self.grid = np.zeros((self.gridsize, self.gridsize))
		ant = []
		#Random start locations
		for i in range(self.n_agents):
			loc = np.random.randint(0,self.gridsize,2)
			while self.grid[loc[0],loc[1]]!=0:
				loc = np.random.randint(0,self.gridsize,2)
			ant.append(loc)
			self.grid[ant[i][0],ant[i][1]]=4
		ant = np.array(ant)

		# generate resources
		resources = []
		resource_types = []
		for i in range(self.n_resources):
			resources.append(np.random.randint(1,self.gridsize-1,2))
			resource_types.append(np.random.randint(3))

		requirements=[[2, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 1, 2]]

		self.ant = ant
		self.targets = [None]*self.n_agents
		self.su = np.zeros(self.n_agents)
		self.resources = np.array(resources)
		self.resource_types = np.array(resource_types)
		self.requirements = np.array(requirements)
		self.posessions = np.array([[0,0,0] for _ in range(self.n_agents)])

		w = 5 #width of the warm start distribution
		self.discounted_su = np.array([
			self.warm_start + np.random.rand()*w - w/2
			for _ in range(self.n_agents)])
		
	def map_grid_to_idx(self, i,j):
		return i*self.gridsize+j

	def map_idx_to_grid(self, idx):
		return idx//self.gridsize, idx%self.gridsize

	def pre_move(self, ant, dir):
		x, y = ant
		dir_map = {
			0: [0,0],
			1: [0,1],
			2: [0,-1],
			3: [-1,0],
			4: [1,0]
		}
		delta_x = dir_map[dir][0]
		delta_y = dir_map[dir][1]
		new_x = x + delta_x
		new_y = y + delta_y
		return new_x, new_y
		
	
	def move(self, ant, dir):
		#Move agent i in direction
		#dir is one of [stay, up, down, left, right]
		illegal=False
		x,y = ant
		new_x, new_y = self.pre_move(ant,dir)

		# Simple bounds checking.
		if 0<=new_x<self.gridsize and 0<=new_y<self.gridsize:
			# No need to check if someone else is there. That is handled by the ILP.
			x, y = new_x, new_y
		else:
			illegal=True
		return x,y, illegal
		
	# Each action is one of [stay, up, down, left, right], and is mapped to a unique grid lcoation
	def step(self, actions):
		# actions[i] is the action taken by agent i
		re = [0]*self.n_agents  #rewards. If an agent constructs a unit, get a reward of one.
		# Shaping reward 
		re_shaping = np.array([0]*self.n_agents)
		newgrid = np.zeros((self.gridsize, self.gridsize))
		consumed_resources = []
		for i in range(self.n_agents):
			if actions[i]!=-1:
				new_x, new_y, _ = self.move(self.ant[i], actions[i])			
				self.ant[i][0] = new_x
				self.ant[i][1] = new_y
				newgrid[new_x, new_y]=i+1

				#Check if agent picked up a resource
				for j in range(self.n_resources):
					if self.resources[j][0]==new_x and self.resources[j][1]==new_y:
						consumed_resources.append(j)
						r_type = self.resource_types[j]
						self.posessions[i][r_type]+=1
						if self.posessions[i][r_type]<=self.requirements[i][r_type]:
							re_shaping[i]+=0.1		
		
		#replenish consumed resources
		for j in consumed_resources:
			self.resources[j] = np.random.randint(1,self.gridsize-1,2)
			self.resource_types[j] = np.random.randint(3)
		
		# Construct units
		for i in range(self.n_agents):
			n_units = 10000
			for j in range(3):
				if self.requirements[i][j]==0:
					continue
				else:
					count = int(self.posessions[i][j]/self.requirements[i][j])
					if count<n_units:
						n_units = count
			re[i]+=n_units
			for j in range(3):
				self.posessions[i][j]-=self.requirements[i][j]*n_units

		self.su+=np.array(re)
		re_return = np.array(re) + re_shaping 		
		#Update the discounted utilities
		self.discounted_su = self.discounted_su*self.past_discount + np.array(re)

		# Add supplemental reward based on distance from closest useful resource
		for i in range(self.n_agents):
			closest_useful = 100000
			for j in range(self.n_resources):
				r_type = self.resource_types[j]
				if self.posessions[i][r_type]<self.requirements[i][r_type]:
					dist = get_distance(self.ant[i], self.resources[j])
					if dist<closest_useful:
						closest_useful = dist
			re_return[i] += -0.01*closest_useful
		
		return re_return

	def get_state(self):
		return self.ant, self.resources, self.resource_type, self.requirements, self.su, self.discounted_su
	
	def set_state(self, state):
		self.ant, self.resources, self.resource_type, self.requirements, self.su, self.discounted_su = state

	def get_obs(self):
		agents = []
		# TODO
		return [agents, []]
	
	def render(self):
		
		return

	def get_post_decision_state_agent(self, obs, action, idx):
		s_i = copy.deepcopy(obs)
		if len(s_i)>1:
			s_i[1] += action*self.agent_scores[idx]
		return s_i

	def get_post_decision_states(self, obs, actions):
		states = []
		for i in range(self.n_agents):
			s_i = self.get_post_decision_state_agent(obs[0][i], actions[i], i)
			states.append(s_i)
		return states
	
	def compute_best_actions(self, model, envt, obs, epsilon=0.0, beta=0.0, direction='both', use_greedy=False, val=False):
		# Greedy strategy: Fastest agent gets the resource
		n_agents = envt.n_agents
		Qvals = [[-1000000 for _ in range(2)] for _ in range(n_agents)]
		#Get a random action with probability epsilon
		if np.random.rand()<epsilon:
			Qvals = [[np.random.rand() for act in range(2)] for _ in range(n_agents)]
		else:
			for i in range(n_agents):
				for j in range(2):
					s_i = self.get_post_decision_state_agent(obs[0][i], j, i)
					Qvals[i][j] = float(model.get(np.array([s_i])))
		if val:
			print(Qvals)
		actions = get_assignment(Qvals, [n_agents, 1])
		return actions