import copy
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

def get_distance(a,b):
	#Get distance between two points
	return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

#TODO: Wrap this in a class
class MatthewEnvt:
	def __init__(self, n_agents, n_resources, max_size, min_size=0.01, size_update=0.005, base_speed=0.01, reallocate=False, simple_obs=False):
		self.n_agents = n_agents
		self.n_resources = n_resources
		self.max_size = max_size
		self.min_size = min_size
		self.size_update = size_update
		self.base_speed = base_speed

		self.reset()

		self.reallocate = reallocate
		self.simple_obs = simple_obs
	
	def reset(self):
		ant = []
		size = []
		speed = []
		su = [0]*self.n_agents
		targets = []
		size_range = (self.max_size-self.min_size)/5
		for i in range(self.n_agents):
			ant.append(np.random.rand(2))
			size.append(self.min_size + np.random.rand()*size_range)
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
	
	def set_speed_from_sizes(self):
		for i in range(self.n_agents):
			self.speed[i] = self.base_speed + self.size[i]
			# self.speed[i] = self.base_speed + self.size[i]**3*10

	def step(self, actions):
		clear_targets = False
		# Actions just decide mapping of agents to resources
		re = [0]*self.n_agents  #rewards. If an agent picks up a resource, get reward of 1
		for i in range(self.n_agents):
			if actions[i]!=-1:
				self.targets[i] = [actions[i]]
				#Add the time to reach the resource to the targets vector
				self.targets[i].append(get_distance(self.ant[i],self.resource[actions[i]])/self.speed[i])

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
		self.su+=np.array(re)

		return re

	def get_state(self):
		return self.ant, self.resource, self.targets, self.size, self.speed, self.su
	def set_state(self, state):
		self.ant, self.resource, self.targets, self.size, self.speed, self.su = state
	
	def get_obs(self):
		#Gets the state of the environment (Vector of each agent states)
		#agent positions, resource positions, size vector, speed vector, number of agents
		state=[]
		agents = []
		for i in range(self.n_agents):
			h=[]
			h.append(self.ant[i][0])
			h.append(self.ant[i][1])
			h.append(self.size[i])
			h.append(self.speed[i])

			# # Get relative size of current agent to other agents
			# rat = self.size[i]/np.mean(self.size) - 1
			# h.append(rat)

			if self.simple_obs:
				# #get number of agents without a target resource
				n = 0
				for j in range(self.n_agents):
					if self.targets[j] is None:
						n+=1
				h.append(n)
				
				# Get relative size of current agent to other agents
				rat = self.size[i]/np.mean(self.size) - 1
				h.append(rat)
			else:
				# Get relative size of current agent to other agents
				rat = self.size[i]/np.mean(self.size) - 1
				h.append(rat)
				others = []
				# append locations of all other agents, sorted by distance to target resource
				for j in range(self.n_agents):
					if j!=i:
						others.append([self.ant[j][0],self.ant[j][1],self.size[j]])
				if self.targets[i] is not None:
					others.sort(key=lambda x: get_distance(x[:2],self.resource[self.targets[i][0]]))
				else:
					others.sort(key=lambda x: get_distance(x[:2],self.ant[i]))

				for j in range(self.n_agents-1):
					h.append(others[j][0])
					h.append(others[j][1])
					h.append(others[j][2])
			
			#Current target resource
			if self.targets[i] is not None:
				h.append(self.resource[self.targets[i][0]][0]) #target resource x
				h.append(self.resource[self.targets[i][0]][1]) #target resource y
				h.append(self.targets[i][1]) #time remaining
			else:
				h.append(-1) #target resource x
				h.append(-1) #target resource y
				h.append(100) #time remaining

			agents.append(h)

		state.append(agents)
		state.append(copy.deepcopy(self.resource))

		return state

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


def get_obs(ant, resource, targets, sizes, speeds, n_agents):
	#Gets the state of the environment (Vector of each agent states)
	#agent positions, resource positions, size vector, speed vector, number of agents
	state=[]
	agents = []
	for i in range(n_agents):
		h=[]
		h.append(ant[i][0])
		h.append(ant[i][1])
		h.append(sizes[i])
		h.append(speeds[i])

		# #get number of agents without a target resource
		n = 0
		for j in range(n_agents):
			if targets[j] is None:
				n+=1
		h.append(n)

		# Get relative size of currnt agent to other agents
		rat = sizes[i]/np.mean(sizes) - 1
		h.append(rat)

		# others = []
		# # append locations of all other agents, sorted by distance to target resource
		# for j in range(n_agents):
		# 	if j!=i:
		# 		others.append([ant[j][0],ant[j][1],sizes[j]])
		# if targets[i] is not None:
		# 	others.sort(key=lambda x: get_distance(x[:2],resource[targets[i][0]]))
		# else:
		# 	others.sort(key=lambda x: get_distance(x[:2],ant[i]))
		
		# for j in range(n_agents-1):
		# 	h.append(others[j][0])
		# 	h.append(others[j][1])
		# 	h.append(others[j][2])

		#Current target resource
		if targets[i] is None:
			h.append(-1)
			h.append(-1)
			h.append(100) #Time to reach target resource
		else:
			h.append(resource[targets[i][0]][0]) #Target resource x
			h.append(resource[targets[i][0]][1]) #Target resource y
			h.append(targets[i][1]) #Time to reach target resource
		
		agents.append(h)
	
	state.append(agents)
	state.append(copy.deepcopy(resource))

	return state

def step(ant,resource, targets, n_resources, n_agents,size,speed,action, max_size=0.1, reallocate=False):
	clear_targets = False
	# Actions just decide mapping of agents to resources
	re=[0]*n_agents  #rewards. If an agent picks up a resource, get reward of 1
	for i in range(n_agents):
		if action[i]!=-1:
			targets[i] = [action[i]]
			#Add the time to reach the resource to the targets vector
			targets[i].append(get_distance(ant[i],resource[action[i]])/speed[i])
		
		#Move each agent towards its target resource
		if targets[i] is not None:
			#Other agents can't pick up the resources if they are claimed
			#if target is overlapped by the agent, remove it
			if targets[i][1]<=1:
				ant[i][0] = resource[targets[i][0]][0]
				ant[i][1] = resource[targets[i][0]][1]
				re[i]=1 #Get reward

				#Reset target resource
				resource[targets[i][0]]=np.random.rand(2)
				size[i]=min(size[i]+0.005,max_size)
				# size[i]=min(size[i]+0.05,1.5)
				speed[i]=0.01+size[i]
				targets[i]=None
				clear_targets = True
			else:
				#Move agent towards target resource. Each step, move 1/time_remaining of the way
				ant[i][0]+=(resource[targets[i][0]][0]-ant[i][0])/targets[i][1]
				ant[i][1]+=(resource[targets[i][0]][1]-ant[i][1])/targets[i][1]
				targets[i][1]-=1
				if get_distance(ant[i],resource[targets[i][0]])<size[i]:
					re[i]=1 #Get reward
					#Reset target resource
					resource[targets[i][0]]=np.random.rand(2)
					size[i]=min(size[i]+0.005,max_size)
					speed[i]=0.01+size[i]
					targets[i]=None
					clear_targets = True

		else:
			#Move in a random direction or stay still
			p_move = 0.8
			dr = np.random.rand()*2*np.pi
			if np.random.rand()<p_move:
				ant[i][0]+=np.cos(dr)*speed[i]
				ant[i][1]+=np.sin(dr)*speed[i]
		if ant[i][0]<0:
			ant[i][0]=0
		if ant[i][0]>1:
			ant[i][0]=1
		if ant[i][1]<0:
			ant[i][1]=0
		if ant[i][1]>1:
			ant[i][1]=1

	# If any resources were picked up, reset the targets
	if reallocate:
		if clear_targets:
			# print("Clearing Targets")
			for i in range(n_agents):
				targets[i]=None

	return ant,resource,size,speed,re, targets