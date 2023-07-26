import copy
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

def get_distance(a,b):
	#Get distance between two points
	return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

#TODO: Wrap this in a class
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
		  ):
		self.n_agents = n_agents
		self.n_resources = n_resources
		self.max_size = max_size
		self.min_size = min_size
		self.size_update = size_update
		self.base_speed = base_speed
		self.warm_start = warm_start
		self.past_discount = past_discount

		self.reset()

		self.reallocate = reallocate
		self.simple_obs = simple_obs
	
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
		#Update the discounted su
		self.discounted_su = self.discounted_su*self.past_discount + np.array(re)

		return re

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
			flist = ['loc','size','speed', 'eaten','n_other_free_agents','relative_size', 'relative_su', 'other_agents', 'target_resource']
			if self.simple_obs:
				flist = ['loc','size','speed','n_other_free_agents','relative_size', 'target_resource']
			
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