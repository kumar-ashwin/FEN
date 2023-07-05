import copy
import numpy as np

#TODO: Wrap this in a class

def get_distance(a,b):
	#Get distance between two points
	return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

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
		# n = 0
		# for j in range(n_agents):
		# 	if targets[j] is None:
		# 		n+=1
		# h.append(n)

		# Get relative size of currnt agent to other agents
		rat = sizes[i]/np.mean(sizes) - 1
		h.append(rat)

		others = []
		# append locations of all other agents, sorted by distance to target resource
		for j in range(n_agents):
			if j!=i:
				others.append([ant[j][0],ant[j][1],sizes[j]])
		if targets[i] is not None:
			others.sort(key=lambda x: get_distance(x[:2],resource[targets[i][0]]))
		else:
			others.sort(key=lambda x: get_distance(x[:2],ant[i]))
		
		for j in range(n_agents-1):
			h.append(others[j][0])
			h.append(others[j][1])
			h.append(others[j][2])

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

	return ant,resource,size,speed,re