import gurobipy as gp
from gurobipy import GRB

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
				assignment.append(j-1)
	return assignment