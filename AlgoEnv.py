class AlgoTradingEnv():
	"""docstring for AlgoTrading"""
	def __init__(self, DataFrame):
		self.i = 0
		self.observation_space = []
		self.action_space = []
		self.DataFrame = DataFrame
		self.numOfEpi = len(self.DataFrame.index)
		self.observation_space.append(self.DataFrame.loc[self.i][0])
		self.action_space.append(self.DataFrame.loc[self.i][1])

	def step(self, action):
		self.action = action
		self.i += 1
		reward = 0.0
		state = 0.0
		if self.i == self.numOfEpi:
			done = True
			print (" State: ", self.observation_space, '\n', "Action: ", self.action_space)
		else:
			done = False
			reward = self.RLReward()
			state = self.DataFrame.loc[self.i][0]
			self.observation_space[0] = self.DataFrame.loc[self.i][0]
			self.action_space[0] = self.DataFrame.loc[self.i][1]
		return (state, reward, done)

	def reset(self):
		self.i = 0
		self.observation_space[0] = self.DataFrame.loc[self.i][0]
		self.action_space[0] = self.DataFrame.loc[self.i][1]
		return self.observation_space

	def RLReward(self):
	    dif = 0
	    for O, A in zip(self.action_space, self.action):
	        if O - A == 0:
	            dif += 1
	        else:
	            dif += 0
	    reward = dif/len(self.action_space)
	    return reward
