class AlgoTradingEnv():
	"""This will create a algotrading env which accept data frame for training AI models"""
	def __init__(self, DataFrame, XColFromLeft, YcolFromRight):
		self.i = 0
		self.x = XColFromLeft
		self.y = YcolFromRight
		self.observation_space = []
		self.action_space = []
		self.DataFrame = DataFrame
		self.numOfEpi = len(self.DataFrame.index)
		self.observation_space.append(self.DataFrame.loc[self.i][:self.x].values)
		self.action_space.append(self.DataFrame.loc[self.i][-self.y:].values)

	def step(self, action):
		self.action = action
		self.i += 1
		reward = 0.0
		state = 0.0
		if self.i == self.numOfEpi:
			done = True	
		else:
			done = False
			reward = self.RLReward()
			state = self.DataFrame.loc[self.i][:self.x].values            
			self.observation_space[0] = self.DataFrame.loc[self.i][:self.x].values
			self.action_space[0] = self.DataFrame.loc[self.i][-self.y:].values
		return (state, reward, done)

	def reset(self):
		self.i = 0
		self.observation_space[0] = self.DataFrame.loc[self.i][:self.x].values
		self.action_space[0] = self.DataFrame.loc[self.i][-self.y:].values
		return self.observation_space[0]
    
	def RLReward(self):
		dif = 0
		reward = 0
		for O, A in zip(self.action_space[0], self.action):
			reward += (A - O) / O
		return reward
