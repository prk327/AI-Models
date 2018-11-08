class AlgoTradingEnv():
	"""docstring for AlgoTrading"""
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
		self.Actual = []
		self.Predicted = []
		self.Date = []
		self.R = 0.
		self.AbsuluteDifference = 0.
		self.SquareDifference = 0.
		self.i = 0
		self.df = {}
		self.observation_space[0] = self.DataFrame.loc[self.i][:self.x].values
		self.action_space[0] = self.DataFrame.loc[self.i][-self.y:].values
		return self.observation_space[0]

	def RLReward(self):
		self.AD = 0.
		self.SD = 0.
		reward = 0
		MAEReward = 0.
		for O, A, Date in zip(self.action_space[0], self.action, self.observation_space[0]): 
			reward += (A - O) / O
			self.AD += abs(O - A)
			self.SD += (O - A) ** 2
		self.R += reward
		self.AbsuluteDifference += self.AD
		self.SquareDifference += self.SD
		self.Actual.append(O)
		self.Predicted.append(A)
		self.Date.append(Date)
		return self.R
    
	def Summary(self):
		self.MAE = self.AbsuluteDifference / self.i
		self.MSE = self.SquareDifference / self.i
		self.RMSE = self.MSE ** (1/2)
		self.df["Date"] = self.Date
		self.df["Actual"] = self.Actual
		self.df["Predicted"] = self.Predicted
		return None
