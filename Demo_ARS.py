import pandas as pd
import AlgoEnv
import ARS_Model

# Running the main code
data = pd.read_csv("PG_2007_2017.csv")
hp = Hp()
np.random.seed(hp.seed)

env = AlgoTradingEnv(data,3,1)
nb_inputs = env.observation_space[0].shape[0]
nb_outputs = env.action_space[0].shape[0]
policy = Policy(nb_inputs, nb_outputs)
normalizer = Normalizer(nb_inputs)
train(env, policy, normalizer, hp)
