#!/usr/bin/env python
import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse
from keras.layers import Dense
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import os
from collections import deque
import random
from keras import initializers
import matplotlib.pyplot as plt
from keras import backend as keras_back

class QNetwork(): 

	def __init__(self, environment_name):
		self.model = Sequential()
		self.env = gym.make(environment_name)
		self.alpha = 0.00025

		self.model.add(Dense(16, input_dim=self.env.observation_space.shape[0], use_bias=True, activation='relu'))
		self.model.add(Dense(24, use_bias=True, activation='relu'))
		self.model.add(Dense(36, use_bias=True, activation='relu'))
		self.model.add(Dense(self.env.action_space.n, activation='linear'))
		self.model.compile(loss=self._huber_loss, optimizer=Adam(lr=self.alpha))

	def _huber_loss(self, target, prediction):
		error = prediction - target
		return keras_back.mean(keras_back.sqrt(1+keras_back.square(error))-1, axis=-1)

class DQN_Agent():
	
	def __init__(self, environment_name, render=False):

		self.env = gym.make(environment_name)
		self.state_size = self.env.observation_space.shape[0]
		self.action_size = self.env.action_space.n

		#experience replay 
		self.replay_memory_size=50000
		self.memory = deque(maxlen=self.replay_memory_size)
		self.batch_size = 32
		self.burn_in = 10000

		if environment_name == 'MountainCar-v0':
			self.gamma = 1
			self.episodes = 3000 #given in handout
			self.iterations = 201 #check how many, stops at 200
			self.terminate = 0
			self.environment_num = 1 
		elif environment_name == 'CartPole-v0':
			self.gamma = 0.99
			self.iterations = 1000000 #given in handout
			self.episodes = 100000000 #check how many
			self.terminate = 1
			self.environment_num = 2
		self.train_epsilon = 1
		self.train_evaluate_epsilon = 0.05
		self.final_update = 50000
		self.q_network = QNetwork(environment_name)


	def epsilon_greedy_policy(self, q_values,train_test_var):
		# Creating epsilon greedy probabilities to sample from.             
		rand = np.random.uniform(0,1)

		if train_test_var:
			if(rand<=self.train_epsilon):
				# print("random")
				return np.random.randint(0,self.action_size)
			else:
				# print("greedy")
				return self.greedy_policy(q_values)
		else:
			if(rand<=self.train_evaluate_epsilon):
				return np.random.randint(0,self.action_size)
			else:
				return self.greedy_policy(q_values) 

	def greedy_policy(self, q_values):
		# Creating greedy policy for test time. 
		return np.argmax(q_values) 



	def train(self):

		save_dir = os.path.join(os.getcwd(), 'saved_models_dqn_replay')
		if not os.path.isdir(save_dir):
			os.makedirs(save_dir)
		os.chdir(save_dir)

		total_updates = 0
		success_count = 0

		starting_epsilon = self.train_epsilon

		for i_episode in range(self.episodes):
			if self.environment_num==2 and total_updates > 1000000:
				break

			state = self.env.reset()
			state = np.reshape(state,[1,self.state_size])	

			# print('Episode no {}, epsilon is {}'.format(i_episode+1, self.train_epsilon))
			total_reward = 0
			ep_terminate = False		

			for t_iter in range(self.iterations):
				action = self.epsilon_greedy_policy(self.q_network.model.predict(state),1) 

				if self.environment_num == 1:
					if (len(self.memory)> self.burn_in) and total_updates<=100000:
						self.train_epsilon = -((starting_epsilon-0.1)/100000)*total_updates + starting_epsilon #decay epsilon 1 to 0.1
					elif total_updates == 0:
						self.train_epsilon = 1
					else:
						self.train_epsilon = 0.1
				elif self.environment_num==2:
					if (len(self.memory)> self.burn_in) and total_updates<=500000:
						self.train_epsilon = -((starting_epsilon-0.1)/500000)*total_updates + starting_epsilon #decay epsilon 1 to 0.1
					elif total_updates == 0:
						self.train_epsilon = 1
					else:
						self.train_epsilon = 0.1

				next_state, reward, done, info = self.env.step(action)
				reward = reward if not done else -10
				next_state = np.reshape(state,[1,self.state_size])	
				self.memory.append((state, action, reward, next_state, done))
				total_reward+=reward	
				#TO DO
				#Are targets stationary		
				state=next_state		

				if len(self.memory) > self.burn_in:
					# print("accessing memory")
					minibatch = random.sample(self.memory, self.batch_size)
					for m_state, m_action, m_reward, m_next_state, m_done in minibatch:
						m_q_value_prime = m_reward
						if not m_done:
							m_q_value_prime = m_reward + self.gamma * np.max(self.q_network.model.predict(m_next_state)[0])
							m_q_value_target = self.q_network.model.predict(m_state)
							m_q_value_target[0][m_action] = m_q_value_prime
							self.q_network.model.fit(m_state,m_q_value_target,epochs=1, verbose=0)	
							total_updates+=1
							w = self.q_network.model.get_weights()

				if done:
					if (self.terminate==0 and total_reward>-200) or (self.terminate==1 and total_reward>=100):
					# if (self.terminate==0 and total_reward>-200) or (self.terminate==1 and t_iter+1==200):
						success_count+=1
						# ep_terminate = True
						# break	
					print("Episode {} total reward {} epsilon {} successes {} total_updates {} Score {}"
																.format(i_episode,total_reward,self.train_epsilon,success_count, total_updates, t_iter + 1))	
					break		

				
				if (total_updates) % 10000 == 0:
					model_name = 'lqn_%d_%d_model.h5' %(self.environment_num,total_updates) 
					filepath = os.path.join(save_dir, model_name)
					self.q_network.model.save(model_name)		

			# print("Total updates is",total_updates)
			# print('----------------')				


		print("Saving final model at",total_updates)
		model_name = 'lqn_%d_%d_model_final.h5' %(self.environment_num,total_updates) 
		filepath = os.path.join(save_dir, model_name)
		self.q_network.model.save(model_name)
		print("Total success is ",success_count)
			# if ep_terminate==True:
			# 	model_name = 'lqn_%d_model.h5' %(total_updates)
			# 	filepath = os.path.join(save_dir, model_name)
			# 	self.q_network.model.save(model_name)
			# 	break



		# If you are using a replay memory, you should interact with environment here, and store these 
		# transitions to memory, while also updating your model.

	def test(self, model_file=None):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory. 
		num_episodes = 20
		plot_mat = np.zeros((num_episodes, (self.final_update/10000) + 1))
		load_dir = os.path.join(os.getcwd(), 'saved_models_lqn_final')
		print(self.final_update)
		for i in range(10000, self.final_update + 10000, 10000):
			model_name = 'lqn_%d_%d_model.h5' %(self.environment_num,i)
			filepath = os.path.join(load_dir, model_name)
			test_model = load_model(filepath)
			for e in range(num_episodes):
				total_epi_reward = 0
				print('Episode no ',e+1)
				state = self.env.reset()
				state = np.reshape(state,[1,self.state_size])	

				for t_iter in range(self.iterations):

					action = self.epsilon_greedy_policy(self.q_network.model.predict(state),0) 					

					self.env.render()
					next_state, reward, done, info = self.env.step(action)
					next_state = np.reshape(state,[1,self.state_size])	
					total_epi_reward+=reward
					
					# if (self.terminate==0 and state[0][0]==0.5) or (self.terminate==1 and t_iter==200):
					# 	print("success")
					# 	ep_terminate = True
					# 	break

					if done:
						print("Model %d" %(i))
						print("Episode finished after {} iterations with episodic reward %d".format(t_iter+1)%(total_epi_reward)) 
						plot_mat[e,(i/10000) - 1] = total_epi_reward
						break			
					state = next_state

		x = range(1,num_episodes + 1)
		for i in range(len(x)):
			plot_mat[i,(self.final_update/10000)] = x[i]


		for i in range(1,self.final_update/10000):
			plt.figure(i)
			print(plot_mat[:,(self.final_update/10000)])
			print(plot_mat[:,i])
			plt.plot(plot_mat[:,(self.final_update/10000)], plot_mat[:,i])
			plt.title('%d Model' %(i))
			# plt.show()			
			plt.savefig('graph_%d.eps' %(i),bbox_inches='tight' )

		exit()

  #   def burn_in_memory():
		# # # Initialize your replay memory with a burn_in number of episodes / transitions. 

		# pass

def parse_arguments():
	parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
	parser.add_argument('--env',dest='env',type=str)
	parser.add_argument('--render',dest='render',type=int,default=0)
	parser.add_argument('--train',dest='train',type=int,default=1)
	parser.add_argument('--model',dest='model_file',type=str)
	return parser.parse_args()

def main(args):

	args = parse_arguments()
	environment_name = args.env
	dqn_agen = DQN_Agent(environment_name)

	dqn_agen.train()
	# dqn_agen.test()

	# Setting the session to allow growth, so it doesn't allocate all GPU memory. 
	gpu_ops = tf.GPUOptions(allow_growth=True)
	config = tf.ConfigProto(gpu_options=gpu_ops)
	sess = tf.Session(config=config)

	# Setting this as the default tensorflow session. 
	keras.backend.tensorflow_backend.set_session(sess)

	# You want to create an instance of the DQN_Agent class here, and then train / test it. 

if __name__ == '__main__':
	main(sys.argv)

# class Replay_Memory():

# 	def __init__(self, memory_size=50000, burn_in=10000):

# 		# The memory essentially stores transitions recorder from the agent
# 		# taking actions in the environment.

# 		# Burn in episodes define the number of episodes that are written into the memory from the 
# 		# randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced. 
# 		# A simple (if not the most efficient) was to implement the memory is as a list of transitions. 
# 		pass

# 	def sample_batch(self, batch_size=32):
# 		# This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples. 
# 		# You will feed this to your model to train.
# 		pass

# 	def append(self, transition):
# 		# Appends transition to the memory. 	
# 		pass

