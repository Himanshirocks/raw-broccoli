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
from gym import wrappers

class QNetwork():

	# This class essentially defines the network architecture. 
	# The network should take in state of the world as an input, 
	# and output Q values of the actions available to the agent as the output. 

	def __init__(self, environment_name):
		self.env = gym.make(environment_name)
		self.model = Sequential()
		self.state_size = self.env.observation_space.shape[0]
		self.action_size = self.env.action_space.n
		self.alpha = 0.001

		self.model.add(Dense(self.action_size, input_dim=self.state_size,use_bias=True, activation='linear'))
		self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha))


class DQN_Agent():

	# In this class, we will implement functions to do the following. 
	# (1) Create an instance of the Q Network class.
	# (2) Create a function that constructs a policy from the Q values predicted by the Q Network. 
	#		(a) Epsilon Greedy Policy.
	# 		(b) Greedy Policy. 
	# (3) Create a function to train the Q Network, by interacting with the environment.
	# (4) Create a function to test the Q Network's performance on the environment.
	# (5) Create a function for Experience Replay.
	
	def __init__(self, environment_name, render=False):

		# Create an instance of the network itself, as well as the memory. 
		# Here is also a good place to set environmental parameters,
		# as well as training parameters - number of episodes / iterations, etc. 
		
		# env = QNetwork(environment_name)
		self.env = gym.make(environment_name)
		self.state_size = self.env.observation_space.shape[0]
		self.action_size = self.env.action_space.n

		self.replay_memory_size=50000
		self.memory = deque(maxlen=self.replay_memory_size)
		self.batch_size = 32
		self.burn_in = 10000

		if environment_name == 'MountainCar-v0':
			self.gamma = 1
			self.episodes = 3000 #given in handout
			self.iterations = 100000 #check how many, stops at 200
			self.terminate = 0
			self.environment_num = 1 
		elif environment_name == 'CartPole-v0':
			self.gamma = 0.99
			self.iterations = 1000000 #given in handout
			self.episodes = 3000 #check how many
			self.terminate = 1
			self.environment_num = 2

		self.train_epsilon_start = 1
		self.train_epsilon_stop = 0.01
		self.decay_rate = 0.0001  
		self.train_evaluate_epsilon = 0.05
		self.final_update = 0
		self.last_iter = 120000
		# episodes = 100

		self.q_network = QNetwork(environment_name)

	def epsilon_greedy_policy(self, q_values, train_test_var):
		# Creating epsilon greedy probabilities to sample from.             
		rand = np.random.uniform(0,1)
		#Different Episolns for fitting model and evaluating the training
		if train_test_var:
			if(rand<=self.train_epsilon):
				return np.random.randint(0,self.action_size)
			else:
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

		save_dir = os.path.join(os.getcwd(), 'saved_models_2_lqn_replay')
		if not os.path.isdir(save_dir):
			os.makedirs(save_dir)
		os.chdir(save_dir)

		total_updates = 0
		success_count = 0

		#Burn_in 

		state = self.env.reset()
		state = np.reshape(state,[1,self.state_size])

		for i in range(self.burn_in):	
			total_reward = 0	
			action = self.env.action_space.sample()
			next_state, reward, done, info = self.env.step(action)
			next_state = np.reshape(next_state,[1,self.state_size])	
			self.memory.append((state, action, reward, next_state, done))
			total_reward+=reward	
			if done:
				print("Episode {} total reward {}".format(i,total_reward))
				state = self.env.reset()
				state = np.reshape(state,[1,self.state_size])
			else:
				state=next_state

		train_rewards = []
		episodes = []	

		plt.ion()
		plt.title('Training Progress')
		plt.xlabel('Episodes')
		plt.ylabel('Total Reward')	
		plt.grid()		

		
		for i_episode in range(self.episodes):

			if self.environment_num==2 and total_updates > 150000:
				print("Saving final model at",total_updates)
				model_name = 'lqn_%d_%d_model_final.h5' %(self.environment_num,total_updates) 
				filepath = os.path.join(save_dir, model_name)
				self.q_network.model.save(model_name)
				self.final_update = total_updates
				plt.savefig("Training Progress.png")
				break

			state = self.env.reset()
			state = np.reshape(state,[1,self.state_size])	

			total_reward = 0		

			for t_iter in range(self.iterations):
				self.train_epsilon = self.train_epsilon_stop + (self.train_epsilon_start - self.train_epsilon_stop)*np.exp(-self.decay_rate*total_updates)
				action = self.epsilon_greedy_policy(self.q_network.model.predict(state),1) 

				next_state, reward, done, info = self.env.step(action)
				next_state = np.reshape(next_state,[1,self.state_size])	

				self.memory.append((state, action, reward, next_state, done))

				total_reward+=reward	

				state=next_state	
	
				inputs = np.zeros((self.batch_size, self.state_size))
				targets = np.zeros((self.batch_size, self.action_size))
				minibatch = random.sample(self.memory, self.batch_size)
				for i in range(self.batch_size):
					m_state, m_action, m_reward, m_next_state, m_done = minibatch[i]
					inputs[i:i+1] = m_state
					m_q_value_prime = m_reward
					if not m_done:
						m_q_value_prime = m_reward + self.gamma * np.amax(self.q_network.model.predict(m_next_state)[0])
						targets[i] = self.q_network.model.predict(m_state)
						targets[i][m_action] = m_q_value_prime
				self.q_network.model.fit(inputs,targets,epochs=1, verbose=0)
				total_updates+=1					

				if done:
					if (self.terminate==0 and total_reward>-200) or (self.terminate==1 and total_reward>=199):
						success_count+=1
					print("Episode {} total reward {} epsilon {} total_updates {}"
																.format(i_episode,total_reward,self.train_epsilon, total_updates))	
					episodes.append(i_episode)
					train_rewards.append(total_reward)
					plt.plot(episodes,train_rewards)
					plt.grid()
					plt.pause(0.001)					
					break		
				
				if (total_updates) % 10000 == 0:
					model_name = 'lqn_%d_%d_model.h5' %(self.environment_num,total_updates) 
					filepath = os.path.join(save_dir, model_name)
					self.q_network.model.save(model_name)	
					self.last_iter = total_updates	


		


	def test(self, model_file=None):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory. 
		num_episodes = 20
		model_num = []
		avg_reward = []
		load_dir = os.path.join(os.getcwd(), 'saved_models_dqn_replay')
		e = 0
		i = 10000
		self.env = wrappers.Monitor(self.env, directory='monitors/final/'+str(self.environment_num),video_callable=lambda episode_id: i % 3 == 0 and e == 0,force=True)

		plt.ion()
		plt.title('Test Models')
		plt.xlabel('Iterations')
		plt.ylabel('Average Reward')	
		plt.grid()

		for i in range(10000,self.last_iter + 10000, 10000):
			model_name = 'lqn_%d_%d_model.h5' %(self.environment_num,i)
			filepath = os.path.join(load_dir, model_name)
			test_model = load_model(filepath)			
			model_num.append(i)
			total_epi_rewards = []
			for e in range(num_episodes):
				total_epi_reward = 0
				state = self.env.reset()
				state = np.reshape(state,[1,self.state_size])	

				for t_iter in range(self.iterations):

					action = self.epsilon_greedy_policy(self.q_network.model.predict(state),0)				

					self.env.render()
					next_state, reward, done, info = self.env.step(action)
					next_state = np.reshape(state,[1,self.state_size])	
					total_epi_reward+=reward
			

					if done:						
						print("Episode {} finished after {} iterations with episodic reward %d".format(e+1,t_iter+1,total_epi_reward)) 
						total_epi_rewards.append(total_epi_reward)
						break			
					state = next_state
			avg_reward.append(np.mean(total_epi_rewards))
			plt.plot(model_num,avg_reward)
			plt.grid()
			plt.pause(0.001)

#Next objective in assignment handout
		epi = 100
		load_dir = os.path.join(os.getcwd(), 'saved_models_dqn_replay')
		model_name = 'lqn_%d_%d_model_final.h5' %(self.environment_num,self.final_update)
		filepath = os.path.join(load_dir, model_name)
		final_test_model = load_model(filepath)
		total_epi_rewards = []
		for e in range(epi):
			total_epi_reward = 0
			state = self.env.reset()
			state = np.reshape(state,[1,self.state_size])
			for t_iter in range(self.iterations):

				action = self.epsilon_greedy_policy(self.q_network.model.predict(state),0)
				next_state, reward, done, info = self.env.step(action)
				next_state = np.reshape(state,[1,self.state_size])	
				total_epi_reward+=reward	

				if done:						
					print("Episode {} finished after {} iterations with episodic reward %d".format(e+1,t_iter+1,total_epi_reward)) 
					total_epi_rewards.append(total_epi_reward)
					break			
				state = next_state

		print("Mean reward {} Standard deviation {}".format(np.mean(total_epi_rewards), np.std(total_epi_rewards)))


	

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


	# Setting the session to allow growth, so it doesn't allocate all GPU memory. 
	gpu_ops = tf.GPUOptions(allow_growth=True)
	config = tf.ConfigProto(gpu_options=gpu_ops)
	sess = tf.Session(config=config)

	dqn_agen.train()
	# dqn_agen.test()

	# Setting this as the default tensorflow session. 
	keras.backend.tensorflow_backend.set_session(sess)


if __name__ == '__main__':
	main(sys.argv)






	