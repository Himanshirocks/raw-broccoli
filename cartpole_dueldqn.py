#!/usr/bin/env python
import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse
from keras.models import Sequential,load_model,Model
from keras.layers import Input,Dense,Activation, merge, Lambda,Add
from keras.optimizers import Adam
from keras import backend as K
from keras import initializers
import os
from collections import deque
import random
import matplotlib.pyplot as plt
from gym import wrappers
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


class QNetwork(): 

	def __init__(self, environment_name):

		self.env = gym.make(environment_name)
		self.alpha = 0.005 #0.001 or 0.0001

		self.model = Sequential()
		# self.state_size = self.env.observation_space.shape[0]
		# self.action_size = self.env.action_space.n

		# input = Input(shape = (self.state_size,))
		# x = Dense(10, input_shape = (self.state_size,), activation = 'relu', kernel_initializer = 'he_uniform')(input)
		# x = Dense(10, activation = 'relu', kernel_initializer = 'he_uniform')(x)

		# state_value = Dense(1, kernel_initializer = 'he_uniform')(x)
		# s = Lambda(lambda s: k.expand_dims(s[:, 0], -1), output_shape=(self.action_size,))(state_value)

		# action_advantage = Dense(self.action_size, kernel_initializer = 'he_uniform')(x)
		# action_advantage = Lambda(lambda a: a[:, :] - k.mean(a[:, :], keepdims=True),output_shape=(self.action_size,))(action_advantage)

		# q_value = merge([ s , action_advantage], mode = 'sum')

		# self.model = Model(input = input, output= q_value)
		# self.model.compile(loss = 'mse', optimizer = Adam(self.alpha))
		# print(self.model.summary())

		input_layer = Input(shape = self.env.observation_space.shape)
		hidden_layer1 = Dense(10, activation='relu')(input_layer)
		hidden_layer2 = Dense(10, activation='relu')(hidden_layer1)
		hidden_layer3 = Dense(10, activation='relu')(hidden_layer2)	
		
		value_layer = Dense(1)(hidden_layer3)
		advantage_layer = Dense(self.env.action_space.n)(hidden_layer3)

		value_layer2 = Lambda(lambda s: K.expand_dims(s[:, 0], axis=-1),output_shape = (self.env.action_space.n,))(value_layer)
		advantage_layer2 = Lambda(lambda a: a[:, :] - K.max(a[:, :], keepdims=True),output_shape = (self.env.action_space.n,))(advantage_layer)
		
		q_value_layer = Add()([value_layer2, advantage_layer2])

		self.model = Model(input=[input_layer], output=[q_value_layer])		

		self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha))

		print(self.model.summary())

		
        

class DQN_Agent():
	
	def __init__(self, environment_name, render=False):

		self.env = gym.make(environment_name)
		self.state_size = self.env.observation_space.shape[0]
		self.action_size = self.env.action_space.n

		#experience replay 
		self.replay_memory_size=50000
		self.memory = deque(maxlen=self.replay_memory_size)
		self.batch_size = 32
		# self.burn_in = 10000
		self.burn_in = 10000

		if environment_name == 'MountainCar-v0':
			self.gamma = 1
			self.episodes = 3000 #given in handout
			self.iterations = 201 #check how many, stops at 200
			self.terminate = 0
			self.environment_num = 1 
		elif environment_name == 'CartPole-v0':
			self.gamma = 0.99
			self.iterations = 200 #given in handout
			self.episodes = 5000 #check how many 
			self.terminate = 1
			self.environment_num = 2

		self.train_epsilon_start = 1
		self.train_epsilon_stop = 0.01
		# self.train_epsilon = self.train_epsilon_start

		self.decay_rate = 0.0001 #0.0001 for cartpole  
		self.train_evaluate_epsilon = 0.05

		self.last_iter = 0
		
		self.final_update = 10000
		self.q_network = QNetwork(environment_name)


	def epsilon_greedy_policy(self, q_values,train_test_var):
		rand = np.random.uniform(0,1)

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
		return np.argmax(q_values) 



	def train(self):

		file_name = 'models_for_CP_DuelDQN_eps_%f_to_%f_with_%f_decay' %(self.train_epsilon_start,self.train_epsilon_stop,self.decay_rate) 
		save_dir = os.path.join(os.getcwd(),file_name)
		if not os.path.isdir(save_dir):
			os.makedirs(save_dir)
		os.chdir(save_dir)

		total_updates = 0
		success_count = 0

		# starting_epsilon = self.train_epsilon

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

		plt.ion()
		plt.title('Training Progress')
		plt.xlabel('Episodes')
		plt.ylabel('Total Reward')	
		# plt.grid()

		train_rewards = []
		episodes = []	

		
		for i_episode in range(self.episodes):

			if self.environment_num==2 and total_updates > 1000000:
				break

			state = self.env.reset()
			state = np.reshape(state,[1,self.state_size])	

			total_reward = 0
			ep_terminate = False	

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
					
				# w = self.q_network.model.get_weights()
				if (total_updates) % 10000 == 0:
					model_name = 'dqn_%d_model.h5' %(total_updates) 
					filepath = os.path.join(save_dir, model_name)
					self.q_network.model.save(model_name)
					self.last_iter = total_updates	
					
				if done:
					if (self.terminate==0 and total_reward>-200) or (self.terminate==1 and total_reward>=199):
						success_count+=1
						if success_count >=100:
							ep_terminate = True
							break						
					else:
						success_count = 0

					print("Episode {} epsilon {} repeat rate {} Score {}".format(i_episode,self.train_epsilon,success_count,t_iter + 1))	
					episodes.append(i_episode)
					train_rewards.append(total_reward)
					plt.plot(episodes,train_rewards,'b')
					# plt.grid()
					plt.pause(0.001)
					break			

		print("Saving final model at",total_updates)
		model_name = 'dqn_%d_model_final.h5' %(total_updates) 
		filepath = os.path.join(save_dir, model_name)
		self.q_network.model.save(model_name)
		print("Total success is ",success_count)
		plt.savefig('graph_cp_duel.png' %(i),bbox_inches='tight' )


		# If you are using a replay memory, you should interact with environment here, and store these 
		# transitions to memory, while also updating your model.

	def test(self, model_file=None):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory. 
		num_episodes = 20
		model_num = []
		avg_reward = []
		# load_dir = os.path.join(os.getcwd(), 'saved_models_dqn_replay')
		# directory_name = 'models_for_CP_DuelDQN_eps_%f_to_%f_with_%f_decay1' %(self.train_epsilon_start,self.train_epsilon_stop,self.decay_rate)
		load_dir = os.path.join(os.getcwd(), 'models_for_CP_DuelDQN_eps_1.000000_to_0.010000_with_0.000100_decay')
		e = 0
		i = 10000
		count = 0
		self.env = wrappers.Monitor(self.env, directory='monitors/duel_final/'+str(self.environment_num),video_callable=lambda episode_id: i % 3 == 0 and e == 0,force=True)

		plt.ion()
		plt.title('Test Models')
		plt.xlabel('Iterations')
		plt.ylabel('Average Reward')	
		plt.grid()

		for i in range(10000, 100000 + 10000, 10000):
			# model_name = 'lqn_%d_%d_model.h5' %(self.environment_num,i)
			model_name = 'dqn_%d_model.h5' %(i)
			filepath = os.path.join(load_dir, model_name)
			self.q_network.model = load_model(filepath)			
			model_num.append(i)
			total_epi_rewards = []
			print("Model Number: {} ".format(i))
			for e in range(num_episodes):
				total_epi_reward = 0
				state = self.env.reset()
				state = np.reshape(state,[1,self.state_size])	
				
				for t_iter in range(self.iterations):

					action = self.epsilon_greedy_policy(self.q_network.model.predict(state),0)				

					self.env.render()
					next_state, reward, done, info = self.env.step(action)
					next_state = np.reshape(next_state,[1,self.state_size])	
					total_epi_reward+=reward
			

					if done:						
						print("Episode {} finished after {} iterations with episodic reward {}".format(e+1,t_iter+1,total_epi_reward)) 
						total_epi_rewards.append(total_epi_reward)
						break			
					state = next_state
			avg_reward.append(np.mean(total_epi_rewards))
			plt.plot(model_num,avg_reward)
			plt.grid()
			plt.pause(0.001)
			if avg_reward[(i//10000) - 1] >= 195:
				count = count + 1
			elif avg_reward[(i//10000) - 1] >= 195 and count == 2:
				break

		plt.savefig("Average_Rewards.png")

		#Next objective in assignment handout
		epi = 100
		# load_dir = os.path.join(os.getcwd(), 'saved_models_dqn_replay')
		# model_name = 'lqn_%d_%d_model_final.h5' %(self.environment_num,self.final_update)
		# filepath = os.path.join(load_dir, model_name)
		# final_test_model = load_model(filepath)
		total_epi_rewards = []
		for e in range(epi):
			total_epi_reward = 0
			state = self.env.reset()
			state = np.reshape(state,[1,self.state_size])
			for t_iter in range(self.iterations):

				action = self.epsilon_greedy_policy(self.q_network.model.predict(state),0)
				next_state, reward, done, info = self.env.step(action)
				next_state = np.reshape(next_state,[1,self.state_size])	
				total_epi_reward+=reward	

				if done:						
					print("Episode {} finished after {} iterations with episodic reward {}".format(e+1,t_iter+1,total_epi_reward)) 
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

	# Setting the session to allow growth, so it doesn't allocate all GPU memory. 
	gpu_ops = tf.GPUOptions(allow_growth=True)
	config = tf.ConfigProto(gpu_options=gpu_ops)
	sess = tf.Session(config=config)

	# Setting this as the default tensorflow session. 
	keras.backend.tensorflow_backend.set_session(sess)

	environment_name = args.env
	dqn_agen = DQN_Agent(environment_name)

	# dqn_agen.train()
	dqn_agen.test()

	# You want to create an instance of the DQN_Agent class here, and then train / test it. 

if __name__ == '__main__':
	main(sys.argv)