#standard Q-learning agent
#(Not very bright, but useful for baseline analysis.)

#STATE REPRESENTATION: elementary hash of game text
#acquired via 'look' commands

#ACTION SPACE: a set of permissible verbs and
#objects as read from user-defined input files

#Q-VALUES: learns joint Q-values representing all possible
#combinations of verbs and objects for a given state.

#EXPLORATION: Epsilon-greedy

import agentBaseClass
import random as rand
import numpy as np
import re


class BaselineQlearner(agentBaseClass.AgentBaseClass):

	def read_verbs(self, filename = "Zork_verbs.txt"):
		file = open(filename)
		self.manipulation_list=[]
		for line in file:
			self.manipulation_list.append(line.rstrip('\n').rstrip('\r'))
		file.close()

	def read_objects(self, filename = "Zork_objects.txt"):
		file = open(filename)
		self.object_list=[]
		for line in file:
			self.object_list.append(line.rstrip('\n').rstrip('\r'))
		file.close()

	def __init__(self, initial_epsilon, training_cycles):
		self.ALPHA = 0.1
		self.GAMMA = 0.95
		self.EPSILON = initial_epsilon
		self.use_intrinsic_rewards = True
		self.epsilon_decay_rate = 0.9
		self.decay_interval = training_cycles/20
		self.num_states = 5000
		self.read_verbs()
		self.read_objects()
		self.navigation_list = ['north','south','east','west','northeast','northwest','southeast','southwest' 'up', 'down', 'enter', 'exit', 'get', 'drop', '']
#		self.manipulation_list = ['open', ''] #DEBUG
#		self.navigation_list = ['north', 'south', 'east', 'west', 'climb', 'down', 'enter', 'exit']  #DEBUG
#		self.object_list = ['egg', 'window', 'large tree', ''] #DEBUG
		self.verb_list = self.manipulation_list + self.navigation_list
		self.Qvalues = np.zeros(((self.num_states, len(self.verb_list), len(self.object_list))))
		self.Qvalues = self.Qvalues + 0.1
		self.refresh()
		

	def refresh(self):
		agentBaseClass.AgentBaseClass.refresh(self)
		self.look_flag = 0
		self.last_state = ''
		self.current_state = ''
		self.visited_states = ['']
		self.last_verb = ''
		self.last_object = ''
		self.last_reward = 0

	def is_novel(self, game_text):
		if game_text not in self.visited_states:
			self.visited_states.append(game_text)
			return True
		else:
			return False

	def bag_of_words(self, game_text):
		#uses a hash function to convert game text
		#into a bag-of-words index
		h = abs(hash(game_text))%self.num_states
		#h = abs(hash(game_text + str(self.total_points_earned)))%self.num_states
		return h

	def get_state_index(self, game_text):
		return self.bag_of_words(game_text)

	def random_action(self, game_text):
		return rand.choice(self.verb_list) + " " + rand.choice(self.object_list)

	def optimal_action(self, game_text):
#		f = open("salience_log5.txt", 'a')
#		f.write("OPTIMIZING!\n")
		s = self.get_state_index(game_text)
		Qvals = self.Qvalues[s]
#		f.write(game_text + "\n")
#		for q in Qvals:
#			f.write(str(q) + ' ')
#		f.write('\n')
		indices = np.where(Qvals == np.amax(Qvals))
		r = rand.choice(range(len(indices[0])))
		optimal_verb_index = indices[0][r]
		optimal_object_index = indices[1][r]
#		f.write('optimal verb index is ' + str(optimal_verb_index) + '\n')
#		f.write('optimal object index is ' + str(optimal_object_index) + '\n')
#		f.write(self.verb_list[optimal_verb_index] + " " + self.object_list[optimal_object_index] + '\n')
#		f.close()
		
		#IMPORTANT: if you overwrite this function be sure you
		#save the last_verb and lasT_object variables.
		#otherwise Q-values won't update correctly
		self.last_verb = self.verb_list[optimal_verb_index]
		self.last_object = self.object_list[optimal_object_index]
		return self.last_verb + " " + self.last_object
			

	def take_action(self, game_text, evaluation_flag = False):
		if evaluation_flag == True:
			 return optimal_action(self, game_text).strip()
		if self.look_flag == 0:
			self.look_flag = 1
			return "look"
		else:
			self.look_flag = 0
			#self.last_state = self.current_state #this update is made in the update function
			self.current_state = game_text
			movement_type = rand.random()
			if movement_type < self.EPSILON:
				return self.explore(game_text).strip()
			else:
				return self.optimal_action(game_text).strip()	

	def update_Qvalues(self, state_index, verb_index, object_index, reward, next_state):
#		print("UPDATING")
#		print(str(state_index) + ', ' + str(verb_index) + ", " + str(object_index) + ', ' +  str(reward) + ', ' + str(next_state))
#		for q in self.Qvalues[state_index]:
#			print q
#		print('\n')
		self.Qvalues[state_index][verb_index][object_index] += self.ALPHA * (reward + self.GAMMA*np.amax(self.Qvalues[next_state]) - self.Qvalues[state_index][verb_index][object_index])
#		for q in self.Qvalues[state_index]:
#			print q
#		print('\n\n')

	def update(self, state, last_action, reward, new_state):
		self.total_points_earned += reward
		if last_action == "look":
			state_index = self.get_state_index(self.last_state)
			next_state = self.get_state_index(new_state)
			vrb = self.verb_list.index(self.last_verb)
			obj = self.object_list.index(self.last_object)
			intrinsic_reward = 0
			if self.is_novel(new_state):
				intrinsic_reward = 1
			if self.use_intrinsic_rewards == True:
				#self.total_points_earned += intrinsic_reward
				self.total_intrinsic_reward += intrinsic_reward
				combined_reward = self.last_reward + intrinsic_reward
			else:
				 combined_reward = self.last_reward
			self.update_Qvalues(state_index, vrb, obj, combined_reward, next_state)

		else:
			self.last_state = state #remember the state that led to reward
			self.last_reward = reward
			return #don't update until the "look" command executes
	
	def decay_epsilon(self):
		self.EPSILON *= self.epsilon_decay_rate

	def get_learning_parameters(self):
		return_val = "ALPHA: " + str(self.ALPHA) + "\n"
		return_val += "GAMMA: " + str(self.GAMMA) + "\n"
		return_val += "EPSILON: " + str(self.EPSILON) + "\n"
		return_val += "EPSILON DECAY RATE: " + str(self.epsilon_decay_rate) + "\n"
		return_val += "Using intrinsic reward: " + str(self.use_intrinsic_rewards) + "\n" 	
		return_val += str(len(self.verb_list)) + " static verbs, " + str(len(self.object_list)) + " static objects.\n"
		return return_val
	

