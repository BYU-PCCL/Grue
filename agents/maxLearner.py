import random as rand
import numpy as np
import re
import slim_scholar.slim_scholar as sch
import nltk
#import verbFinder
import sys

class maxLearner():

	#def read_verbs(self, filename):
	#	file = open(filename)
	#	verb_list=[]
	#	for line in file:
	#		verb_list.append(line.rstrip('\n').rstrip('\r'))
	#	file.close()
	#	return verb_list

	#def read_objects(self, filename):
	#	file = open(filename)
	#	object_list=[]
	#	for line in file:
	#		object_list.append(line.rstrip('\n').rstrip('\r'))
	#	file.close()
	#	return object_list

	def __init__(self, initial_epsilon, training_epochs):
		print "*****INITIALIZING AGENT*****"	
	
		#standard initialization
		self.ALPHA = .1 #THIS VALUE IS NOT USED WHEN QVALUE UPDATE METHOD IS 'MAX'
		self.GAMMA = 0.95
		self.EPSILON = initial_epsilon
		self.epsilon_decay_rate = 0.85
		self.decay_interval = training_epochs/20
		self.num_states = 50000
		self.use_intrinsic_rewards = False
		self.use_inventory = False
		self.augmented_state_space = False
		self.look_at_objects = False
		self.qvalue_update_method = "MAX"
		#self.qvalue_update_method = "STANDARD"
		self.verbose = False
		
		#Verb- and Object-space reduction
		self.exploration_style = "RANDOM"
		#self.exploration_style = "RANDOM_REDUCTION"   #uses a randomly-selected subset
		#self.exploration_style = "SYNTAX_REDUCTION"   #uses verbFinder
		#self.exploration_style = "ANALOGY_REDUCTION"  #uses scholar
		self.NUM_VERBS = 1000  
		self.NUM_MATCHING_VERBS = 15
		self.TWO_WORD_OBJECTS = True
		self.filtering_objects = True
		self.NUM_MANIPULATABLE_OBJECTS = 15
		
		#populate member variables
		self.scholar = sch.Scholar()
		self.manipulation_list=self.scholar.get_most_common_words('VB', self.NUM_VERBS)
		self.manipulation_list.append('')
		#self.essential_manipulation_list = ['get', 'drop', 'inventory', 'examine']
		self.essential_manipulation_list = ['get', 'drop', 'push', 'pull']
		#self.navigation_list =['north', 'south', 'east', 'west','northeast','northwest','southeast','southwest','up', 'down', 'enter', 'exit']
		self.navigation_list =['north', 'south', 'east', 'west', 'up', 'down', 'enter', 'exit']
		self.verb_list = self.manipulation_list + self.navigation_list + self.essential_manipulation_list
		self.analogy_verbs = {} #caches nouns for each state, to save time
		#self.found_objects = {} #caches objects found for each state, to save time
		self.max_num_objects = 100 #number of object QValues to store		
		self.Qvalues = np.zeros(((self.num_states, len(self.verb_list), self.max_num_objects)))
		self.refresh()
		


	def refresh(self):
		print "REFRESHING AGENT"
		print "Max Qval is " + str(np.max(self.Qvalues))

		#reward tracking
		self.total_points_earned = 0
		self.total_intrinsic_reward = 0
		self.last_reward = 0
		
		#record of past events
		self.visited_states = ['']
		self.look_flag = 0
		self.last_state = ''
		self.last_verb = ''
		self.last_object = ''
		self.last_action = ''
		
		#current events
		self.current_state = ''
		self.inventory_text = ""
		
		#statistics and averages
		self.num_verbs = [] #measures verb-space reduction
		self.num_objects = [] #measures object-space reduction


	def is_novel(self, game_text):
		if game_text not in self.visited_states:
			self.visited_states.append(game_text)
			return True
		else:
			return False


	def bag_of_words(self, game_text):
		#uses a hash function to convert game text
		#into a bag-of-words index
		if self.augmented_state_space == True:
			h = abs(hash(game_text + str(self.total_points_earned)))%self.num_states
		else:
			h = abs(hash(game_text))%self.num_states
		return h

	def get_state_index(self, game_text):
		filtered_text = re.sub(r'\d+', '', game_text.strip()) #strip out score/num_moves/etc
		return self.bag_of_words(filtered_text)

	def decay_epsilon(self):
		self.EPSILON *= self.epsilon_decay_rate

	def find_objects(self, narrative):
		game_text = narrative

		#self.found_objects
		#if game_text in self.found_objects.keys():
			#count the number of nouns we actually used (for statistics)
			#self.num_objects.append(len(self.found_objects[game_text]))
			#return self.found_objects[game_text]
			
		#assumes an object is salient if it appears as a noun in the game text
		#('' is also considered a salient object)
		tokens = nltk.word_tokenize(game_text)
		tags = nltk.pos_tag(tokens)
		nouns = [word for word,pos in tags if word.isalnum() and (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')]

		#if we filter, do it before we find the two-word variants
		if len(nouns) > self.NUM_MANIPULATABLE_OBJECTS and self.filtering_objects == True:
			#get the w2v vectors for each object
			data = []
			for n in nouns:
				tagged_noun = n.lower() + "_NN"
				if self.scholar.exists_in_model(tagged_noun):
					data.append(self.scholar.model[tagged_noun])
				else:
					data.append(np.mean(self.scholar.model.vectors, axis=0))

			#flatten them according to 'forest' - 'tree' and 'building' - 'brick'
			vector= self.scholar.model['forest_NN'] - self.scholar.model['tree_NN']
			flattened_data = np.dot(vector, np.array(data).T).T

			#take the n words with the smallest value/l2 norm
			zipped = zip(flattened_data, nouns)
			zipped.sort()
			sorted_nouns = [nouns for (flattened_data, nouns) in zipped]
			nouns = []
			ctr = 0
			for i in range(len(sorted_nouns)):
				if sorted_nouns[i] not in nouns and ctr < self.NUM_MANIPULATABLE_OBJECTS:
					nouns.append(sorted_nouns[i])
					ctr += 1		


		if self.TWO_WORD_OBJECTS == True:
			tokens = nltk.word_tokenize(game_text)
			tags = nltk.pos_tag(tokens)
			for i in range(0, len(tags) - 1):
				if (tags[i][1] == "JJ") and (tags[i+1][1] in ["NN", "NNP", "NNS", "NNPS"]):
					nouns.append(tags[i][0] + " " + tags[i+1][0])

		#always include the empty noun (to allow verb-only commands)
		nouns.append('')
		
		#count the number of nouns we actually used (for statistics)
		self.num_objects.append(len(nouns))
	
		#self.found_objects[game_text] = nouns
		return nouns


	def optimal_action(self, current_state):
		s = self.get_state_index(current_state)
		object_list = self.find_objects(current_state)

		if self.verbose == True:
			print("Getting optimal action for state " + str(s))
			print(self.verb_list)
			print(object_list)
			print(self.Qvalues_for_state(current_state))
		
		Qvals = self.Qvalues[s]
		sliced_Qvals = Qvals[0:len(self.verb_list), 0:len(object_list)] #only use found objects
		if self.verbose == True:
			print ("SLICED QVALS:")
			print sliced_Qvals
			for i in range(len(self.verb_list)):
				print self.verb_list[i] + ": " + str(sliced_Qvals[i][:len(object_list)])
		
		indices = np.where(sliced_Qvals == np.amax(sliced_Qvals))
		r = rand.choice(range(len(indices[0])))
		optimal_verb_index = indices[0][r]
		optimal_object_index = indices[1][r]
		
		#IMPORTANT; if you overwrite this function be sure you
		#save the last_verb and last_object variables.
		#otherwise Q-values won't update correctly.
		self.last_verb = self.verb_list[optimal_verb_index]
		self.last_object = object_list[optimal_object_index]

		if self.verbose == True:
			print( "OPTIMAL action: " + self.last_verb + " " + self.last_object)
			#if np.max(self.Qvalues[s]) > 8:
			#	raw_input("pause")
		return self.last_verb + " " + self.last_object
	
	def get_analogy_verb(self, obj):
		#uses scholar's analogy function to find verbs that
		#'match' the given noun. 
		words = obj.split()
		if len(words) > 0:
			obj = words[-1]

		#if obj == '':
		#	vrb = np.random.choice(self.navigation_list)
		#	return vrb
		
		tryList = []

		if obj in self.analogy_verbs.keys():
			tryList = self.analogy_verbs[obj]
		else:
			matching_verbs = self.scholar.get_verbs(obj, self.NUM_MATCHING_VERBS)
			#print matching_verbs
			if len(matching_verbs) > 0:
				for i in range(len(matching_verbs)):
					#strip scholar's tag info from verb
					matching_verbs[i] = matching_verbs[i][:-3]

			for v in matching_verbs:
				if v in self.manipulation_list:
					tryList.append(v)

			self.analogy_verbs[obj] = tryList
		self.num_verbs.append(len(tryList))

		vrb = np.random.choice(tryList+self.navigation_list+self.essential_manipulation_list)
		return vrb


	def get_dependent_verb(self, obj):
		#uses verbFinder to calculate dependency counts for
		#each verb/object pair. Randomly selects a verb with
		#probability proportional to its dependency count
		#(Variations - scale count by verb frequency)
		#(anoter Variation - use weighted average of counts and Qvals)
		words = obj.split()
		if len(words) > 0:
			obj = words[-1]

		my_dist = []

		for v in self.verb_list:
			words = v.split()
			if len(words) > 0:
				vrb = words[0]
			else:
				vrb = v
			
			if obj == '' or vrb == '':
				my_dist.append(5.0)
			else:
				tagged_verb = vrb.lower() 
				tagged_object = obj.lower()
				try:
					my_dist.append(self.verbFinder.numDependencies(verb.lower(), obj.lower()))
				except:
					my_dist.append(0.0)
			
		if np.sum(my_dist) == 0:
			my_dist = np.ones(len(my_dist))
			my_dist = my_dist/np.sum(my_dist)
		else:
			#my_dist = np.absolute(my_dist) 
			my_dist = my_dist / np.sum(my_dist)
		
		vrb = np.random.choice(self.verb_list, 1, p=my_dist)[0]
		return vrb
		

	def get_action_based_on_analogy(self, game_text):
		object_list = self.find_objects(game_text)
		self.last_object = rand.choice(object_list)
		self.last_verb = self.get_analogy_verb(self.last_object)
		#print self.last_verb + ' ' + self.last_object
		return self.last_verb + " " + self.last_object
		
	def get_action_based_on_dependency_count(self, game_text):
		object_list = self.find_objects(game_text)
		self.last_object = rand.choice(object_list)
		self.last_verb = self.get_dependent_verb(self.last_object)
		return self.last_verb + " " + self.last_object
	
	def random_action(self, game_text):
		#print "Getting random action..."
		#IMPORTANT; if you overwrite this function be sure you
		#save the last_verb and last_object variables.
		#otherwise Q-values won't update correctly.
		self.num_verbs.append(len(self.verb_list))
		object_list = self.find_objects(game_text)
		self.last_verb = rand.choice(self.verb_list)
		self.last_object = rand.choice(object_list)
		if self.verbose==True:
			print( "RANDOM action: " +  self.last_verb + ' ' + self.last_object)
		return self.last_verb + " " + self.last_object

	def explore(self, game_text):
		if self.exploration_style == "RANDOM":
			return self.random_action(game_text)
		elif self.exploration_style == "RANDOM_REDUCTION":
			#takes a random action, but the verb list and object
			#list have already been reduced to a random subset
			return self.random_action(game_text) 
		elif self.exploration_style == "DEPENDENCY_REDUCTION":
			return self.get_action_based_on_dependency_count(game_text)
		elif self.exploration_style == "ANALOGY_REDUCTION":
			return self.get_action_based_on_analogy(game_text)
		else:
			sys.exit("commonWordsQlearner.explore -- unrecognized exploration style")		

	def take_action(self, game_text, evaluation_flag = False):
		if evaluation_flag == True:
			 return optimal_action(self, game_text).strip()
		if self.look_flag == 0:
			self.look_flag = 1
			if self.look_at_objects == True:
				r = rand.randint(0,1)
			else:
				r = 0
			if r == 0:
				return "look"
			else:
				look_obj = rand.choice(self.find_objects(game_text))
				return "look at " + look_obj
		else:
			self.look_flag = 0
			if self.use_inventory == True:
				self.current_state = game_text + self.inventory_text
			else:
				self.current_state = game_text
		
			#if self.verbose==True:
			#	print("\n\nGame text is " + str(self.get_state_index(game_text.strip())) + game_text[:30])
			#	print("Agent current state is " + str(self.get_state_index(self.current_state)) + self.current_state[:30])
			#	print("Best Qval in this state is " + str(np.max(self.Qvalues[self.get_state_index(self.current_state)])))
			#	print(str(np.argmax(self.Qvalues[self.get_state_index(self.current_state)])))
			movement_type = rand.random()
			if movement_type < self.EPSILON:
				return self.explore(self.current_state)
			else:
				return self.optimal_action(self.current_state)	

	def update_Qvalues(self, state_index, verb_index, object_index, reward, next_state):
		
		if self.qvalue_update_method == "MAX":
			self.Qvalues[state_index][verb_index][object_index] = max(self.Qvalues[state_index][verb_index][object_index], reward + self.GAMMA*np.amax(self.Qvalues[next_state]))
		
		elif self.qvalue_update_method == "STANDARD":
			self.Qvalues[state_index][verb_index][object_index] += self.ALPHA * (reward + self.GAMMA*np.amax(self.Qvalues[next_state]) - self.Qvalues[state_index][verb_index][object_index])		
		
		elif self.qvalue_update_method == "NONE":
			pass
		else:
			sys.exit("ERROR: INVALID VALUE FOR self.qvalue_update_method")	


	def update(self, state, action, reward, new_state):
		self.total_points_earned += reward
		if self.verbose == True:
			print "\nUPDATE: state=" + state[:30] + ", last_action=" + action + ", reward = " + str(reward)
			print "last_verb: " + self.last_verb + ", last_object" + self.last_object
			print "New state is [" +  str(self.get_state_index(new_state)) + "]: " + new_state[:30]  


		#update Qvalues only if the selected verb is in the verb list
		#(otherwise, for now, we just won't remember that we took that action.)
		if self.last_verb not in self.verb_list:
			print "ERROR: last_verb '" + self.last_verb + "' not in verb list! Aborting update"
			return #don't update
			#This bug probably got fixed when I fixed the verb_list bug

		if action == "inventory":
			self.inventory_text = new_state

		if action[:4] == "look" and self.last_state != '': #we looked and we didn't just barely refresh
			state_index = self.get_state_index(self.last_state)
			next_state = self.get_state_index(new_state.strip())
			object_list = self.find_objects(self.last_state)
			vrb = self.verb_list.index(self.last_verb)
			#obj = object_list.index(self.last_object)
			if self.last_object in object_list:
                                obj = object_list.index(self.last_object)
                        	errflag = 0
			else:
                                print "ERROR"
                                print "object not in list: " + self.last_object
				print object_list
				print "self.current_state is " + self.current_state
				print "self.last_state is " + self.last_state
                                print "inventory list is:"
                                print self.inventory_text
                                print "current state is " + new_state
                                print "last state is " + self.last_state
                                print "last action is " + self.last_action
                                print "last verb is " + self.last_verb
				errflag = 1


			intrinsic_reward = 0
			if self.is_novel(new_state):
				intrinsic_reward = 1
			if self.use_intrinsic_rewards == True:
				#self.total_points_earned += intrinsic_reward
				self.total_intrinsic_reward += intrinsic_reward
				combined_reward = self.last_reward + intrinsic_reward
			else:
				 combined_reward = self.last_reward
			if errflag == 0: #hack to work around an infrequent bug
				if self.verbose == True:
					print "\n\nABOUT TO UPDATE Qvals for state [" + str(state_index) + "] " + self.last_state[:30]
					#sliced_qvals = self.Qvalues[state_index][:len(self.verb_list)][:len(object_list)]
					#print sliced_qvals
					
					print self.Qvalues[state_index][:10,:10]
					print "Qvals for next state:"
					print self.Qvalues[next_state][:10,:10]
					print self.verb_list
					print object_list 
					print "last_object = " + self.last_object
					print "updated object = " + object_list[obj]
				self.update_Qvalues(state_index, vrb, obj, combined_reward, next_state)
				if self.verbose == True:
					print "post-update"
					#sliced_qvals =  self.Qvalues[state_index][:len(self.verb_list)][:len(object_list)]
					#print sliced_qvals
					print self.Qvalues[state_index][:10,:10]
					#if np.max(self.Qvalues[state_index]) > 8:
					#	raw_input("pause")

		else:
			if self.verbose == True:
				print "This is a non-look state. No Qvalue update."
			self.last_state = state.strip() #remember the state that led to reward
			#self.last_state = self.current_state #remember the state that led to reward
			self.last_reward = reward
			return #don't update until the "look" command executes

	def get_status(self):
		#generates a 'status update' which will be
		#printed to the screen by autoplay.py
		return 'TOTAL POINTS = ' + str(self.total_points_earned)

	def get_total_points_earned(self):
		#useful for writing data files that track 
		#obtained reward over time
		return self.total_points_earned

	def get_intrinsic_reward(self):
		#useful for writing data files that track 
		#obtained reward over time
		return self.total_intrinsic_reward

	def get_learning_parameters(self):
		return_val = "MAX LEARNER\n"
		return_val += "NUM_VERBS: " + str(self.NUM_VERBS) + "\n"
		return_val += "NUM_MATCHING_VERBS: " + str(self.NUM_MATCHING_VERBS) + "\n"
		return_val += "filtering objects? " + str(self.filtering_objects) + "\n"
		return_val += "NUM_MANIPULATABLE_OBJECTS: " + str(self.NUM_MANIPULATABLE_OBJECTS) + "\n"
		return_val += "Exploration Style: " + str(self.exploration_style) + "\n"
		return_val += "Two-word objects?" + str(self.TWO_WORD_OBJECTS) + "\n"		
		return_val += "ALPHA: " + str(self.ALPHA) + "\n"
		return_val += "GAMMA: " + str(self.GAMMA) + "\n"
		return_val += "EPSILON: " + str(self.EPSILON) + "\n"
		return_val += "EPSILON DECAY RATE: " + str(self.epsilon_decay_rate) + "\n"
		return_val += "Using intrinsic reward: " + str(self.use_intrinsic_rewards) + "\n" 	
		#return_val += str(len(self.verb_list)) + " static verbs, " + str(len(self.object_list)) + " static objects.\n"
		return_val += str(len(self.verb_list) ) + " static verbs,\n"
		return_val += "Using inventory: " + str(self.use_inventory) + "\n"
		return_val += "Augmented state space: " + str(self.augmented_state_space) + "\n"
		return_val += "Look at objects: " + str(self.look_at_objects) + "\n"
		return_val += "Qvalue update method: " + self.qvalue_update_method + "\n"
		
		return return_val

	def Qvalues_for_state(self, s):
		returnval = "\n"
		objects = self.find_objects(s)
		for o in objects:
			returnval += "'" + o + "', "
		returnval += '\n'
		for i in range(len(self.verb_list)):
			returnval += self.verb_list[i] + ' '
			returnval += str(self.Qvalues[self.get_state_index(s)][i][:len(objects)]) + '\n'
		return returnval

	def write_epoch_data(self, base_filename, timestamp, suffix=""):
		f = open(base_filename + "averages" + str(timestamp) + suffix, 'a')
		verb_sum = 0.0
		if len(self.num_verbs) > 0:
			for v in self.num_verbs:
				verb_sum += v
			f.write("Average number of verbs: " + str(verb_sum/len(self.num_verbs)) + '\n')	
		else:
			f.write("No verbs found!\n")			
		if len(self.num_objects) > 0:
			object_sum = 0.0
			for o in self.num_objects:
				object_sum += o
			f.write("Average number of objects: " + str(object_sum/len(self.num_objects)) + '\n')	
		else:
			f.write("No objects found!")
		f.write("Max Qval: " + str(np.max(self.Qvalues)) + "\n")
		f.write("Average Qval: " + str(np.average(self.Qvalues)) + "\n")
		f.close()


