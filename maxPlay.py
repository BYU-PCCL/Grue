import sys, os, time
import textplayer.textPlayer as tp
#import agents.agentWord2Vec as ac
#import agents.baselineQlearner as b
#import agents.salienceQlearner as s
#import agents.probabilisticQlearner as pr
#import agents.affordanceAgent as af
#import agents.wordFinder as wf
#import agents.objectSalience as os
#import agents.factoredQlearner as fq
#import agents.commonWordsQlearner as cw
#import agents.reductionLearner as rl
#import agents.cooccuranceQlearner as cq
import numpy as np
import matplotlib .pyplot as p
import time
import agents.maxLearner as ml

from multiprocessing import Process, Lock


# Describes an agents life span
def agent_action_loop(output_lock, a, t):

	counter = 0
	last_score = 0

	# A game is started
	current_game_text = t.run()

	# While training continues...
	while (counter < training_steps):

		# Get the current command from the agent, given the current game text
		current_command = a.take_action(current_game_text, False)

		# New game text is acquired after executing the command
		last_game_text = current_game_text
		current_game_text = t.execute_command(current_command)

		#print_output(output_lock, str(a) + ' ' + current_command + '\n' + current_game_text)
		#print_output(output_lock, '\n\n' + current_game_text + '\n' + current_command)

		# The agent is rewarded
		if t.get_score() != None:
			score, possible_score = t.get_score()
			reward = score - last_score
			last_score = score
		else:
			reward = 0

		#raw_input("")

		#print_output(output_lock, "UPDATING")
		#print_output(output_lock, a.Qvalues[a.get_state_index(a.last_state)])
		a.update(last_game_text, current_command, reward, current_game_text)
		#print_output(output_lock, a.Qvalues[a.get_state_index(a.last_state)])

		#print_output(output_lock, "TOTAL REWARD: " + str(a.get_total_points_earned()))
		counter += 1

# Print standard output using a lock
def print_output(lock, text):
	lock.acquire()
	try:
		print(text)
	finally:
		lock.release()

# A lock is created for managing output
output_lock = Lock()

number_agents = 1
current_game_file = 'zork1.z5'
if len(sys.argv) > 1:
	current_game_file = sys.argv[1]
	
base_filename = "data/junk/"
if len(sys.argv) > 3:
	base_filename = sys.argv[3]

if len(sys.argv) > 4:
	timestamp = sys.argv[4]
else:
	timestamp = time.time()

# Agents are created and assigned a process
for x in xrange(number_agents):
	initial_epsilon = 1 #agent starts completely random
	epochs = 1000
	#training_steps = 500
	training_steps = 1000

	print "Game file is " + current_game_file
	print "saving results to directory \'" + base_filename + "\'"

	a = None
	if sys.argv[2] == "1":
		print "Agent type is 1: no reduction, no intrinsic reward"
		a = ml.maxLearner(initial_epsilon, epochs)
		a.exploration_style = "RANDOM"
		a.filtering_objects = False
		a.use_intrinsic_rewards = False
		suffix = ".agent1"
	elif sys.argv[2] == "2":
		print "Agent type is 2: verb space reduction, no intrinsic reward"
		a = ml.maxLearner(initial_epsilon, epochs)
		a.exploration_style = "ANALOGY_REDUCTION"
		a.filtering_objects = False
		a.use_intrinsic_rewards = False
		suffix = ".agent2"
	elif sys.argv[2] == "3":
		print "Agent type is 3: object space reduction, no intrinsic reward"
		a = ml.maxLearner(initial_epsilon, epochs)
		a.exploration_style = "RANDOM"
		a.filtering_objects = True
		a.use_intrinsic_rewards = False
		suffix = ".agent3"
	elif sys.argv[2] == "4":
		print "Agent type is 4: verb AND object space reduction, no intrinsic reward"
		a = ml.maxLearner(initial_epsilon, epochs)
		a.exploration_style = "ANALOGY_REDUCTION"
		a.filtering_objects = True
		a.use_intrinsic_rewards = False
		suffix = ".agent4"
	elif sys.argv[2] == "5":
		print "Agent type is 5: no reduction, using intrinsic reward"
		a = ml.maxLearner(initial_epsilon, epochs)
		a.exploration_style = "RANDOM"
		a.filtering_objects = False
		a.use_intrinsic_rewards = True
		suffix = ".agent5"
	elif sys.argv[2] == "6":
		print "Agent type is 6: verb-space reduction, using intrinsic reward"
		a = ml.maxLearner(initial_epsilon, epochs)
		a.exploration_style = "ANALOGY_REDUCTION"
		a.filtering_objects = False
		a.use_intrinsic_rewards = True
		suffix = ".agent6"
	elif sys.argv[2] == "7":
		print "Agent type is 7: object-space reduction, using intrinsic reward"
		a = ml.maxLearner(initial_epsilon, epochs)
		a.exploration_style = "RANDOM"
		a.filtering_objects = True
		a.use_intrinsic_rewards = True
		suffix = ".agent7"
	elif sys.argv[2] == "8":
		print "Agent type is 8: verb AND object reduction, using intrinsic reward"
		a = ml.maxLearner(initial_epsilon, epochs)
		a.exploration_style = "ANALOGY_REDUCTION"
		a.filtering_objects = True
		a.use_intrinsic_rewards = True
		suffix = ".agent8"
	else:
		sys.exit("ERROR: Invalid agent type")

	#initial values
	ctr = 0
	x = []
	total_reward = []
	num_visited_states = []

	# game is initialized
	f = open(base_filename + "learning_params" + str(timestamp) + suffix, 'w')
	f.write("EPOCHS = " + str(epochs) + "\n")
	f.write("Current game is " + current_game_file)
	f.write("training steps = " + str(training_steps) + "\n")
	f.write("initial epsilon = " + str(initial_epsilon) + "\n")
	f.write(a.get_learning_parameters() + "\n")
	f.write(str(a.verb_list))
	f.close()

	# The game resets for each epoch	
	while (ctr < epochs):
		a.refresh()
		t = tp.TextPlayer(current_game_file)
		agent_action_loop(output_lock, a, t)
		t.quit()
		ctr = ctr + 1
		print("\n\n" + str(ctr) + " EPOCH DONE!\n")
		print("TOTAL REWARD: " + str(a.get_total_points_earned()) + "\n")
		print("Current game is " + current_game_file)
		print("Max QValue is " + str(np.max(a.Qvalues)))
		print("Average QValue is " + str(np.mean(a.Qvalues)))
		print("EPSILON IS " + str(a.EPSILON))
		#for s in a.visited_states:
		#	print( "[" + str(a.get_state_index(s)) + "] " + s + "\n")
		#	print( "Max Q-value is: " + str(np.amax(a.Qvalues[a.get_state_index(s)])) + '\n' )
		#	print(a.Qvalues_for_state(s))
		#	print("Optimal action is: " + a.optimal_action(s))

		f = open(base_filename + "learning_params" + str(timestamp) + suffix, 'a')
		f.write( "\n\n" + str(ctr) + " EPOCH DONE!\n" )
		f.write("Current game is " + current_game_file + '\n')
		f.write( "TOTAL REWARD: " + str(a.get_total_points_earned()) + "\n")
		f.write( "EPSILON is: " + str(a.EPSILON) + "\n" )
		#f.write( "Max Q-value is: " + str(np.amax(a.Qvalues)) + "\n" )
		f.write( "\nVISITED STATES:\n" )
		for s in a.visited_states:
			f.write( "[" + str(a.get_state_index(s)) + "] " + s + "\n")
			f.write( "Max Q-value is: " + str(np.amax(a.Qvalues[a.get_state_index(s.strip())])) + '\n' )
			f.write(a.Qvalues_for_state(s))
			f.write("Optimal action is: " + a.optimal_action(s.strip()))
			f.write( "\n" )
		f.close()

		f = open(base_filename + "rewards" + str(timestamp) + suffix, 'a')
		f.write(str(a.get_total_points_earned()) + "\n")
		f.close()
		#f = open(base_filename + "states_visited" + str(timestamp) + suffix, 'a')
		#f.write(str(len(a.visited_states)) + "\n")
		#f.close()
		#f = open(base_filename + "intrinsic_reward" + str(timestamp) + suffix, 'a')
		#f.write(str(a.get_intrinsic_reward()) + "\n")
		#f.close()
		a.write_epoch_data(base_filename, timestamp, suffix)
		#a.write_epoch_data(base_filename, timestamp)
		x.append(ctr-1)
		total_reward.append(a.get_total_points_earned())
		num_visited_states.append(len(a.visited_states))
		if ctr%a.decay_interval == 0:
			a.decay_epsilon()
		print("reward was: " + str(a.get_total_points_earned()) + "\n")
		print("visited states: ")
		for s in a.visited_states:
			#print("     " + s[:50] + " maxQ=" + str(np.max(a.Qvalues[a.get_state_index(s)])) )
			print("     " + s[:50] + " maxQ=" + str(np.max(a.Qvalues[a.get_state_index(s)])) + " " + str(a.optimal_action(s)))
			
	print("\nTRAINING COMPLETE!")
	print "Max Q-value is: " + str(np.amax(a.Qvalues))

	print("plotting reward curve for timestamp " + str(timestamp) + ".")
	#p.plot(x, total_reward)
	#p.savefig(base_filename + "graph_reward" + str(timestamp) + suffix + ".png")
	#p.show()

	print("plotting number of visited states for timestamp " + str(timestamp) + ".")
	#p.clf()
	p.plot(x, num_visited_states)
	p.savefig(base_filename + "graph_num_states" + str(timestamp) + suffix + ".png")

	print( "DATA FROM LAST EPOCH:\n")
	print( "TOTAL REWARD: " + str(a.get_total_points_earned()) )
	print( "Max Q-value is: " + str(np.amax(a.Qvalues)) )
	print( "\n" )
	#print( "VISITED STATES:" )
	#for s in a.visited_states:
	#	print( "[" + str(a.get_state_index(s)) + "] " + s + "\n")
	#	#print( a.Qvalues[a.get_state_index(s)] )
	#	print( "\n" )
	#	print a.Qvalues_for_state(s)
	

		# Each agent gets its own background process
		#Process(target=agent_action_loop, args=(output_lock, a, t)).start()

	print ("BEGGINING OPTIMAL RUN:")
	a.refresh()
	a.epsilon = 0 #agent will now perform optimal actions
	a.verbose = True #blabber aboutwhat you're doing on every time step
	t = tp.TextPlayer(current_game_file)
	agent_action_loop(output_lock, a, t)
	t.quit()
