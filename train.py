#!/usr/bin/env python

import numpy as np
import time
import sys
from env.env_ import environment
from agent.agent_ddqn import DQNAgent
from agent.agent_ddqn import EPISODES, EPISODE_LENGTH, BATCH_SIZE
import matplotlib.pyplot as plt
from collections import deque
import os.path

fds_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', \
			'h', 'i', 'j', 'k', 'l', 'm', 'n', \
			'o', 'p', 'q', 'r', 's', 't', 'u', \
			'v', 'w', 'x', 'y', 'z', \
			'A', 'B', 'C', 'D', 'E', 'F', 'G', \
			'H', 'I', 'J', 'K', 'L', 'M', 'N', \
			'O', 'P', 'Q', 'R', 'S', 'T', 'U', \
			'V', 'W', 'X', 'Y', 'Z', \
			'0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

fds_origin_loc = []

def load_character(character_path):
	if os.path.exists(character_path):
		with open(character_path, 'r') as f:
			for content in f.readlines():
				if -1 == content.find('{'):
					continue
				content = content.split('{')[1]
				content = content.split('}')[0]
				content = content.split(',')

				loc = []
				for n in range(len(content)):
					value = int(content[n], 16)
					for b in range(8):
						if (value & (1 << b)) :
							loc.append(environment.Loc(x = b + 1, y = n))
				fds_origin_loc.append([loc, 9, len(content) - 3])

def main(char):
	enable_plot = False

	if enable_plot:
	    fig, ax = plt.subplots()
	    fig.show()
	    fig.canvas.draw()
	    steps = deque(maxlen=200)
	    episodes = deque(maxlen=200)
	    rewards = deque(maxlen=200)

	character_path = 'samples/char.TXT'
	load_character(character_path)

	R = 0
	if char == None:
		char_index = 0
	else:
		char_index = fds_list.index(char)
	field = None
	episode = 0
	targets_loc, width, height = fds_origin_loc[char_index % len(fds_origin_loc)]
	nFDs = len(targets_loc)
	if field == None:
		field = environment(width, height, targets_loc)
		# Initialize DQN agent
		n_actions = field.n_actions
		agent = DQNAgent(width, height, n_actions, epsilon = 1.0)
		modelpath = 'models/char.h5'
		import os.path
		if os.path.exists(modelpath):
		    agent.load(modelpath)
		
	n_freedom = field.n_freedom

	terminated = [False] * n_freedom
	need_reset = [True] * n_freedom

	while episode < EPISODES:
		#for char in fds_origin_loc:
		step = 0
		while step < EPISODE_LENGTH:
			step += 1
			for n in range(n_freedom):
				if terminated[n]:
					continue

				if need_reset[n]:
					state = field.reset_freedom(n)
					need_reset[n] = False
				state = field.obsv(n)
				action = agent.get_action(state)
				next_state, reward, terminated[n], _ = field.step_freedom(n, action)
				agent.remember(state, action, reward, next_state, terminated[n])
				R += reward
				field.render()

			if False not in terminated:
				agent.update_target_model()
				need_reset = [True] * n_freedom
				terminated = [False] * n_freedom
				break
		episode += 1
		print "episode: ", episode, "/", EPISODES, " steps: ", step, "rewards", R / nFDs, "e: ", agent.epsilon, fds_list[char_index % len(fds_origin_loc)]
		if len(agent.memory) >= BATCH_SIZE:
			l = agent.experience_replay(BATCH_SIZE)

		if enable_plot:
			episodes.append(episode)
			steps.append(step)
			rewards.append(R / nFDs)
			plt.plot(episodes, steps, 'r')
			plt.plot(episodes, rewards, 'b')
			plt.xlim([int(episode / 100) * 100, int(episode / 100) * 100 + 100])
			plt.xlabel("Episodes")
			plt.legend(('Steps per episode', 'Rewards per episode'))
			fig.canvas.draw()
		 # Save trained agent every once in a while
		if episode % 100 == 0:
		    if enable_plot:
		        ax.clear()
		    agent.save(modelpath)

		if True not in terminated:
			R = 0

			if char == None:
				char_index += 1
				targets_loc, width, height = fds_origin_loc[char_index % len(fds_origin_loc)]
				field.update_env(targets_loc)
				n_freedom = field.n_freedom

				terminated = [False] * n_freedom
				need_reset = [True] * n_freedom


if __name__ == "__main__":
	
	from argparse import ArgumentParser
	parser = ArgumentParser(description='character')
	parser.add_argument('char', nargs = '?', default = None, help = 'which character')

	args = parser.parse_args()
	
	main(args.char)					
