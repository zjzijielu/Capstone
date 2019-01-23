import sys
import argparse
import numpy as np
from model import RRNN
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from utils import showDifPlot

def timeSince(since):
	now = time.time()
	s = now - since
	m = math.floor(s / 60)
	s -= m * 60
	return '%dm %ds' % (m, s)


def parse_data(file_addr):
	file = open(file_addr, "r")

	lines = file.readlines()

	note = []
	dynamic = []
	start_t = []
	end_t = []
	order = []

	for i in lines:
		line = i.split('\t')
		note.append(float(line[0]))
		dynamic.append(float(line[1]))
		start_t.append(float(line[2]))
		end_t.append(float(line[3]))
		order.append(float(line[4]))

	return note, dynamic, start_t, end_t, order

def compute_exp_timing(score, perf, score_order, perf_order):
	exp_timing = []
	for i in range(len(perf)):
		# print(i, perf_order[i])
		score_index = score_order.index(perf_order[i])
		exp_timing.append(round(perf[i] - score[score_index], 4))
	# print(exp_timing)
	return exp_timing

def train_data_generate(p, start_t_mel, end_t_mel, dynamic_mel, start_t_acc, end_t_acc, dynamic_acc, exp_timing_mel, exp_timing_acc):
	input = []
	target = []
	for i in range(len(start_t_acc)):
		acc_t = start_t_acc[i]
		# get the closest time in melody performance
		closest_mel_t = min(start_t_mel, key = lambda x: (abs(x - acc_t), x))
		closest_mel_index = start_t_mel.index(closest_mel_t)
		if closest_mel_index >= p:
			closest_exp_timing_mel = exp_timing_mel[closest_mel_index-p+1:closest_mel_index+1]
		else:
			closest_exp_timing_mel = exp_timing_mel[0:closest_mel_index+1]
		if i >= p:
			closest_exp_timing_acc = exp_timing_acc[i-p:i]
		else:
			closest_exp_timing_acc = [0] * (p - i) + exp_timing_acc[0:i]
			assert(len(closest_exp_timing_acc) == p)
		# concatenate expressive timing of accompaniment
		exp_timing_feature = closest_exp_timing_mel # + closest_exp_timing_acc
		assert(len(exp_timing_feature) == p)
		input.append(exp_timing_feature)
		target.append([exp_timing_acc[i]])
	input = np.asarray(input, dtype=np.float32)
	target = np.asarray(target, dtype=np.float32)
	return input, target

def train_acc(acc_net, input, target, hidden_size, p, learning_rate, net_optimizer, clip, plot_every):
	all_losses = []
	initial_loss = []
	half_loss = []
	last_loss = []
	input_tensor = []
	target_tensor = []
	loss = 0

	hidden = acc_net.initHidden()

	input_tensor = torch.from_numpy(input)
	target_tensor = torch.from_numpy(target)
	# for i in range(len(input)):
	# 	# print("#" * 10 + str(i) + "-th input: ")
	# 	# print(inputs[i])
	# 	# print(targets[i])
	# 	input_tensor = torch.from_numpy(input)
	# 	target_tensor = torch.from_numpy(target[i])
	criterion = nn.MSELoss()

	# for i in range(len(inputs_tensor)):
	hidden = acc_net.initHidden()
	for i in range(len(input)):
		net_optimizer.zero_grad()
		loss = 0
		# print("#" * 20)
		# print(i, j)	
		output, hidden = acc_net(input_tensor[i], hidden)
		l = criterion(output, target_tensor[i])
		loss += l
		if i == 0:
			print("1st note loss: ", l)
			all_losses.append(l)
			initial_loss.append(l)
		if i == int(len(input) / 2):
			print("half way loss: ", l)
			all_losses.append(l)
			half_loss.append(l)
		if i == len(input) - 1:
			print("last note loss: ", l)
			all_losses.append(l)
			last_loss.append(l)
		# if i % plot_every == 0:
		# 	all_losses.append(l)
		# print(l, output, targets[i][j])
		if (l > 10000):
			raise ValueError('Gradient explode')
		# if j == 0:
		# 	print("input tensor:")
		# 	print(l)
		# 	print(inputs_tensor[i][j])
		# 	print(output, targets_tensor[i][j])
		loss.backward()
		torch.nn.utils.clip_grad_norm_(acc_net.parameters(), clip)
		net_optimizer.step()
	
	return output, loss.item(), all_losses, initial_loss, half_loss, last_loss

def train_acc_epoch(epoch, print_every, plot_every, learning_rate, acc_net, inputs, targets, hidden_size, p, clip):
	all_losses = []
	initial_losses = []
	half_losses = []
	last_losses = []
	total_loss = 0
	start = time.time()
	iter = 1

	net_optimizer = optim.SGD(acc_net.parameters(), lr=learning_rate)

	print(len(inputs))

	for e in range(epoch):
		print("#" * 20 + " epoch " + str(e) + " " + "#" * 20)
		for i in range(len(inputs)):
			output, loss, all_loss, initial_loss, half_loss, last_loss = train_acc(acc_net, inputs[i], targets[i], hidden_size, p, learning_rate, net_optimizer, clip, plot_every)
			all_losses.extend(all_loss)
			total_loss += loss
			initial_losses.extend(initial_loss)
			half_losses.extend(half_loss)
			last_losses.extend(last_loss)
			if iter % print_every == 0:
				print('%s (%d) %.4f' % (timeSince(start), iter, loss))

			# if iter % plot_every == 0:
			# 	all_losses.append(total_loss / plot_every)
			# 	total_loss = 0
			iter += 1
			# print(iter)
	
	# showPlot(all_losses)
	showDifPlot(initial_losses, half_losses, last_losses)

def test_acc(acc_net, input, target):
	hidden_state = acc_net.initHidden()
	for i in range(len(input)):
		break
		# output, loss = acc_net(input, hidden_state, output)

	return 
	

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--hidden", '-hid', type=int, default=32, 
						help="hidden state size")
	parser.add_argument("--song", '-s', type=str, default="boy", 
						help="song to play")
	parser.add_argument("--p_closest", '-p', type=int, default=3, 
						help="p closest notes")
	parser.add_argument('--epoch', '-e', type=int, default=10, 
						help="number of epochs")
	parser.add_argument('--learning_rate', '-lr', type=float, default=0.001, 
						help="learning rate")
	parser.add_argument('--print_every', '-print', type=int, default=10, 
						help="frequency of printing results")
	parser.add_argument('--plot_every', '-plot', type=int, default=10, 
						help="frequency of plotting results")
	parser.add_argument('--clip', '-c', type=float, default=0.25,
						help="clip gradient")

	args = parser.parse_args()
	hidden_size = args.hidden
	song = args.song
	p = args.p_closest
	epoch = args.epoch
	print_every = args.print_every
	plot_every = args.plot_every
	learning_rate = args.learning_rate
	clip = args.clip 

	print("#" * 10 + " Settings " + "#" * 10)
	print("size of mental state: ", hidden_size)
	print("song to play: ", song)
	print("p closest notes: ", p)
	print("epochs: ", epoch)
	print("print_every: ", print_every)
	print("plot_every: ", plot_every)

	note_score_mel = []
	note_score_acc = []
	note_perf_mel = []
	note_perf_acc = []
	dynamic_score_mel = []
	dynamic_score_acc = []
	dynamic_perf_mel = []
	dyanmic_perf_acc = []
	start_t_score_mel = []
	start_t_score_acc = []
	start_t_perf_mel = []
	start_t_perf_acc = []
	end_t_score_mel = []
	end_t_score_acc = []
	end_t_perf_mel = []
	end_t_perf_acc = []
	order_score_mel = []
	order_score_acc = []
	order_perf_mel = []
	order_perf_acc = []
	
	start_exp_timing_mel = []
	start_exp_timing_acc = []

	performer = ["cmaw", "czpp", "draw", "smtl", "ymgl", "ysyf"]

	print("#" * 10 + " Parsing data " + "#" * 10)
	# parse score melody
	score_mel_file = "data/polysample_boy/aligned_melody/aligned_mel_labeled-" + song + "-score.txt"
	note_score_mel, dynamic_score_mel, start_t_score_mel, end_t_score_mel, order_score_mel = parse_data(score_mel_file)
	# parse score accompany
	score_acc_file = "data/polysample_boy/aligned_accompany/aligned_acc_labeled-" + song + "-score.txt"
	note_score_acc, dynamic_score_acc, start_t_score_acc, end_t_score_acc, order_score_acc = parse_data(score_acc_file)
	# parse melody accompaniment performance
	inputs = []
	targets = []
	for i in range(6):
		for j in range(6):
			mel_file_name = "data/polysample_boy/aligned_melody/aligned_mel_labeled-" + song + "-" + performer[i] + "-" + str(j+1)+ "-perf.txt"
			# print(mel_file_name)
			new_note_mel, new_dynamic_mel, new_start_t_mel, new_end_t_mel, new_order_mel = parse_data(mel_file_name)
			new_exp_timing_mel = compute_exp_timing(start_t_score_mel, new_start_t_mel, order_score_mel, new_order_mel)
			# start_exp_timing_mel.append(compute_exp_timing(start_t_score_mel, new_start_t, order_score_mel, new_order))
			# note_perf_mel.append(new_note)
			# dynamic_perf_mel.append(new_dynamic)
			# start_t_perf_mel.append(new_start_t)
			# end_t_perf_mel.append(new_end_t)

			acc_file_name = "data/polysample_boy/aligned_accompany/aligned_acc_labeled-"+ song + "-" + performer[i] + "-" + str(j+1)+ "-perf.txt"
			# print(acc_file_name)
			new_note_acc, new_dynamic_acc, new_start_t_acc, new_end_t_acc, new_order_acc = parse_data(acc_file_name)
			new_exp_timing_acc = compute_exp_timing(start_t_score_acc, new_start_t_acc, order_score_acc, new_order_acc)
			# start_exp_timing_acc.append(compute_exp_timing(start_t_score_acc, new_start_t, order_score_acc, new_order))
			# note_perf_acc.append(new_note)
			# dyanmic_perf_acc.append(new_dynamic)
			# start_t_perf_acc.append(new_start_t)
			# end_t_perf_acc.append(new_end_t)
			# create one input-target pair for each file
			input, target = train_data_generate(p, new_start_t_mel, new_end_t_mel, new_dynamic_mel, new_start_t_acc, new_end_t_acc, new_dynamic_acc, new_exp_timing_mel, new_exp_timing_acc)
			# input, target = train_data_generate(p, start_t_perf_mel, end_t_perf_mel, dynamic_perf_mel, start_t_perf_acc, end_t_perf_acc, dyanmic_perf_acc, start_exp_timing_mel, start_exp_timing_acc)
			inputs.append(input)
			targets.append(target)

	# initialize model
	print("#" * 10 + " Initialize model " + "#" * 10)
	acc_net = RRNN(p, hidden_size, 1)
	
	# train
	print("#" * 10 + " Train model " + "#" * 10)
	train_acc_epoch(epoch, print_every, plot_every, learning_rate, acc_net, inputs, targets, hidden_size, p, clip)

	# test

if __name__ == '__main__':
	main()