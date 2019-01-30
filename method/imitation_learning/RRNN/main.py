import argparse
import time
import numpy as np
from model import RRNN, FFNN
from utils import *
from sample_perf import sample_perf
from sample_perf_acc import sample_perf_acc
from stretch_fill import stretch_fill
from stretch_fill_acc import stretch_fill_acc
from stretch_follow import stretch_follow
from stretch_follow_acc import stretch_follow_acc
from get_fgt_acc import get_fgt_acc
from get_last_syncix import get_last_syncix
from get_median_alignedperf import get_median_alignedperf

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from model import RRNN

model_num = 0

def timeSince(since):
	now = time.time()
	s = now - since
	m = math.floor(s / 60)
	s -= m * 60
	return '%dm %ds' % (m, s)

def train_acc_st(acc_st_net, input, target, hidden_size, p, learning_rate, net_optimizer, clip):
	target_tensor = torch.from_numpy(target)
	criterion = nn.MSELoss()
	hidden = []
	global model_num
	if model_num == 1:
		hidden = acc_st_net.initHidden()
	total_loss = 0
	loss = 0
	all_losses = []

	for i in range(target.shape[0]):
		net_optimizer.zero_grad()
		input_tensor = torch.from_numpy(input[i])
		target_tensor = torch.from_numpy(target[i:i+1])
		if model_num == 0:
			output = acc_st_net(input_tensor)
		elif model_num == 1:
			output, hidden = acc_st_net(input_tensor, hidden)
		# print(input[i][-1], output, target_tensor)
		# output, hidden = acc_st_net(input_tensor, hidden)
		loss = criterion(output, target_tensor)
		total_loss += loss
		all_losses.append(loss)

		# if (loss > 10000):
		# 	raise ValueError("Gradient explode")
		
		loss.backward()
		if clip:
			nn.utils.clip_grad_norm_(acc_st_net.parameters(), clip)
		net_optimizer.step()	

	return output, total_loss / target.shape[0], all_losses

def train_acc_st_epoch(epoch, learning_rate, acc_st_net, inputs, targets, hidden_size, p, clip):
	total_loss = 0
	avg_losses = []
	start = time.time()
	iter = 1
	all_losses = []
	avg_losses_to_plot = []

	net_optimizer = optim.SGD(acc_st_net.parameters(), lr=learning_rate)
	train_num = inputs[0].shape[1]

	for e in range(epoch):
		print("#" * 20 + " epoch " + str(e) + " " + "#" * 20)
		for i in range(train_num):
			output, avg_loss, losses = train_acc_st(acc_st_net, inputs[:, :, i].astype(np.float32), targets[:, i].astype(np.float32), hidden_size, p, learning_rate, net_optimizer, clip)
			avg_losses.append(avg_loss)
			all_losses.extend(losses)
			if i == 3:
				avg_losses_to_plot.append(avg_loss)
			print("avg_loss of perf %d : %.4f" % (i, avg_loss))
			# if iter % print_every == 0:
			# 	print('%s (%d) %.4f' % (timeSince(start), iter, loss))

			# iter += 1
	showDifPlot(avg_losses, train_num, epoch, learning_rate, hidden_size, p, clip, model_num)
	return

def test_acc_st(acc_st_net, test_input, test_target):
	perf_num = test_target.shape[1]
	notes_num = test_target.shape[0]
	predict_output = np.zeros((notes_num, perf_num))
	criterion = nn.MSELoss()
	avg_losses = []
	hidden = []
	global model_num
	if model_num == 1:
		hidden = acc_st_net.initHidden()
	
	print("test_input shape: ", test_input.shape)
	print("test_target shape: ", test_target.shape)

	for i in range(perf_num):
		total_loss = 0
		for j in range(notes_num):
			input_tensor = torch.from_numpy(test_input[j, :, i].astype(np.float32))
			target_tensor = torch.from_numpy(test_target[j:j+1, i].astype(np.float32))
			if model_num == 0:
				output = acc_st_net(input_tensor)
			elif model_num == 1:
				output, hidden = acc_st_net(input_tensor, hidden)
			output_np = output.detach().numpy()
			predict_output[j, i] = output_np
			# raise ValueError
			loss = criterion(output, target_tensor)
			total_loss += loss
		avg_loss = total_loss / notes_num
		avg_losses.append(avg_loss)
		print("test sample %d avg loss: %.4f" % (i+1, avg_loss))

	return predict_output, avg_losses


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--hidden", '-hid', type=int, default=32, 
						help="hidden state size")
	parser.add_argument("--song", '-s', type=str, default="boy", 
						help="song to play")
	parser.add_argument("--p_closest", '-p', type=int, default=4, 
						help="p closest notes")
	parser.add_argument('--epoch', '-e', type=int, default=5, 
						help="number of epochs")
	parser.add_argument('--learning_rate', '-lr', type=float, default=0.001, 
						help="learning rate")
	# parser.add_argument('--print_every', '-print', type=int, default=10, 
	# 					help="frequency of printing results")
	# parser.add_argument('--plot_every', '-plot', type=int, default=10, 
	# 					help="frequency of plotting results")
	parser.add_argument('--clip', '-c', type=float, default=0.25,
						help="clip gradient")
	parser.add_argument('--beat_p_bar', '-bpb', type=int, default=4,
                        help="beat per bar")
	parser.add_argument('--sec_p_beat', '-spb', type=int, default=1,
                        help="second per beat")                  
	parser.add_argument('--duration', '-dur', type=int, default=1,
						help='whether to include duration info')
	parser.add_argument('--end_time', '-et', type=int, default=0, 
						help="whether to predict end time")
	parser.add_argument('--model', '-m', type=int, default=0,
						help="which model to choose\n0. FFNN\n1. RRNN")

	args = parser.parse_args()
	hidden_size = args.hidden
	song = args.song
	p = args.p_closest
	epoch = args.epoch
	# print_every = args.print_every
	# plot_every = args.plot_every
	learning_rate = args.learning_rate
	clip = args.clip 
	beat_p_bar = args.beat_p_bar
	sec_p_beat = args.sec_p_beat
	duration = args.duration
	if_et = args.end_time
	global model_num
	model_num = args.model

	print("#" * 10 + " Settings " + "#" * 10)
	print("mental state size: \t", hidden_size)
	print("song to play: \t", song)
	print("p closest notes: \t", p)
	print("epochs: \t", epoch)
	print("beat_p_bar: \t", beat_p_bar)
	print("sec_p_beat: \t", sec_p_beat)
	print("duration: \t", bool(duration))
	print("end_time: \t", bool(if_et))
	print("model: \t", model_num)
	print("learning_rate: \t", learning_rate)

	# if to predict end time, output size should be 2
	if if_et:
		output_size = 2
	else:
		output_size = 1

    # load score, median and perf files
	folder_full = '/Users/luzijie/Desktop/Capstone/data/polysample_' + song + '/'
	folder_mel = folder_full + 'aligned_melody/'
	folder_acc = folder_full + 'aligned_accompany/'
	fn_mel = find('aligned_mel_*.txt', folder_mel)
	fn_acc = find('aligned_acc_*.txt', folder_acc)
	folder_score_mel = folder_mel + 'score/'
	folder_score_acc = folder_acc +'score/'
	fn_score_mel = find('aligned_mel_*.txt', folder_score_mel)
	fn_score_acc = find('aligned_acc_*.txt', folder_score_acc)

	score_mel = np.loadtxt(folder_score_mel + fn_score_mel[0])
	score_acc = np.loadtxt(folder_score_acc + fn_score_acc[0])
	
	sampled_score_folder = "/Users/luzijie/Desktop/Capstone/data/polysample_" + song + "/sampled_score/"
	median_perf_folder = "/Users/luzijie/Desktop/Capstone/data/polysample_" + song + "/syn/"
	acc_st_folder = "./results/test_acc_st_output/"

	# quantify the dynamics
	score_acc[:,1] = np.floor(np.mean(score_acc[:,1]))
	score_mel[:,1] = np.floor(np.mean(score_mel[:,1]))

	# useful parameters
	N = score_mel.shape[0]
	M = score_acc.shape[0]
	L = len(fn_mel)
	train_num = 0
	test_num = 0
	sr = 2 # sample rate
	period = sec_p_beat / sr

	# load sampled score
	sampled_score_mel = sample_perf(score_mel,score_mel,score_acc, score_acc, sec_p_beat, beat_p_bar, sr, 1)
	sampled_score_acc = sample_perf_acc(score_acc, score_acc, sampled_score_mel, sampled_score_mel, sec_p_beat, sr, 1)
	matrix2midi(sampled_score_acc, sampled_score_folder + 'sampled_score_acc.mid')
	p_slope = p
	p_sampleslope = p / 4
	# sampled_score_mel = np.loadtxt(sampled_score_folder + 'sampled_score_mel.txt')
	# sampled_score_acc = np.loadtxt(sampled_score_folder + 'sampled_score_acc.txt')
	SN = sampled_score_mel.shape[0]
	SM = sampled_score_acc.shape[0]

	# load median perf file
	median_perf_mel = np.loadtxt(median_perf_folder + 'median_mel.txt')
	median_perf_acc = np.loadtxt(median_perf_folder + 'median_acc.txt')
	# sampled perf and score 
	sampled_perf_mels = np.zeros((SN, 5, L))
	sampled_perf_accs = np.zeros((SM, 5, L))
	sampled_perf_mels_slopes = np.zeros((SN, L))
	sampled_perf_accs_slopes = np.zeros((SM, L))

	# create train set and test set
	perf_mels = []
	perf_accs = []
	trainfile_idx = []
	testfile_idx = []

	for i in range(len(fn_mel)):
		new_perf_mel = np.loadtxt(folder_mel + fn_mel[i])
		new_perf_acc = np.loadtxt(folder_acc + fn_acc[i])
		perf_mels.append(new_perf_mel)
		perf_accs.append(new_perf_acc)
		if (i + 1) % 7 != 0:
			trainfile_idx.append(i)
			train_num += 1
		else:
			testfile_idx.append(i)
			test_num += 1

	# for now follow baseline method and use fake ground truth
	# construct experiment truth and rslt container
	# (fake) ground truth
	aligned_perf_mels_fgt = np.zeros((N, 4, L))
	aligned_perf_accs_fgt = np.zeros((M, 4, L))
	missing_melix = []
	missing_accix = []
	missing_accevtix = []
	# median for decoding each songs 
	aligned_median_mels = np.zeros((N, 4, L))
	aligned_median_accs = np.zeros((M, 4, L))
	train_aligned_median_mels = np.zeros((N, 4, train_num))
	train_aligned_median_accs = np.zeros((M, 4, train_num))
	# sampled perf and score 
	train_sampled_perf_mels = np.zeros((SN, 5, train_num))
	train_sampled_perf_accs = np.zeros((SM, 5, train_num))
	test_sampled_perf_mels = np.zeros((SN, 5, test_num))
	test_sampled_perf_accs = np.zeros((SM, 5, test_num))
	# rslt for only timing stretch, keey pervious dy
	aligned_perf_accs_dmsm = np.zeros((M,4,L))
	aligned_perf_mels_dmsm = np.zeros((N,4,L))
	aligned_perf_accs_dssm = np.zeros((M,4,L))
	aligned_perf_mels_dssm = np.zeros((N,4,L))
	# rslt for timing stretch + dy from reference
	aligned_perf_accs_dmscale = np.zeros((M,4,L))
	aligned_perf_mels_dmscale = np.zeros((N,4,L))
	aligned_perf_accs_dsscale = np.zeros((M,4,L))
	aligned_perf_mels_dsscale = np.zeros((N,4,L))
	# rslt for stretch follow sampled perf mel/acc
	aligned_perf_mels_dsssm = np.zeros((N,4,L))
	aligned_perf_accs_dsssm = np.zeros((M,4,L))
	sampled_perf_mels_dsssm = np.zeros((SN,4,L)) # sssm = score stretch sampled melody
	sampled_perf_accs_dsssm = np.zeros((SM,4,L))
	sampled_perf_accs_dsssa = np.zeros((SM,4,L)) # sssm = score stretch sampled acc
	# slopes as feature
	train_sampled_perf_mels_slopes = np.zeros((SN, train_num))
	train_sampled_perf_accs_slopes = np.zeros((SM, train_num))
	test_sampled_perf_mels_slopes = np.zeros((SN, test_num))
	test_sampled_perf_accs_slopes = np.zeros((SM, test_num))


	# compute the fake ground truth of perf_mel and perf_acc by perfs'
	for i in range(L):
		aligned_perf_mels_fgt[:, :, i], missing_melix_new = stretch_fill(score_mel, perf_mels[i], p)
		missing_melix.append(missing_melix_new)
		aligned_perf_accs_fgt[:, :, i], missing_accix_new, missing_accevtix_new = get_fgt_acc(score_mel, score_acc, perf_mels[i], perf_accs[i], p, sec_p_beat)
		missing_accix.append(missing_accix_new)
		missing_accevtix.append(missing_accevtix_new)

	c_melix, c_accix = get_last_syncix(score_mel, score_acc)
	# median to decode each piece (decoded piece excluded)
	perffile_idx = np.arange(0, L)
	trainfile_idx = np.array(trainfile_idx)
	for i in range(L):
		medianfile_idx = np.setdiff1d(perffile_idx, i)
		aligned_median_mels[:, :, i] = get_median_alignedperf(aligned_perf_mels_fgt[:, :, medianfile_idx], c_melix)
		aligned_median_accs[:, :, i] = get_median_alignedperf(aligned_perf_accs_fgt[:, :, medianfile_idx], c_accix)

	for i in range(len(trainfile_idx)):
		medianfile_idx = np.setdiff1d(trainfile_idx, i)
		train_aligned_median_mels[:, :, i] = get_median_alignedperf(aligned_perf_mels_fgt[:, :, medianfile_idx], c_melix)
		train_aligned_median_accs[:, :, i] = get_median_alignedperf(aligned_perf_accs_fgt[:, :, medianfile_idx], c_accix)

	# compute sampled melody and acc
	p_extra = 1	
	for i in perffile_idx:
		sampled_perf_mels[:, :, i] = sample_perf(score_mel, aligned_perf_mels_fgt[:, :, i], score_acc, aligned_perf_accs_fgt[:, :, i], sec_p_beat, beat_p_bar, sr, p_extra)
		sampled_perf_accs[:, :, i] = sample_perf_acc(score_acc, aligned_perf_accs_fgt[:, :, i], sampled_score_mel, sampled_perf_mels[:, :, i], sec_p_beat, sr, p_extra)

	train_idx = 0
	test_idx = 0
	for i in perffile_idx:
		if (i+1) % 7 != 0:
			train_sampled_perf_mels[:, :, train_idx] = sampled_perf_mels[:, :, i]
			train_sampled_perf_accs[:, :, train_idx] = sampled_perf_accs[:, :, i]
			train_idx += 1
		else:
			test_sampled_perf_mels[:, :, test_idx] = sampled_perf_mels[:, :, i]
			test_sampled_perf_accs[:, :, test_idx] = sampled_perf_accs[:, :, i]
			test_idx += 1

	# for i in range(len(trainfile_idx)):
	# 	idx = trainfile_idx[i]
	# 	medianfile_idx = np.setdiff1d(trainfile_idx, idx)
	# 	train_sampled_perf_mels[:, :, i] = sample_perf(score_mel, aligned_perf_mels_fgt[:, :, idx], score_acc, aligned_perf_accs_fgt[:, :, idx], sec_p_beat, beat_p_bar, sr, p_extra)
	# 	train_sampled_perf_accs[:, :, i] = sample_perf_acc(score_acc, aligned_perf_accs_fgt[:, :, idx], sampled_score_mel, train_sampled_perf_mels[:, :, i], sec_p_beat, sr, p_extra)

	# for i in range(len(testfile_idx)):
	# 	idx = testfile_idx[i]
	# 	medianfile_idx = np.setdiff1d(perffile_idx, idx)
	# 	test_sampled_perf_mels[:, :, i] = sample_perf(score_mel, aligned_perf_mels_fgt[:, :, idx], score_acc, aligned_perf_accs_fgt[:, :, idx], sec_p_beat, beat_p_bar, sr, p_extra)
	# 	test_sampled_perf_accs[:, :, i] = sample_perf_acc(score_acc, aligned_perf_accs_fgt[:, :, idx], sampled_score_mel, train_sampled_perf_mels[:, :, i], sec_p_beat, sr, p_extra)

	assert(np.array_equal(test_sampled_perf_mels[:, :, 0],sampled_perf_mels[:, :, 6]))

	# extract sampled slopes
	print("#" * 10 + " Extract slopes of unsampled perf " + "#" * 10)
	train_idx = 0
	test_idx = 0 
	for i in perffile_idx:
		_, _, _, _, sampled_perf_mels_slopes[:, [i]] = stretch_follow(sampled_score_mel, sampled_perf_mels[:, :, i], p_sampleslope, sec_p_beat)
		if (i+1) % 7 != 0:
			_, _, _, _, train_sampled_perf_mels_slopes[:, [train_idx]] = stretch_follow(sampled_score_mel, sampled_perf_mels[:, :, i], p_sampleslope, sec_p_beat)
			train_idx += 1
		else:
			_, _, _, _, test_sampled_perf_mels_slopes[:, [test_idx]] = stretch_follow(sampled_score_mel, sampled_perf_mels[:, :, i], p_sampleslope, sec_p_beat)
			test_idx += 1

	print("#" * 10 + " Initialize model " + "#" * 10)
	if duration:
		feature_len = p * 4 + 1
	else: 
		feature_len = p * 3 + 1
	
	if model_num == 0:
		acc_st_net = FFNN(feature_len, hidden_size, output_size)
	else:
		acc_st_net = RRNN(feature_len, hidden_size, output_size)
	
	# generate training data
	# target_perf_accs = np.zeros((SM - p, 2, train_num))
	input_features = np.zeros((SM - p, feature_len, train_num))
	target_perf_accs_st = train_sampled_perf_accs[p:, 2, :]
	# target_perf_accs[:, 0, :] = train_sampled_perf_accs[p:, 2, :]
	# target_perf_accs[:, 1, :] = train_sampled_perf_accs[p:, 3, :]
	for i in range(train_num):
		for j in range(SM - p):
			pre_melidx = np.where(train_sampled_perf_mels[:, 2, i] < train_sampled_perf_accs[j+p, 2, i])[0][-p:]
			if pre_melidx.shape[0] < p:
				pad = p - pre_melidx.shape[0]
				input_features[j, :pad, i] = 0
				input_features[j, pad:p, i] = train_sampled_perf_mels[pre_melidx, 2, i]
				input_features[j, p:p+pad, i] = 0
				input_features[j, p+pad:2*p, i] = train_sampled_perf_mels_slopes[pre_melidx, i]
				input_features[j, 2*p:2*p+pad, i] = 0
				input_features[j, 2*p+pad:3*p, i] = sampled_score_mel[pre_melidx, 2]
				if duration: 
					input_features[j, 3*p:3*p+pad, i] = 0
					input_features[j, 3*p+pad:, i] = train_sampled_perf_mels[pre_melidx, 3, i] - train_sampled_perf_mels[pre_melidx, 2, i]
			else:
				input_features[j, :p, i] = train_sampled_perf_mels[pre_melidx, 2, i]
				input_features[j, p:2*p, i] = train_sampled_perf_mels_slopes[pre_melidx, i]
				input_features[j, 2*p:3*p, i] = sampled_score_mel[pre_melidx, 2]
				if duration:
					input_features[j, 3*p:-1, i] = train_sampled_perf_mels[pre_melidx, 3, i] - train_sampled_perf_mels[pre_melidx, 2, i]
			input_features[j, -1, i] = sampled_score_acc[j+p, 2]

	# generate test data
	# test_target_accs = np.zeros((SM - p, 2, test_num))
	test_input_features = np.zeros((SM - p, feature_len, test_num))
	test_target_accs_st = test_sampled_perf_accs[p:, 2, :]
	# test_target_accs[:, 0, :] = test_sampled_perf_accs[p:, 2, :]
	# test_target_accs[:, 1, :] = test_sampled_perf_accs[p:, 3, :]
	assert(test_input_features.shape[:-1] == input_features.shape[:-1])
	for i in range(test_num):
		for j in range(SM - p):
			pre_melidx = np.where(test_sampled_perf_mels[:, 2, i] < test_sampled_perf_accs[j+p, 2, i])[0][-p:]
			# print(pre_melidx)
			if pre_melidx.shape[0] < p:
				pad = p - pre_melidx.shape[0]
				test_input_features[j, :pad, i] = 0
				test_input_features[j, pad:p, i] = test_sampled_perf_mels[pre_melidx, 2, i]
				test_input_features[j, p:p+pad, i] = 0
				test_input_features[j, p+pad:2*p, i] = test_sampled_perf_mels_slopes[pre_melidx, i]
				test_input_features[j, 2*p:2*p+pad, i] = 0
				test_input_features[j, 2*p+pad:3*p, i] = sampled_score_mel[pre_melidx, 2]
				if duration:
					test_input_features[j, 3*p:3*p+pad, i] = 0
					test_input_features[j, 3*p+pad:, i] = test_sampled_perf_mels[pre_melidx, 3, i] - test_sampled_perf_mels[pre_melidx, 2, i]
			else:
				test_input_features[j, :p, i] = test_sampled_perf_mels[pre_melidx, 2, i]
				test_input_features[j, p:2*p, i] = test_sampled_perf_mels_slopes[pre_melidx, i]
				test_input_features[j, 2*p:3*p, i] = sampled_score_mel[pre_melidx, 2]
				if duration:
					test_input_features[j, 3*p:-1, i] = test_sampled_perf_mels[pre_melidx, 3, i] - test_sampled_perf_mels[pre_melidx, 2, i]
			test_input_features[j, -1, i] = sampled_score_acc[j+p, 2]


	# train
	print("#" * 10 + " Train model " + "#" * 10)
	train_acc_st_epoch(epoch, learning_rate, acc_st_net, input_features, target_perf_accs_st, hidden_size, p, clip)
	
	# test
	predict_loss = np.zeros((test_num, 2))
	print("#" * 10 + " Test model " + "#" * 10)
	predict_acc_st, predict_loss[:, 1] = test_acc_st(acc_st_net, test_input_features, test_target_accs_st)
	predict_loss[:, 0] = np.array(testfile_idx)
	testfile_name = "test_loss_e" + str(epoch) + "_lr" + str(learning_rate) + "_h" + str(hidden_size) + "_p" + str(p) + "_c" + str(clip) + "_dur" + str(duration) + ".txt"
	np.savetxt(acc_st_folder + testfile_name, predict_loss, fmt="%.4f")
	# output test predict result to midi file


if __name__ == '__main__':
    main()