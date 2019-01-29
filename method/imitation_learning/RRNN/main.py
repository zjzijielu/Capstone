import argparse
import numpy as np
from model import RRNN
from utils import *
from sample_perf import sample_perf
from sample_perf_acc import sample_perf_acc
from stretch_fill import stretch_fill
from get_fgt_acc import get_fgt_acc
from get_last_syncix import get_last_syncix
from get_median_alignedperf import get_median_alignedperf

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--hidden", '-hid', type=int, default=32, 
						help="hidden state size")
	parser.add_argument("--song", '-s', type=str, default="boy", 
						help="song to play")
	parser.add_argument("--p_closest", '-p', type=int, default=2, 
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
	parser.add_argument('--beat_p_bar', '-bpb', type=int, default=4,
                        help="beat per bar")
	parser.add_argument('--sec_p_beat', '-spb', type=int, default=1,
                        help="second per beat")                   

	args = parser.parse_args()
	hidden_size = args.hidden
	song = args.song
	p = args.p_closest
	epoch = args.epoch
	print_every = args.print_every
	plot_every = args.plot_every
	learning_rate = args.learning_rate
	clip = args.clip 
	beat_p_bar = args.beat_p_bar
	sec_p_beat = args.sec_p_beat

	print("#" * 10 + " Settings " + "#" * 10)
	print("size of mental state: ", hidden_size)
	print("song to play: ", song)
	print("p closest notes: ", p)
	print("epochs: ", epoch)
	print("print_every: ", print_every)
	print("plot_every: ", plot_every)
	print("beat_p_bar: ", beat_p_bar)
	print("sec_p_beat: ", sec_p_beat)
	print("#" * 20)

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
	# testfile_idx = []
	train_perf_mels = []
	train_perf_accs = []
	test_perf_mels = []
	test_perf_accs = []
	for i in range(len(fn_mel)):
		new_perf_mel = np.loadtxt(folder_mel + fn_mel[i])
		new_perf_acc = np.loadtxt(folder_acc + fn_acc[i])
		perf_mels.append(new_perf_mel)
		perf_accs.append(new_perf_acc)
		if (i + 1) % 7 != 0:
			train_perf_mels.append(new_perf_mel)
			train_perf_accs.append(new_perf_acc)
			trainfile_idx.append(i)
			train_num += 1
		else:
			test_perf_mels.append(new_perf_mel)
			test_perf_accs.append(new_perf_acc)
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



if __name__ == '__main__':
    main()