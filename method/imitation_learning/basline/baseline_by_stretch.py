import os, fnmatch
from get_scorefile_ix import get_scorefile_ix
from sample_perf import sample_perf
from sample_perf_acc import sample_perf_acc
from utils import *
from get_last_syncix import get_last_syncix
import numpy as np

def baseline_by_strectch(folder_full, p, sec_p_beat, beat_p_bar, key, flag_debug=0):
    '''
    baseline_by_stretch: compute and save the the BL by stretch method.
    output: BL contain
    1. fake gt for perf mel and acc, and sampled version
    2. missing melix for mel; missing accix for acc
    3. rslt for simple stretch and meidan stretch
    4. rsdl for simple stretch and median stretch
    input: 
        folder_full: the folder contain the data
        p: lag parameter in beat
        flag_debug: 1 when run alone, doesn't save or modify data
                    0 when be called in a chain, save data
    '''
    
    # test
    p = 2
    sec_p_beat = 1
    folder_full = '/Users/luzijie/Desktop/Capstone/project/data/polysample_boy/'
    flag_debug = 0
    beat_p_bar = 4
    key = 60

    # path
    fn_full = find('labeled*.txt', folder_full)
    folder_mel = folder_full + 'aligned_melody/'
    fn_mel = find('aligned_mel_*.txt', folder_mel)
    folder_acc = folder_full + 'aligned_accompany/'
    fn_acc = find('aligned_acc_*.txt', folder_acc)
    folder_syn = folder_full + 'syn/'
    folder_exp = folder_full + 'exp_rslt'

    # read in raw files and compute size parameters
    # read in scores
    scorefile_ix = get_scorefile_ix(folder_full)
    assert(scorefile_ix != -1)

    score_mel = np.loadtxt(folder_mel + fn_mel[scorefile_ix])
    score_acc = np.loadtxt(folder_acc + fn_acc[scorefile_ix])
    print(score_acc.shape)
    
    # quantify the dynamics
    score_acc[:,1] = np.floor(np.mean(score_acc[:,1]))
    score_mel[:,1] = np.floor(np.mean(score_mel[:,1]))

    N = score_mel.shape[0]
    M = score_acc.shape[0]
    sr = 2 # sample rate
    period = sec_p_beat / sr
    sampled_score_mel = sample_perf(score_mel,score_mel,score_acc, score_acc, sec_p_beat, beat_p_bar, sr, 1)
    SN = sampled_score_mel.shape[0]
    sampled_score_acc = sample_perf_acc(score_acc, score_acc, sampled_score_mel,sampled_score_mel,sec_p_beat, sr, 1)
    SM = sampled_score_acc.shape[0]
    p_slope = p
    p_sampleslope = p / 4

    # read in perfs, mel and acc into 2 structure array
    # NOTE: perf_ * s[scorefile_ix] = []
    perffile_ix = np.arange(0, len(fn_mel))
    perffile_ix[scorefile_ix] = -1
    L = len(perffile_ix)
    perf_mels = []
    perf_accs = []
    perf_fulls = []
    for i in range(len(perffile_ix)):
        idx = perffile_ix[i]
        perf_mels.append(np.loadtxt(folder_mel + fn_mel[idx]))
        perf_accs.append(np.loadtxt(folder_acc + fn_acc[idx]))
        perf_fulls.append(np.loadtxt(folder_full + fn_full[idx]))
    
    # construct experiment truth and rslt container
    # (fake) ground truth
    aligned_perf_mels_fgt = np.zeros((N, 4, L+1))
    aligned_perf_accs_fgt = np.zeros((M, 4, L+1))
    missing_melix = np.zeros((L+1, 1))
    missing_accix = np.zeros((L+1, 1))
    missing_accevtix = np.zeros((L+1, 1))
    # all slopes are selfnote-included slopes
    aligned_perf_mels_slopes = np.zeros((N, L+1))
    aligned_perf_accs_slopes = np.zeros((M, L+1))
    # sampled perf and score 
    sampled_perf_mels = np.zeros((SN, 5, L+1))
    sampled_perf_accs = np.zeros((SM, 5, L+1))
    sampled_perf_mels_slopes = np.zeros((SN, L+1))
    sampled_perf_accs_slopes = np.zeros((SM, L+1))
    # median for decoding each songs 
    aligned_median_mels = np.zeros((N, 4, L+1))
    aligned_median_accs = np.zeros((M, 4, L+1))
    # results for different method 
    # rslt for only timing stretch, keep previous dy
    aligned_perf_accs_dmsm = np.zeros((M,4,L+1))
    aligned_perf_mels_dmsm = np.zeros((N,4,L+1))
    aligned_perf_accs_dssm = np.zeros((M,4,L+1))
    aligned_perf_mels_dssm = np.zeros((N,4,L+1))
    # rslt for timing stretch + dy from reference
    aligned_perf_accs_dmscale = np.zeros((M,4,L+1))
    aligned_perf_mels_dmscale = np.zeros((N,4,L+1))
    aligned_perf_accs_dsscale = np.zeros((M,4,L+1))
    aligned_perf_mels_dsscale = np.zeros((N,4,L+1))
    # index when doing stretch follow
    Begin_ix_acc = np.zeros((L+1,1))
    Sparse_ix_acc = np.zeros((L+1,1))
    Begin_ix_mel = np.zeros((L+1,1))
    Sparse_ix_mel = np.zeros((L+1,1))
    # rslt for stretch follow sampled perf mel/acc
    aligned_perf_mels_dsssm = np.zeros((N,4,L+1))
    aligned_perf_accs_dsssm = np.zeros((M,4,L+1))
    sampled_perf_mels_dsssm = np.zeros((SN,4,L+1)) # sssm = score stretch sampled melody
    sampled_perf_accs_dsssm = np.zeros((SM,4,L+1))
    sampled_perf_accs_dsssa = np.zeros((SM,4,L+1)) # sssm = score stretch sampled acc

    # residuals for different method
    Rsdl_accs_dmsm = np.zeros((M,4,L+1))
    Rsdl_mels_dmsm = np.zeros((N,4,L+1))
    Rsdl_accs_dssm = np.zeros((M,4,L+1))
    Rsdl_mels_dssm = np.zeros((N,4,L+1))
    Rsdl_accs_dmscale = np.zeros((M,4,L+1))
    Rsdl_mels_dmscale = np.zeros((N,4,L+1))
    Rsdl_accs_dsscale = np.zeros((M,4,L+1))
    Rsdl_mels_dsscale = np.zeros((N,4,L+1))
    # rsdl for stretch follow sampled perf mel/acc
    Rsdl_mels_dsssm = np.zeros((N,4,L+1))
    Rsdl_accs_dsssm = np.zeros((M,4,L+1)) 
    Rsdl_smels_dsssm = np.zeros((SN,4,L+1))
    Rsdl_saccs_dsssm = np.zeros((SM,4,L+1))
    Rsdl_saccs_dsssa = np.zeros((SM,4,L+1))

    # compute the fake ground truth of perf_mel and perf_acc by perfs'
    for i in range(len(perffile_ix)):
        idx = perffile_ix[i]
        
    # the overall median
    c_melix, c_accix = get_last_syncix(score_mel, score_acc)

    

def main():
    baseline_by_strectch(0,0,0,0,0)

if __name__ == '__main__':
    main()