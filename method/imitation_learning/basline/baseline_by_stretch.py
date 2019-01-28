import os, fnmatch
from get_scorefile_ix import get_scorefile_ix
from sample_perf import sample_perf
from sample_perf_acc import sample_perf_acc
from utils import *
from notes2cevt import notes2cevt
from get_last_syncix import get_last_syncix
from stretch_fill import stretch_fill
from get_fgt_acc import get_fgt_acc
from extract_perfs_info import extract_perfs_info
from stretch_follow import stretch_follow
from stretch_follow_acc import stretch_follow_acc
from get_median_alignedperf import get_median_alignedperf
import numpy as np
from matrix2midi import matrix2midi

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
    song = "boy"
    folder_full = '/Users/luzijie/Desktop/Capstone/data/polysample_' + song + '/'
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
    folder_score_mel = folder_mel + 'score/'
    fn_score_mel = find('aligned_mel_*.txt', folder_score_mel)
    folder_score_acc = folder_acc + 'score/'
    fn_score_acc = find('aligned_acc_*.txt', folder_score_acc)

    # read in raw files and compute size parameters
    # read in scores
    scorefile_ix = get_scorefile_ix(folder_full)
    assert(scorefile_ix != -1)

    score_mel = np.loadtxt(folder_score_mel + fn_score_mel[0])
    score_acc = np.loadtxt(folder_score_acc + fn_score_acc[0])
    
    # quantify the dynamics
    score_acc[:,1] = np.floor(np.mean(score_acc[:,1]))
    score_mel[:,1] = np.floor(np.mean(score_mel[:,1]))

    N = score_mel.shape[0]
    M = score_acc.shape[0]
    sr = 2 # sample rate
    period = sec_p_beat / sr
    sampled_score_mel = sample_perf(score_mel,score_mel,score_acc, score_acc, sec_p_beat, beat_p_bar, sr, 1)
    SN = sampled_score_mel.shape[0]
    sampled_score_acc = sample_perf_acc(score_acc, score_acc, sampled_score_mel, sampled_score_mel, sec_p_beat, sr, 1)
    # print("sampled_score_acc", sampled_score_acc, sampled_score_acc.shape)
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
    aligned_perf_mels_fgt = np.zeros((N, 4, L))
    aligned_perf_accs_fgt = np.zeros((M, 4, L))
    missing_melix = []
    missing_accix = []
    missing_accevtix = []
    # all slopes are selfnote-included slopes
    aligned_perf_mels_slopes = np.zeros((N, L))
    aligned_perf_accs_slopes = np.zeros((M, L))
    # sampled perf and score 
    sampled_perf_mels = np.zeros((SN, 5, L))
    sampled_perf_accs = np.zeros((SM, 5, L))
    sampled_perf_mels_slopes = np.zeros((SN, L))
    sampled_perf_accs_slopes = np.zeros((SM, L))
    # median for decoding each songs 
    aligned_median_mels = np.zeros((N, 4, L))
    aligned_median_accs = np.zeros((M, 4, L))
    # results for different method 
    # rslt for only timing stretch, keep previous dy
    aligned_perf_accs_dmsm = np.zeros((M,4,L))
    aligned_perf_mels_dmsm = np.zeros((N,4,L))
    aligned_perf_accs_dssm = np.zeros((M,4,L))
    aligned_perf_mels_dssm = np.zeros((N,4,L))
    # rslt for timing stretch + dy from reference
    aligned_perf_accs_dmscale = np.zeros((M,4,L))
    aligned_perf_mels_dmscale = np.zeros((N,4,L))
    aligned_perf_accs_dsscale = np.zeros((M,4,L))
    aligned_perf_mels_dsscale = np.zeros((N,4,L))
    # index when doing stretch follow
    Begin_ix_acc = []
    Sparse_ix_acc = []
    Begin_ix_mel = []
    Sparse_ix_mel = []
    # rslt for stretch follow sampled perf mel/acc
    aligned_perf_mels_dsssm = np.zeros((N,4,L))
    aligned_perf_accs_dsssm = np.zeros((M,4,L))
    sampled_perf_mels_dsssm = np.zeros((SN,4,L)) # sssm = score stretch sampled melody
    sampled_perf_accs_dsssm = np.zeros((SM,4,L))
    sampled_perf_accs_dsssa = np.zeros((SM,4,L)) # sssm = score stretch sampled acc

    # residuals for different method
    Rsdl_accs_dmsm = np.zeros((M,4,L))
    Rsdl_mels_dmsm = np.zeros((N,4,L))
    Rsdl_accs_dssm = np.zeros((M,4,L))
    Rsdl_mels_dssm = np.zeros((N,4,L))
    Rsdl_accs_dmscale = np.zeros((M,4,L))
    Rsdl_mels_dmscale = np.zeros((N,4,L))
    Rsdl_accs_dsscale = np.zeros((M,4,L))
    Rsdl_mels_dsscale = np.zeros((N,4,L))
    # rsdl for stretch follow sampled perf mel/acc
    Rsdl_mels_dsssm = np.zeros((N,4,L))
    Rsdl_accs_dsssm = np.zeros((M,4,L)) 
    Rsdl_smels_dsssm = np.zeros((SN,4,L))
    Rsdl_saccs_dsssm = np.zeros((SM,4,L))
    Rsdl_saccs_dsssa = np.zeros((SM,4,L))

    # compute the fake ground truth of perf_mel and perf_acc by perfs'
    for i in range(len(perffile_ix)):
        idx = perffile_ix[i]
        aligned_perf_mels_fgt[:, :, idx], missing_melix_new = stretch_fill(score_mel, perf_mels[idx], p)
        missing_melix.append(missing_melix_new)
        aligned_perf_accs_fgt[:, :, idx], missing_accix_new, missing_accevtix_new = get_fgt_acc(score_mel, score_acc, perf_mels[idx], perf_accs[i], p, sec_p_beat)

    # compute the median perf_mel and perf_acc    
    # the overall median
    c_melix, c_accix = get_last_syncix(score_mel, score_acc)
    # print("aligned_perf_accs_fgt", aligned_perf_accs_fgt)
    median_perf_mel = get_median_alignedperf(aligned_perf_mels_fgt[:, :, perffile_ix], c_melix)
    median_perf_acc = get_median_alignedperf(aligned_perf_accs_fgt[:, :, perffile_ix], c_accix)
    median_perf_concat = np.concatenate((median_perf_mel, median_perf_acc), axis=0)
    median_perf_full_ix = np.argsort(median_perf_concat[:, 2])
    median_perf_full = median_perf_concat[median_perf_full_ix]
    
    # round median perf
    median_perf_mel = np.around(median_perf_mel, decimals=4)
    median_perf_acc = np.around(median_perf_acc, decimals=4)
    median_perf_full = np.around(median_perf_full, decimals=4)

    if flag_debug == 0:
        matrix2midi(median_perf_mel, song + '/' + "median_mel.mid")
        matrix2midi(median_perf_acc, song + '/' + 'median_acc.mid')
        matrix2midi(median_perf_full, song + '/' + 'median_full.mid')
        rslt_path = "/Users/luzijie/Desktop/Capstone/method/imitation_learning/basline/results/" + song + '/'
        np.savetxt(rslt_path + 'median_mel.txt', median_perf_mel)
        np.savetxt(rslt_path + 'median_acc.txt', median_perf_acc)
        np.savetxt(rslt_path + 'median_full.txt', median_perf_full)
    
    # median to decode each piece (decoded piece excluded)
    for i in perffile_ix:
        medianfile_ix = np.setdiff1d(perffile_ix, i)
        aligned_median_mels[:, :, i] = get_median_alignedperf(aligned_perf_mels_fgt[:, :, medianfile_ix], c_melix)
        aligned_median_accs[:, :, i] = get_median_alignedperf(aligned_perf_accs_fgt[:, :, medianfile_ix], c_accix)
    
    # compute sampled melody and acc
    p_extra = 1
    for i in perffile_ix:
        sampled_perf_mels[:, :, i] = sample_perf(score_mel, aligned_perf_mels_fgt[:, :, i], score_acc, aligned_perf_accs_fgt[:, :, i], sec_p_beat, beat_p_bar, sr, p_extra)
        sampled_perf_accs[:, :, i] = sample_perf_acc(score_acc, aligned_perf_accs_fgt[:, :, i], sampled_score_mel, sampled_perf_mels[:, :, i], sec_p_beat, sr, p_extra)

    print(np.median(sampled_perf_accs, 2))

    # decode the melody AND accompaniment by median stretch
    # create the tensor to store the result of decoded perf accompany by
    # stretch_follow_acc (median stretching melody)

    # folder to store the (median stretch melody) rslt
    # if flag_debug == 0:
    #     folder_syn_msm = folder_syn + "MSM/"
    #     if os.path.exists(folder_syn_msm):
    #         os.rmdir(folder_syn_msm)
    #     os.makedirs(folder_syn_msm)

    # loop through perffile to do decoding
    for i in perffile_ix:
        # decode the accompaniment by median stretch the melody 
        aligned_perf_accs_dmsm[:, :, i], aligned_perf_accs_dmscale[:, :, i], _, _, _ = \
        stretch_follow_acc(aligned_median_mels[:, :, i], aligned_median_accs[:, :, i], perf_mels[i], p, score_mel, score_acc, 0, sec_p_beat)
        # decode the melody by median stretch the melody
        aligned_perf_mels_dmsm[:, :, i], aligned_perf_mels_dmscale[:, :, i], _, _, _ = \
        stretch_follow(aligned_median_mels[:, :,i], perf_mels[i], p, sec_p_beat)
        # write acc into midi
        # print(aligned_perf_accs_dmsm[:, :, i].shape)
        # print(perf_mels[i][:, 0:4].shape)
        new_matrix = np.append(aligned_perf_accs_dmsm[:, :, i], perf_mels[i][:, 0:4], axis=0)
        rslt_notes_ix = np.argsort(new_matrix[:, 2])
        rslt_notes = new_matrix[rslt_notes_ix]
        # TODO: write midi_acc

    # compute the residual
    st_ix = np.where(aligned_perf_accs_dmsm[:, 0, 0])[0][0]
    Rsdl_accs_dmsm[st_ix:, :, :] = aligned_perf_accs_dmsm[st_ix, :, :] - aligned_perf_accs_fgt[st_ix:, :, :]
    Rsdl_accs_dmscale[st_ix:, :, :] = aligned_perf_accs_dmscale[st_ix:, :, :] - aligned_perf_accs_fgt[st_ix:, :, :]

    st_ix = np.where(aligned_perf_mels_dmsm[:, :, 0])[0][0]
    Rsdl_mels_dmsm[st_ix:, :, :] = aligned_perf_mels_dmsm[st_ix:, :, :] - aligned_perf_mels_fgt[st_ix:, :, :]
    Rsdl_mels_dmscale[st_ix:, :, :] = aligned_perf_mels_dmscale[st_ix:, :, :] - aligned_perf_mels_fgt[st_ix:, :, :]

    # decode the melody AND accompaniment by stretch/scale score
    # create the tensor to store the result of decoded perf accompany by
    # stretch_follow_Acc (score sttretching melody)
    # NOTE: for SS, we use melody fgt to decode. The rsdl_SS and slopes will be used to train higher level model
    if flag_debug == 0:
        # floder to store the (simple stretch melody) rslt
        # TODO
        pass
    
    # fake mono acc and mono sacc
    _, _, _, cevts_stix = notes2cevt(sampled_score_acc)
    mono_sampled_score_acc = sampled_score_acc[cevts_stix, 0:4]
    _, sampled_perf_accst = extract_perfs_info(3, sampled_perf_accs, sampled_perf_accs, sampled_score_acc, sampled_score_acc, p, sec_p_beat, 'n', 0)
    _, _, _, cevts_stix2 = notes2cevt(score_acc)
    mono_score_acc = score_acc[cevts_stix2, 0:4]
    _, perf_accst = extract_perfs_info(3, aligned_perf_accs_fgt, aligned_perf_accs_fgt, score_acc, score_acc, p, sec_p_beat, 'n', 0)

    # loop through perffile to do decoding 
    for i in perffile_ix:
        perf_mels_fgt_wix = np.concatenate((aligned_perf_mels_fgt[:, :, i], np.reshape(np.arange(0, N), (N,1))), axis=1)

        # decode the accompaniment
        # Predict umsampled acc by unsampled mel: stretch + scale
        aligned_perf_accs_dssm[:, :, i], aligned_perf_accs_dsscale[:, :, i], new_Begin_ix_acc, new_Sparse_ix_acc, _ = \
        stretch_follow_acc(score_mel, score_acc, perf_mels_fgt_wix, p, score_mel, score_acc, 0, sec_p_beat)
        Begin_ix_acc.append(new_Begin_ix_acc)
        Sparse_ix_acc.append(new_Sparse_ix_acc)
        # Predict unsapled acc by unsampled acc: self-included slopes
        # we have to creat a fake monophonic acc perf, and feed that in to the melody parameter
        mono_perf_acc = aligned_perf_accs_fgt[cevts_stix2, 0:4, i]
        mono_perf_acc[:, 2] = perf_accst[:, i]
        _, _, _, _, aligned_perf_accs_slopes[:, [i]] = stretch_follow_acc(mono_score_acc, score_acc, mono_perf_acc, p_slope, mono_score_acc, score_acc, 0, sec_p_beat, 1, aligned_perf_accs_fgt[:, :, i])
        # predict unsampled acc by sampled mel: stretch
        aligned_perf_accs_dsssm[:, :, i], _, _, _, _ = stretch_follow_acc(sampled_score_mel, score_acc, \
        sampled_perf_mels[:, :, i], p, sampled_score_mel, score_acc, 1, sec_p_beat)
        # predict sampled acc by sampled mel: stretch
        sampled_perf_accs_dsssm[:, :, i], _, _, _, _ = stretch_follow_acc(sampled_score_mel, sampled_score_acc, \
        sampled_perf_mels[:, :, i], p, sampled_score_mel, sampled_score_acc, 0, sec_p_beat)
        # predict sampled acc by sampled acc: stretch
        # we have to create a fake monophonic sampled acc, and feed that into the melody parametere
        mono_perf_sacc = sampled_perf_accs[cevts_stix, 0:4, i]
        mono_perf_sacc[:, 2] = sampled_perf_accst[:, i]
        sampled_perf_accs_dsssa[:, :, i], _, _, _, _ = stretch_follow_acc(mono_sampled_score_acc, sampled_score_acc, \
        mono_perf_sacc, p_sampleslope, mono_sampled_score_acc, sampled_score_acc, 0, sec_p_beat, 1, sampled_perf_accs[:, : ,i])

        # decode the melody
        

        
        

def main():
    baseline_by_strectch(0,0,0,0,0)

if __name__ == '__main__':
    main()