import numpy as np 
from stretch_fill_acc import stretch_fill_acc
from stretch_follow_acc import stretch_follow_acc
from notes2cevt import notes2cevt
from delete_extreme_cevt_rsdls import delete_extreme_cevt_rsdls
from utils import *

def get_fgt_acc(score_mel, score_acc, perf_mel, perf_acc, p, sec_p_beat):
    # this function fix the misalginment of polyalign and fill in the missing notes of perf acc
    # 1. call stretch_fill_acc to get a full perf_accs
    # 2. based on this do stretch_follow_acc, detect miss alginment, delete missing notes
    # 3. call stretch_fill_acc again to get the final perf_acc

    DEFAULT_MAXSPAN = .3
    # non_o = lambda x: x != 0
    
    # 1. call stretch fill to get a raw fgt
    perf_acc_aligned_filled, oaccix, oaccevtix = stretch_fill_acc(score_mel, score_acc, perf_mel, perf_acc, p, sec_p_beat)
    # 2. delete missing alignment of the polyalign
    misalign_accix = []
    # stretch follow
    rslt_stretch, _, st_ix, _, _ = stretch_follow_acc(score_mel, score_acc, perf_mel, p, score_mel, score_acc, 0, sec_p_beat)
    # compute the residual
    M = perf_acc_aligned_filled.shape[0]
    rsdl_stretch = np.zeros((M, 4))
    rsdl_stretch[st_ix:, :] = rslt_stretch[st_ix:, :] - perf_acc_aligned_filled[st_ix:, :]
    # compute cevts rsdl and find the outlier
    cevts, cevtsix, cevts_st, cevts_stix = notes2cevt(score_acc)
    CM = len(cevts_st)
    last_cevt_rsdl = 0
    for i in range(0, CM):
        cur_cevt_flg = np.where(cevtsix == i)[0]
        cur_cevt = rsdl_stretch[cur_cevt_flg, 2]
        if i == 0 or i == CM-1:
            span = DEFAULT_MAXSPAN
        else:
            pre_pcevt_t = np.mean(np.nonzero(perf_acc_aligned_filled[cevts_stix[i-1]:cevts_stix[i]-1, 2])[0])
            next_pcevt_t = np.mean(np.nonzero(perf_acc_aligned_filled[cevts_stix[i]:cevts_stix[i+1]+len(cevts[i+1])-1, 2])[0])
            span = (next_pcevt_t - pre_pcevt_t) / 2
        new_rsdls, retain_localix = delete_extreme_cevt_rsdls(cur_cevt, last_cevt_rsdl, span)
        # print("retain_localix", retain_localix, "\tcevtstix", cevts_stix[])

        # the deleted accix
        delete_localix = np.setdiff1d(np.arange(0, len(cur_cevt)), retain_localix)
        if delete_localix.size != 0:
            delete_accix = cevts_stix[i] + delete_localix
            misalign_accix = np.append(misalign_accix, delete_accix, axis=0)
        
        # assign the last_rsdls to be the mean of current rsdl
        last_cevt_rsdl = np.mean(new_rsdls)
    
    # true misaligned one doesn't include oaccix (which perf_acc doesn't even have)
    misalign_accix = np.setdiff1d(misalign_accix, oaccix)
    # delete corresponding misaligned rows of perf_accs
    _, d_rowix = ismember_ix(misalign_accix, perf_acc[:, 4])
    perf_acc_fixed = perf_acc
    if d_rowix != []:
        perf_acc_fixed = np.delete(perf_acc_fixed, d_rowix, axis=0)
    
    # print("perf_acc_fixed: ", perf_acc_fixed)
    # 3. call stretch fill again to get final fgt
    aligned_perf_acc_fgt, missing_accix, missing_accevtix = stretch_fill_acc(score_mel, score_acc, perf_mel, perf_acc_fixed, p, sec_p_beat)

    if np.array_equal(missing_accevtix, oaccevtix) != True:
        print("missing_accevtix: ", missing_accevtix)
        print("missing_accix", missing_accix)
        print("oaccix", oaccix)
        raise ValueError("***bug: whole cevts delete")
    
    # print(aligned_perf_acc_fgt)
    return aligned_perf_acc_fgt, missing_accix, missing_accevtix