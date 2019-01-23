import numpy as np
import math
from notes2cevt import notes2cevt
from utils import *

def sample_perf_acc(score, perf, score_smel, perf_smel, sec_p_beat, sr, p_extra):
    #  this function sample the monophonic performance. By using exsiting notes
    #  as anchor notes, do regression and fill in the sample notes
    #   input:
    #     score: M * 5, score for acc
    #     perf: M * 4/5, perf acc,could also be score version to compute sampled score.
    #     sec_p_beat: second per beat in for the score
    #     sr: sample rate, sample per beat
    #     p_extra: if the nearest previous&next melody note are the anchor, 
    #        how long extra is the regression list the sample performance
    #        unit is beat. if p = 0, we just use nearest two notes.
    #     flag_dy: how to sample dynamics
    #   output:
    #     sampled_perf_cevts: SCM * 5. the 5th cln is the melody index. SCM is the new
    #     length of the sampled performance

    max_periods_gap = 4
    getslop = lambda x, y: np.sum((y-np.mean(y))*(x-np.mean(x)))/np.sum(np.power(x-np.mean(x), 2))
    getinter = lambda s, x, y: np.mean(y)-s*np.mean(x)
    # v-t map interpolation of rubato assuming linear v change
    # t is realtime, v is beat/sec, v-t area is beat.
    getti = lambda v1, vn, n, i, tn: tn*(-v1*n+math.sqrt(v1**2 * n**2 + n*i*(vn**2-v1**2))) / (n*(vn-v1)) 
    # preprocess the perf, let it be in order as score
    N = score.shape[0]
    aligned_perf = np.zeros((N, 4))
    if perf.shape[1] == 5: # raw perf with index
        aligned_perf[(perf[:,4].astype(int)-1),:] = perf[:,0:4]
    else:
        aligned_perf = perf
    
    # compute sampled score_cevt time and reserve the anchor notes
    cevts, cevtsix, cevts_st, cevts_stix = notes2cevt(score)
    cevtix2ix = lambda c_ix: np.arange(cevts_stix[c_ix], cevts_stix[c_ix] + len(cevts[c_ix]))
    # compute the resoluton and begin/end index    
    period = sec_p_beat / sr
    index = np.where(np.mod(cevts_st, period) == 0)
    begin_ix = index[0][0]
    end_ix = index[0][-1]
    # sampled score cevts starting time
    sampled_scorecevttime = np.arange(cevts_st[begin_ix], cevts_st[end_ix], period).T
    SCM = len(sampled_scorecevttime)
    # get the reserved perf and score
    ix = ismember(score[:, 2], sampled_scorecevttime)
    score_reserve = score[ix, :]
    perf_reserve = np.concatenate((aligned_perf[ix, :], score_reserve[:, 4:]), axis=1)   
    # perf_reserve = [aligned_perf[ix, :], score_reserve[:, 4]] # NOT SURE
    # NOTE: 1. perf_reserve could be shorter than M since not every cevt is
    # anchor (non anchor cevt get thrown). 
    # 2. there could be zero rows if perf is missing an anchor note

    # loop throught missing sampled_cevt index
    perf_add = np.zeros((0, 5))
    osix = np.setdiff1d(sampled_scorecevttime, score_reserve[:, 2])
    osix = setdiff(sampled_scorecevttime, score_reserve[:, 2])
    for i in range(len(osix)):
        # get indices around the missing sample note
        # the anchor note's cevt index right before/after the missing sample note
        # current sampled score cevt time
        cur_stime = sampled_scorecevttime[osix[i].astype(int)]
        # right previous and after cevt ix (cix)
        aft_cix = np.where((cevts_st > cur_stime) & (np.mod(cevts_st, period) == 0))[0][0] 
        pre_cix = np.where((cevts_st < cur_stime) & (np.mod(cevts_st, period) == 0))[0][-1]
        # head ix is the starting point of regression for the beginning case 
        head_cix = np.where(cevts_st <= (cevts_st[pre_cix] - p_extra * sec_p_beat))[0]
        if head_cix.size == 0:
            head_cix = 1
        else:
            head_cix = head_cix[-1]
        # fill in the information        
        new = np.zeros((len(cevts[pre_cix]), 5))
        # fill pitch and index keep the same
        row_idx = np.array(cevtix2ix(pre_cix))
        new[:, [0, 1]] = aligned_perf[row_idx[:, None], [0, 1]]
        # fill in the starting times
        if pre_cix < 1:
            cix = np.arange(head_cix, aft_cix)
            st_xs = cevts_st[cix] # score cevts times
            st_ys = np.zeros((len(st_xs), 1))
            for j in range(len(st_ys)):
                st_ys[j] = np.mean(aligned_perf[cevtix2ix(cix[j]), 3], axis=0)
            local_ocix = np.where(st_ys == 0)[0]
            if local_ocix.size != 0:
                st_xs[local_ocix] = []
                st_ys[local_ocix] = []
            slop = getslop(st_xs, st_ys)
            inter = getinter(slop, st_xs, st_ys)
            new[:, 2] = sampled_scorecevttime[i] * slop + inter
        else:
            # n is the sample numbers of the gap between pre_cix and aft_cix
            n = (cevts_st[aft_cix] - cevts_st[pre_cix]) / period
            # if the gap is large, then just adopt sampled perf_mel's timings
            if n >= max_periods_gap and np.any(score_smel[:, 2] == cur_stime):
                new[:, 2] = perf_smel[np.where(score_smel[:, 2] == cur_stime), 2]
            else:
                j = (cur_stime - cevts_st[pre_cix]) / period
                tn = np.mean(aligned_perf[cevtix2ix(aft_cix), 2], axis=0) - np.mean(aligned_perf[cevtix2ix(pre_cix), 2], axis=0)
                v1 = (cevts_st[pre_cix] - cevts_st[pre_cix-1]) / ((np.mean(aligned_perf[cevtix2ix(pre_cix), 2], axis=0) - (np.mean(aligned_perf[cevtix2ix(pre_cix-1), 2], axis=0))))
                # debug vn
                vn = 2 * n * period / tn - v1
                if v1 != vn:
                    ti = getti(v1, vn, n, j, tn)
                else:
                    ti = tn * j / n
                new[:, 2] = ti + np.mean(aligned_perf[cevtix2ix(pre_cix), 2], axis=0)
    
        # fill in durations
        dur_ratio = np.divide((aligned_perf[cevtix2ix(pre_cix), 3] - (aligned_perf[cevtix2ix(pre_cix), 2])), (score[cevtix2ix(pre_cix), 3] - score[cevtix2ix(pre_cix), 2]))
        new[:, 3] = new[:, 2] + period * dur_ratio
        
        # fill in the index (keep the last index of pre cevts's last note)
        new[:, 4] = cevts_stix[pre_cix] + len(cevts[pre_cix]) - 1
        # add the new info into perf_add
        # print(perf_add.shape)
        # print(new.shape)
        perf_add = np.append(perf_add, new, axis=0)

    # print("append", np.append(perf_add, perf_reserve, axis=0))
    combined = np.append(perf_add, perf_reserve, axis=0)
    sampled_perf_ix = np.argsort(combined[:, 3])
    sampled_perf_acc = combined[sampled_perf_ix, :]
    if np.where(sampled_perf_acc[:, 0] == 0)[0].shape[0] != 0:
        raise ValueError("Result of Sampling has pitch value 0 in it.")

    return sampled_perf_acc