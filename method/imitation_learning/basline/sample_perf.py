import math
import numpy as np

def intersect_mtlb(a, b):
    a1, ia = np.unique(a, return_index=True)
    b1, ib = np.unique(b, return_index=True)
    aux = np.concatenate((a1, b1))
    aux.sort()
    c = aux[:-1][aux[1:] == aux[:-1]]
    return c, ia[np.isin(a1, c)], ib[np.isin(b1, c)]

def sample_perf(score, perf, score_acc, perf_acc, sec_p_beat, beat_p_bar, sr, p):
    '''
    this function sample the monophonic performance. By using exsiting notes
    as anchor notes, do regression and fill in the sample notes
    input:
    score: N * 5, score of mel
    perf: n * 4/5, perf of mel, could also be score version to compute sampled score.
    score_acc, M * 5, score of acc
    perf_acc, m * 5/4, perf of acc
    sec_p_beat: second per beat in for the score
    sr: sample rate, sample per beat
    p: if the nearest previous&next melody note are the anchor, 
        how long extra is the regression list the sample performance
        unit is beat. if p = 0, we just use nearest two notes.
    flag_dy: how to sample dynamics
    outputL
    sampled_perf: SN * 5. the 5th cln is the melody index. SN is the new
    length of the sampled performance
    '''

    # test
    # p = 4
    # sec_p_beat = 1
    # beat_p_bar = 3
    # sr = 2

    max_measure_gap = 1.5
    getslop = lambda x, y: np.sum((y-np.mean(y))*(x-np.mean(x)))/np.sum(np.power(x-np.mean(x), 2))
    getinter = lambda s, x, y: np.mean(y)-s*np.mean(x)
    # v-t map interpolation of rubato assuming linear v change
    # t is realtime, v is beat/sec, v-t area is beat.
    getti = lambda v1, vn, n, i, tn: tn*(-v1*n+math.sqrt(v1**2 * n**2 + n*i*(vn**2-v1**2))) / (n*(vn-v1)) 
    
    # preprocess the perf, let it be in order as score
    N = score.shape[0]
    aligned_perf = np.zeros((N, 4))
    if perf_acc.shape[1] == 5: # raw perf with index
        # print(perf_acc[:,4])
        aligned_perf[(perf[:,4].astype(int)-1),:] = perf[:,0:4]
    else:
        aligned_perf = perf
    M = score_acc.shape[0]
    aligned_perf_acc = np.zeros((M,4))
    if perf_acc.shape[1] == 5: # raw perf with index
        aligned_perf_acc[(perf_acc[:,4].astype(int)-1),:] = perf_acc[:,0:4]
    else:
        aligned_perf_acc = perf_acc
    
    # construct sample_perf, put existing perf notes in it
    # compute the resolution and begin end ix
    period = sec_p_beat / sr
    index = np.where(np.mod(score[:,2], period) == 0)
    begin_ix = index[0][0]
    end_ix = index[0][-1]
    # keep the sampled score starting time
    sampled_scoretime = np.arange(score[begin_ix, 2], score[end_ix, 2], period)
    SN = len(sampled_scoretime)
    sampled_perf = np.zeros((SN, 5))
    _, ix, s_ix = intersect_mtlb(score[:,2], sampled_scoretime)
    sampled_perf[s_ix, 0:4] = aligned_perf[ix, :]
    # put in existing notes ix
    sampled_perf[s_ix, 4] = ix
    # print("sampled perf:", sampled_perf)
    # keep a copy of raw sampled score
    sampled_score = np.zeros((SN, 5))
    # the variable keep unsampled for sample notes
    sampled_score[s_ix,:] = score[ix,:]
    # print("sampled_score", sampled_score)
    # fill the large gaps by adopting perf acc notes
    osix = np.where(sampled_perf[:,0] == 0)[0]
    # print(sampled_score)
    for k in range(len(osix)):
        i = osix[k]
        # get indices around the missing sample note
        # the anchor note's index right before/after the missing sample note
        
        # print(i, np.where(sampled_score[i:, 0]))
        # print(sampled_score[i:, 0])
        try:
            aft_six = np.where(sampled_score[i:,0])[0][0] + i
        except:
            aft_six = i
        pre_six = np.where(sampled_score[0:i,0])[0][-1]
        # print("aft_six", aft_six)
        # if large gap, fill in information
        # print(aft_six, pre_six, max_measure_gap * beat_p_bar * sec_p_beat / period)
        if ((aft_six - pre_six) >= (max_measure_gap * beat_p_bar * sec_p_beat / period)) and np.any(score_acc[:,3] == sampled_score[i, 3]):
            # index
            print("large gap")
            sampled_perf[i, 4] = sampled_perf[pre_six, 4]
            # others follow the corresponding acc highest pitch note
            match_accix = np.where(score_acc[:,2] == sampled_scoretime[i])[0]
            match_perf_acc = aligned_perf_acc[match_accix, :]
            match_score_acc = score_acc[match_accix, :]
            lix_pitch = np.max(match_score_acc[:,0])
            # in case the highest pitch in perf is missing
            lix_other = np.max(match_perf_acc[:,0])
            sampled_perf[i, 0] = match_score_acc[lix_pitch, 1]
            sampled_perf[i, 1:4] = match_perf_acc[lix_other, 1:4]

    # fill other zero rows
    osix = np.where(sampled_perf[:,0] == 0)[0]
    for k in range(len(osix)):
        i = osix[k]
        # get indices around the missing sample note
        # the anchor note's index right before/after the missing sample note
        try:
            aft_six = np.where(sampled_score[i:,0])[0][0] + i
        except:
            aft_six = i 
        pre_six = np.where(sampled_score[0:i,0])[0][-1]
        # head ix is the starting point of regression (only for pre_six < 2 case)
        if pre_six - p*sr <= 0:
            head_six = 0
        else:
            head_six = pre_six - p*sr
        # fill in the information
        # fill pitch and index keep the same
        sampled_perf[i, [0,1,4]] = sampled_perf[pre_six, [0,1,4]]
        # dynamics keep decay/linear interpolation
        # fill starting time
        if pre_six < 1:
            non_o_six = np.setdiff1d(np.arange(head_six, aft_six), osix)
            non_o_ix = sampled_perf[non_o_six, 4]
            st_xs = score[non_o_ix, 2]
            st_ys = aligned_perf[non_o_ix, 2]
            slop = getslop(st_xs, st_ys)
            inter = getinter(slop, st_xs, st_ys)
            sampled_perf[i, 3] = sampled_scoretime[i] * slop + inter
        else: # todo: consider larger interval for v1 and vn?
            if aft_six == len(sampled_perf) - 1:
                n = aft_six - pre_six + 1
                tn = sampled_perf[pre_six, 3] - sampled_perf[pre_six, 2]
            else:
                n = aft_six - pre_six
                tn = sampled_perf[aft_six, 2] - sampled_perf[pre_six, 2]
            j = i - pre_six
            assert(tn != 0)
            v1 = period / (sampled_score[pre_six, 2] - sampled_perf[pre_six-1, 2])
            vn = 2 * n * period / tn - v1
            if v1 != vn:
                ti = getti(v1, vn, n, j, tn)
            else:
                ti = tn * j / n
            sampled_perf[i, 2] = ti + sampled_perf[pre_six, 2]
        # fill et by duration, carry the orignal perf dur/ score dur
        pre_six = pre_six.astype(int)
        pre_ix = sampled_perf[pre_six, 4].astype(int)
        dur_perf = sampled_perf[pre_six, 3] - sampled_perf[pre_six, 2]
        dur_score = score[pre_ix, 3] - score[pre_ix, 2]
        dur_ratio = dur_perf / dur_score
        sampled_perf[i, 3] = sampled_perf[i, 2] + dur_ratio * period
    
    # replace the original sample note's ending time
    for i in range(len(sampled_perf) - 1):
        if sampled_perf[i, 3] > sampled_perf[i+1, 2]:
            sampled_perf[i, 3] = sampled_perf[i+1, 2]

    return sampled_perf