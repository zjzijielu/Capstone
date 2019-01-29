import numpy as np

def stretch_follow(score, perf, p, sec_p_beat=1, flag_self=0, flag_anchor=0):
    # ## this function do the simple stretching vt-rt time map to predict the timing of 
    # a performance. We assume that we know that score note each perf note matches to
    # so that we don't run score_follow here. (we have index in perf file)
    # 
    # input: (both score and perf are aligned)
    #     score: the reference performance version. Usually is the median perf.
    #     size of N * 4
    #   
    #     perf: the performance to be followed and predicted, size is as
    #           aligned_perf. M * 4, where M <= N
    #     p: the lag parameter. the previous melody as a cursor, use its
    #        p previous beats' notes (include head and tail) as regression
    #        list.
    #     sec_p_beat: sec_p_beat * p would be the lag
    #     flag_self: binary. when it is 1, the regression list contain itself.
    #     flag_anchor: binary. 0 means we use last melody note (either sampled/not sampled)
    #                  as the end of regression list. 1 means use last anchor
    #                  note as end of regression list.
    #     NOTE: use last anchor note will leads more similar result as using
    #     unsampled melody as reference to do the stretch.
    # output:
    #       rslt_stretch : the predicted result of performance melody
    #                       by using simple rt-vt linear mapping. Size N * 4
    #       rslt_scale : stretch is only in timing, scale also predict
    #                            dynamic by rt-vt linear mapping. Size N * 4
    #      begin_ix: the begin_ix of acc that start to decode
    #      sparse_ix: the accix that don't have enough points for regression
    #      slopes: the slopes of the melody note. P beat, self-include. N*1
    ## NOTE: if reslt size is same as score. If the M < N, the perf_aligned var will have zero
    # entries. But we just use non-zero entries to do prediction.

    getslop = lambda x, y: np.sum((y-np.mean(y))*(x-np.mean(x)))/np.sum(np.power(x-np.mean(x), 2))
    getinter = lambda s, x, y: np.mean(y)-s*np.mean(x)

    # read and preprocess inputs
    score = score[:, 0:4]
    N = score.shape[0]
    perf_aligned = np.zeros((N, 4))
    if perf.shape[1] == 5 and len(perf[:, 4]) == len(np.unique(perf[:, 4])): # raw perf with index
        perf_aligned[perf[:, 4].astype(int)-1, :] = perf[:, 0:4]
    else: # if not unique index column it's a fgt or a sampledversion
        perf_aligned = perf[:, 0:4]
    # if it is a sampled version and use anchor do regression, ix is useful, otherwise not
    if flag_anchor == 1:
        if perf.shape[1] == 4:
            raise ValueError("an unsampled version should not have flag_anchor == 1")
        ix = perf[:, 4]

    # initiate outputs
    rslt_stretch = np.zeros((N, 4))
    rslt_scale = np.zeros((N, 4))
    slopes = np.zeros((N, 1))

    # find begin ix
    st = score[0, 2] + p * sec_p_beat
    begin_ix = np.where(score[:, 2] > st)[0][0]
    sparse_ix = []
    
    # loop through the score
    for i in range(begin_ix, N):
        # find the previous anchor melody note
        if flag_anchor == 0: # if not anchor note, it's simply the previous one
            pre_ix = i - 1
        else:
            pre_ix = np.where(ix == ix[i-1])[0][0]
        
        if flag_self ==0: # exclude self point case
            head_ix = np.where(score[:, 2] >= (score[pre_ix, 2] - p * sec_p_beat))[0][0]
            non_o_ix = np.where(perf_aligned[head_ix:pre_ix, 0])[0] + head_ix
        else: # include self point case
            head_ix = np.where(score[:, 2] >= score[i, 2] - p * sec_p_beat)[0][0]
            non_o_ix = np.where(perf_aligned[head_ix:i, 0])[0] + head_ix
        if len(non_o_ix) < p:
            non_o_ix = np.where(perf_aligned[0:i-1, 2])[0][-p:]
            sparse_ix.append(i)
        
        st_xs = score[non_o_ix, 2]
        st_ys = perf_aligned[non_o_ix, 2]
        slop = getslop(st_xs, st_ys)
        inter = getinter(slop, st_xs, st_ys)
        # keep the pitch
        rslt_stretch[i, 0] = score[i, 0]
        # make the velocity same as last perf mel
        rslt_stretch[i, 1] = perf_aligned[max(i-1, 0), 1]
        # stretch strating time and endig time according to 
        rslt_stretch[i, 2:4]= score[i, 2:4] * slop + inter
        # fill in the prediction perf by scale (only second cln is different)
        rslt_scale[i, :] = rslt_stretch[i, :]
        rslt_scale[i, 1] = score[i, 1]
        # write down the slope of this melody note
        slopes[i] = slop

    return rslt_stretch, rslt_scale, begin_ix, sparse_ix, slopes