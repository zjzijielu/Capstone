import numpy as np

def stretch_follow_acc(ref_mel, ref_acc, perf_mel, p, score_mel, score_acc, flag_anchor,sec_p_beat, flag_self=0, perf_acc_fgt=None):
    # ## this function do the simple stretching vt-rt time map (of score melody and perf melody)
    # to predict the timing of 
    # a performance accompaniment. We assume we know what score melody note each perf melody note matches to
    # so that we don't run score_follow here. (we have index in perf file)
    # ever since version 306 we modify note-index based regression into
    # beat-based regression
    # 
    # input: (both score and perf are aligned)
    #     ref_mel: the reference performance version. a quantized score or the median perf.
    #     size of N * 4
    #     ref_acc: the reference performance version of accompaniment. size of O * 4. O + N 
    #                will be the total size of score note

    #     perf_mel: the performance to be followed and predicted, size is as
    #           aligned_perf. n * 5/4, where n <= N
    #     p: the lag parameter. use previous mel note's p previous
    #        beat to do regression.(include previous mel note and 
    #        previous mel note's -p beat note)
    #     flag_anchor: binary. 0 means we use last melody note (either sampled/not sampled)
    #                  as the end of regression list. 1 means use last anchor
    #                  note as end of regression list.
    #     flag_self: binary, set to 1 to include a note itself for regression
    #     perf_acc_fgt: when flag_self is 1, we need a fake ground truth of
    #                   perf_acc. 

    # output:
    #     rslt_stretch (perf_acc): the predicted result of performance accompaniment
    #                       by using simple rt-vt linear mapping. Size M * 4
    #     rslt_scale (perf_acc): stretch is only in timing, scale also predict
    #                            dynamic by rt-vt linear mapping. Size M * 4
    #      begin_ix: the begin_ix of acc that start to decode
    #      sparse_ix: the accix that don't have enough points for regression
    #      slopes: the slopes of mel perfs notes within p beats(when flag_self==1 selfnote
    #              included to compute the slope)
    ## NOTE: if reslt size is same as score. If the n < N, the perf_aligned var will have zero
    # entries. But we just use non-zero entries to do prediction.

    getslop = lambda x, y: np.sum((y-np.mean(y))*(x-np.mean(x)))/np.sum(np.power(x-np.mean(x), 2))
    getinter = lambda s, x, y: np.mean(y)-s*np.mean(x)
    
    # define order flag
    order_flag = 's'
    if order_flag == 'r':
        score_mel = ref_mel
        score_acc = ref_acc
    #     order_flag: 's' use score order to do regression.
    #                 'r' use reference order to do regression.
    # NOTE: it's unclear which one should be used is a fair compairsion with
    #        ARD. 'r' leads to more accurate result by taking advantage of same
    #        idea time different real time cases. But ARD could use this kind
    #        of information as well.
    # get the perf_mel in order, align the score
    ## read and preprocess inputs

    ref_mel = ref_mel[:, 0:4]
    N = ref_mel.shape[0]
    perf_mel_aligned = np.zeros((N, 4))
    if perf_mel.shape[1] == 5 and len(perf_mel[:, 4]) == len(np.unique(perf_mel[:, 4])): # a raw perf with index
        perf_mel_aligned[(perf_mel[:, 4]-1).astype(int), :] = perf_mel[:, 0:4]
    else: # not unique index cln, it's a fgt or a sampled version
        perf_mel_aligned = perf_mel[:, 0:4]
    if flag_anchor == 1: # if it is a sampled version and use last anchor reg, ix is useful, otherwise not 
        if perf_mel.shape[1] == 4:
            print('an unsampled version should not have flag_anchor == 1')
            return
        ix = perf_mel[:, 4]
    M = ref_acc.shape[0]
    # initialize outputs
    slopes = np.zeros((M, 1))
    rslt_stretch = np.zeros((M, 4))
    rslt_scale = np.zeros((M, 4))
    
    # find the first acc to predict
    st = score_mel[0, 2] + p * sec_p_beat
    begin_ix = np.where(score_acc[:, 2] > st)[0][0]
    sparse_ix = []
    # loop through the ref_acc
    for i in range(begin_ix, M):
        # find melody notes for regression list
        # find the previous (right before) the score_acc note, (end of regression list)
        pre_melix = np.where(score_mel[:, 2] < score_acc[i, 2])[0][-1]
        if flag_anchor > 0: # find the previous anchor melody note
            pre_melix = np.where(ix == ix[pre_melix])[0][0]
        # find the melix of ref_mel that the regression list should begin from
        if flag_self == 1: # for self-include case
            head_melix = np.where(score_mel[:, 2] >= (score_acc[i, 2] - p*sec_p_beat))[0][0]
        else: # for self-exclude case
            head_melix = np.where(score_mel[:, 2] >= (score_mel[pre_melix, 2] - p*sec_p_beat))[0][0]
        # deal with zero case: find the non-zero index of perf
        non_o_melix = np.where(perf_mel_aligned[head_melix:pre_melix, 0]) + head_melix - 1
        # if melody list is too short based on p beat, we do 4 notes
        if len(non_o_melix) < p:
            non_o_melix = np.where(perf_mel_aligned[0:pre_melix])
            sparse_ix.append(i)
        st_xs = ref_mel[non_o_melix, 2]
        st_ys = perf_mel_aligned[non_o_melix, 2]
        # if flag_self is 1, we include the perf_acc note itself into the list
        if flag_self == 1:
            st_xs = np.append(st_xs, score_acc[i, 2], axis=0)
            st_ys = np.append(st_ys, perf_acc_fgt[i, 2], axis=0)
        # fill in the prediction perf by stretch
        st_slop = getslop(st_xs, st_ys)
        st_inter = getinter(st_slop, st_xs, st_ys)
        # keep the pitch
        rslt_stretch[i, 0] = ref_acc[i, 0]
        # make the velocity same as last perf mel
        rslt_stretch[i, 1] = perf_mel_aligned[max(pre_melix, 0), 1]
        # stretch starting time and ending time according to 
        rslt_stretch[i, 2:4] = ref_acc[i, 2:4] * st_slop + st_inter

        # fill in the prediction perf by scale (only second cln is different)
        rslt_scale[i, :] = rslt_stretch[i, :]
        rslt_scale[i, 1] = ref_acc[i, 1]
        # write down the slope of this perf_acc note, self included
        slopes[i] = st_slop
    
    return rslt_stretch, rslt_scale, begin_ix, sparse_ix, slopes
        