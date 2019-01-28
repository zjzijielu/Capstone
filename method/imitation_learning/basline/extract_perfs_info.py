import numpy as np
from notes2cevt import notes2cevt
# import matplotlib.pyplot as plot

def extract_perfs_info(DIM, ref_info, target_info, ref_s, target_s, p, sec_p_beat, flag='b', f_anchor=1, v=0):
    # This function fetch target performance's previous p notes/beats' infomation for 
    # each reference cevt. The information is already stored in ref_info or target_info,
    # this function just turn it into the cevts version. See notes t-p-m-n and d-p-m-n
    # 
    # NOTE1: for perfs features. We all extract all perfs' features at once
    # rather than extract them one-by-one

    # input:
    #     Dim: The dimension of the tensor we care about, if ref_info and target_info are tensor. 
    #          In other words we care about ref_info(:,Dim,:) and target_info(:,Dim,3). If
    #          info is rsdl_accs{mel}_SS. Then dim=1 is pitch, 2 is dynamics, 3
    #          is delt_t.
    #     tensor/mat ref_info: the tensor or marix for reference info M * 4 * L
    #                          or M*L
    #     tensor/mat target_info: the tensor for target info. N/M*4*L or N/M*L
    #     mat ref_s: reference score M * 4
    #     mat target_s: target score M/N * 4
    #     int p: the lag parameter. we take previous p notes/beats
    #     char flag: 'n' to take previous p notes' info; 
    #                 'b' to take previous p beats info(we use unsampled perf but sample the feature)
    #     binary f_anchor: 1 is we take in sampled version perf_info and the
    #                      pretarget_info_cevts last note from anchor.
    # NOTE2: we can take in either unsampled or sampled target perf info. 
    #        For unsampled target info: we can use either flag 'n'(note based feature) or 
    #        'b'(beat based feature). f_anchor doesn't make difference. 
    #        For sampled target info, we should always use flag 'n'. f_anchor
    #        now make a difference.

    # output:
    #     tensor pre_targetperfs_info: a tensor to store the refs' previous p
    #     targets' notes/beats info
    #     size almost always CM * (p/p*samplerate) * L. (i,j,k) means for kth performance, ith reference cevts, 
    #     the (p-j+1)th previous target's note/beat rsdl.
    #     refperfs_info: the final reference rsdl we want to get. CM *
    #     L, (i,j) means that for jth performance ith cevts rsdl.
    #     binary anchor. 1 if we use anchor point as the previous target notes
    #                    the reference note.

    ## get the size info and resize the tensor to mat
    # anchor = t means count p beat from previous target note
    # anchor = r means count p beat directly from the reference note

    anchor = 't'
    sample_p_beat = 2
    N = target_info.shape[0]
    M = ref_info.shape[0]
    if target_info.ndim != ref_info.ndim:
        raise ValueError("ref and target dim does not match")
    
    NDIM = target_info.ndim
    if NDIM == 3:
        L = target_info.shape[2]
        target_info = np.reshape(target_info[:, DIM-1, :], (N, L))
        ref_info = np.reshape(ref_info[:, DIM-1, :], (M, L))
    else:
        L = target_info.shape[1]

    if f_anchor == 1:
        t_ix = target_s[:, 4]
    
    # turn target_info into target_info_cevts CN * L
    _, t_cevtsix, t_cevts_st, _ = notes2cevt(target_s)
    CN = len(t_cevts_st)
    target_info_cevts = np.zeros((CN, L))
    
    for i in range(0, CN):
        tmp_target_flag = np.where(t_cevtsix == i)[0]
        # print(tmp_target_flag)
        tmp_part = target_info[tmp_target_flag, :]
        target_info_cevts[i, :] = np.mean(tmp_part, axis=0)

    # turn reference rsdl into cevts-rsdl CM * L
    _, cevtsix, cevts_st, _ = notes2cevt(ref_s)
    CM = len(cevts_st)
    ref_info_cevts = np.zeros((CM, L))

    for i in range(0, CM):
        tmp_ref_flag = np.where(cevtsix == i)[0]
        tmp_part = ref_info[tmp_ref_flag, :]
        ref_info_cevts[i, :] = np.mean(tmp_part, axis=0)

    # construct the output target cevts rsdl tensor CM * p * L
    if flag == 'n':
        pretarget_info_cevts = np.zeros((CM, p, L))
        st_melcevtix = p - 1 # NOTSURE
        if f_anchor == 1:
            st_melcevtix = np.where(t_ix == t_ix[st_melcevtix])[0][-1]

        begin_ix = np.where(cevts_st > t_cevts_st[st_melcevtix])[0][0]
        for i in range(begin_ix, CM):
            pre_target_cevtix = np.where(t_cevts_st < cevts_st[i])[0][-p:]
            if f_anchor == 1:
                shift = pre_target_cevtix[-1] - np.where(t_ix == t_ix[pre_target_cevtix[-1]])[0][0]
                part_output = target_info_cevts[pre_target_cevtix, :]
                pretarget_info_cevts[i, :, :] = part_output
    else: # for flag == b. If it's a sample version, we should use flag == n, because it's faster
        pretarget_info_cevts = np.zeros((CM, p * sample_p_beat+1, L))
        t_begin_ix = np.where(t_cevts_st >= (t_cevts_st[0] + p * sec_p_beat))[0][0]
        begin_ix = np.where(cevts_st > t_cevts_st[t_begin_ix])[0][0]
        for i in range(begin_ix, CM):
            if anchor == 't': # use previous target as anchor
                tailix = np.where(t_cevts_st < cevts_st[i])[0][-1]
                local_et = t_cevts_st[tailix]
            else: # use current ref as anchor
                local_et = cevts_st[i]

            # starting time is local end time - pbeats
            local_st = local_et - p * sec_p_beat
            # sample the interval by half a beat (if sample_p_beat = 2)
            local_grid = np.arange(local_st, local_et, sec_p_beat / sample_p_beat)
            for k in range(L):
                # np.interp(newx, x, y)
                local_contour = np.interp(local_grid, t_cevts_st, target_info_cevts[:, k])
                pretarget_info_cevts[i, :, k] = local_contour
    
    # TODO: visualize

    return pretarget_info_cevts, ref_info_cevts