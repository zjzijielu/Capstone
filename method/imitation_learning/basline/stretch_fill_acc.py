import numpy as np
from notes2cevt import notes2cevt
from utils import *

def stretch_fill_acc(score_mel, score_acc, perf_mel, perf_acc, p, sec_p_beat):
    #  this function smooth out the 0 cases in mel_t by playing liner interpolation
    #  of the previous 2 beat + afterwards 2 beat (4 in total). first try to do
    #  linear interpolation just based on melody. If melody notes is less than
    #  MIN_MEL_REGNUM then we add surrounding acc notes in the interpolation
    #  list.
    #  input:
    #     same as stretch follow
    #  output:
    #     perf_aligned_filled: the new perf_t after linear interpolation
    #     missing_accix: which accix are missing
    #     missing_accevtix: which entire accevt are missing

    # get the perf in order, align the score_mel and score_acc
    MIN_MEL_REGNUM = 4
    # align the melody
    score_mel = score_mel[:, 0:4]
    N = score_mel.shape[0]
    perf_mel_aligned = np.zeros((N, 4))
    melix = perf_mel[:, 4] - 1
    perf_mel_aligned[melix.astype(int), :] = perf_mel[:, 0:4]
    # print("perf_mel_aligned", perf_mel_aligned)
    # align the accompany
    score_acc = score_acc[:, 0:4]
    M = score_acc.shape[0]
    perf_acc_aligned = np.zeros((M, 4))
    accix = perf_acc[:, 4] - 1
    perf_acc_aligned[accix.astype(int), :] = perf_acc[:, 0:4]
    # get cevts for acc
    cevts, cevtsix, cevts_st, cevts_stix = notes2cevt(score_acc)
    CM = len(cevts_st)

    # find which perf_acc_aligned is missing
    oaccix = np.where(perf_acc_aligned[:, 0] == 0)[0]
    oacc_cevtix = []
    # loop throught the missing perf_acc_aligned
    getslop = lambda x, y: np.sum((y-np.mean(y))*(x-np.mean(x)))/np.sum(np.power(x-np.mean(x), 2))
    getinter = lambda s, x, y: np.mean(y)-s*np.mean(x)
    for i in range(len(oaccix)):
        # find if there are score_acc notes that happen at the same time
        idx = oaccix[i]
        score_cevt_accix = np.where(score_acc[:, 2] == score_acc[idx, 2])[0]
        perf_cevt_accix = setdiff(score_cevt_accix, oaccix)
        # if yes, use the perf_acc average time as the timing. Also use
        # the average velocity as velocity
        if perf_cevt_accix.shape[0] != 0:
            perf_acc_aligned[idx, 1] = np.sum(perf_acc_aligned[perf_cevt_accix, 1]) / len(perf_cevt_accix)
            perf_acc_aligned[idx, 2:4] = np.median(perf_acc_aligned[perf_cevt_accix, 2:4], axis=0)
        # otherwise we have to do regression filling based on perf_mel and perf_acc notes
        else:
            # add the cevtix into missing cevtsix
            oacc_cevtix.append(cevtsix[idx])
            # get the melix for regression list 
            # find the score_mel melix that is right before and after missing perf_acc
            try:
                rpre_melix = np.where(score_mel[:, 2] < score_acc[idx, 2])[0][-1]
            except:
                rpre_melix = np.where(score_mel[:, 2] < score_acc[idx, 2])[0]
            try:
                raft_melix = np.where(score_mel[:, 2] > score_acc[idx, 2])[0][0]
            except:
                rpre_melix = np.where(score_mel[:, 2] < score_acc[idx, 2])[0]
            # the acc may come before melody or after 
            # it seems just use acc to fill is a better option
            if rpre_melix == []:
                rpre_melix = raft_melix 
            elif raft_melix == []:
                raft_melix = rpre_melix
            
            # find the head and tail score_mel melix for regression list
            head_melix = np.where(score_mel[:, 2] >= (score_mel[rpre_melix, 2] - p/2*sec_p_beat))[0][0]
            tail_melix = np.where(score_mel[:, 2] <= (score_mel[raft_melix, 2] + p/2*sec_p_beat))[0][-1] 
            # deal with zero case: find the non-zero index of perf
            non_ohead_melix = np.where(perf_mel_aligned[head_melix:rpre_melix, 0] != 0)[0] + head_melix - 1
            non_otail_melix = np.where(perf_mel_aligned[raft_melix:tail_melix, 0] != 0)[0] + raft_melix - 1
            non_o_melix = np.append(non_ohead_melix, non_otail_melix, axis=0)
            # form the regression and fill
            st_xs = score_mel[non_o_melix, 2]
            st_ys = perf_mel_aligned[non_o_melix, 2]
            # if mel notes note enough, add the acc cevtix for regression list
            if len(non_o_melix) < MIN_MEL_REGNUM:
                # find acc cevtix that is right before and after missing pref_acc
                rpre_acc_cevtix = np.where(cevts_st < score_acc[idx, 2])[0][-1]
                raft_acc_cevtix = np.where(cevts_st > score_acc[idx, 2])[0][0]
                # find the head and tail score acc cevtix for regression list
                head_accevtix = np.where(cevts_st >= cevts_st[rpre_acc_cevtix] - p/2*sec_p_beat)[0][0]
                tail_accevtix = np.where(cevts_st <= cevts_st[raft_acc_cevtix] + p/2*sec_p_beat)[0][-1]
                # acc_cevtix for regression
                acc_cevtix = np.append(np.arange(head_accevtix, rpre_acc_cevtix), np.arange(raft_acc_cevtix, tail_accevtix))
                # according to acc_cevtix get score_acc cevts_st time and perf_acc cevt median time
                cevts_st = np.array(cevts_st)
                x = cevts_st[acc_cevtix.astype(int)] # score-acc cevts time
                y = np.zeros((len(acc_cevtix), 1)) # perf_acc cevts timing
                # loop through the acc cevts
                for j in range(len(y)):
                    local_accevtix = acc_cevtix[j] # the cevtix
                    # the corresponding accix
                    local_accix = np.arange(cevts_stix[local_accevtix], cevts_stix[local_accevtix] + len(cevts[local_accevtix]))
                    # deal with o cases
                    non_o_local_accix = np.where(perf_acc_aligned[local_accix, 0])[0]
                    if non_o_local_accix.shape[0] != 0:
                        y[j] = np.median(perf_acc_aligned[non_o_local_accix + local_accix[0] - 2, 2])
                non_o_accevtix = np.where(y)[0]
                x = x[non_o_accevtix]
                y = y[non_o_accevtix]
                # st_xs = np.concatenate((st_xs, x), axis=0)
                st_xs = np.append(st_xs, x)
                st_ys = np.append(st_ys, y)
                # st_ys = np.concatenate((st_ys, y), axis=0)
                
            # do regression filling
            slop = getslop(st_xs, st_ys)
            inter = getinter(slop, st_xs, st_ys)
            # use regression filling timing as the missing perf_acc notes'
            # timing. Also use previous accis velocity as their velocity
            perf_acc_aligned[idx, 1] = (1 - (i==1)) * perf_acc_aligned[i-1, 1] + (i==1) * np.mean(perf_acc[:, 1])
            perf_acc_aligned[idx, 2:4] = score_acc[idx, 2:4] * slop + inter

        # use score pitch as the pitch
        perf_acc_aligned[idx, 0] = score_acc[idx, 0]

    # assign the output
    perf_acc_aligned_filled = perf_acc_aligned
    missing_accix = oaccix
    missing_accevtix = np.unique(oacc_cevtix)

    return perf_acc_aligned_filled, missing_accix, missing_accevtix


