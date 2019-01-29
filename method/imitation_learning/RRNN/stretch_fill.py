import numpy as np

def stretch_fill(score, perf, p):
    #  this function smooth out the 0 cases in mel_t by playing linear interpolation
    #  of the previous 2 + afterwards 2 (4 in total) notes
    #  input:
    #     same as stretch follow
    #  output:
    #     perf_aligned_filled: the new perf_t after linear interpolation

    # get the perf in order, align the score
    score = score[:, 0:4]
    N = score.shape[0]
    perf_aligned = np.zeros((N, 4))
    ix = perf[:, 4] - 1
    perf_aligned[ix.astype(int), :] = perf[:, 0:4]
    
    oix = np.where(perf_aligned[:, 0] == 0)[0]
    # loop through 0 index
    getslop = lambda x, y: np.sum((y-np.mean(y))*(x-np.mean(x)))/np.sum(np.power(x-np.mean(x), 2))
    getinter = lambda s, x, y: np.mean(y)-s*np.mean(x)

    for i in range(len(oix)):
        idx = oix[i]
        # deal with zero case: find the non-zero index of perf
        pre_ix = np.where(perf_aligned[0:idx, 2] == p/2)[0][-int(p/2):]
        aft_ix = np.where(perf_aligned[idx:, 2] == p/2)[0][:int(p/2)]
        # print("aft_ix", aft_ix)
        if len(pre_ix) < p/2:
            aft_ix = np.where(perf_aligned[idx:, 2])[0][:int(p-len(pre_ix))]
        if len(aft_ix) < p/2:
            pre_ix = np.where(perf_aligned[0:idx, 2])[0][-int(p-len(aft_ix)):]

        aft_ix = aft_ix + idx
        # print("i", i)
        # print("aft_ix", aft_ix)
        non_o_ix = np.append(pre_ix, aft_ix, axis=0)

        st_xs = score[non_o_ix, 2]
        st_ys = perf_aligned[non_o_ix, 2]

        slop = getslop(st_xs, st_ys)
        inter = getinter(slop, st_xs, st_ys)
        # fill in the prediction perf
        # keep the pitch and vel
        perf_aligned[idx, 0] = score[idx, 0]
        perf_aligned[idx, 1] = np.sum(perf_aligned[non_o_ix, 1]) / p
        # stretch starting time and ending time according to 
        perf_aligned[idx, 2:4] = score[idx, 2:4] * slop + inter
    
    perf_aligned_filled = perf_aligned
    missing_melix = oix

    return perf_aligned_filled, missing_melix