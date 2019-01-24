import numpy as np

def get_median_alignedperf(aligned_perfs, cursor_ix):
    # ## this function takes in multi performances and return the median version of them by
    # 1. find the median length (median length is the last note's starting time!)
    # 2. normalize all perfs to the median length
    # 3. find median starting time, dur, vel for each note. Make it a median note
    # 4. concatenate the median notes to be the median perf
    # 
    # input: 
    #     aligned_perfs : a tensor of aligned_perfs
    # output:
    #     median_perf: the median perf, a N*4 matrix

    # cursor_ix = aligned_perfs.shape[0] - 1
    perfs = aligned_perfs

    # normalize overall length by median length
    # print(perfs[:, 2, :], perfs[:,2,:].shape)
    med_length = np.median(perfs[:,2,:], axis=1)
    for i in range(perfs.shape[2]):
        # debug last ix, for aligned perf it seems not necessary
        perfs[:, 2:4, i] = np.divide(perfs[:, 2:4, i], perfs[cursor_ix, 2, i]) * med_length[cursor_ix]
    
    # normalize each note
    # print(perfs)
    
    median_perf = np.median(perfs, 2)
    # round the .5 velocity
    median_perf[:, 1] = np.ceil(median_perf[:, 1])  

    return median_perf
