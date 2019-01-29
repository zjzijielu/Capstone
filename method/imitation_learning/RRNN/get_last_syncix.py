def get_last_syncix(score_mel, score_acc):
    # get the last sync melix and accix for score. This is only for compute median perf
    melix = score_mel.shape[0] - 1
    accix = score_acc.shape[0] - 1

    while score_mel[melix, 2] != score_acc[accix, 2]:
        if score_mel[melix, 2] < score_acc[accix, 2]:
            accix -= 1
        else:
            melix -= 1
        
    return melix, accix