import numpy as np

def notes2cevt(fullnotes):
    # this function takes in a fullnote matrix and return a compound event cell arry
    # input:
    #     fullnotes: a N * 4 note matrix, which must be a score fullnotes.
    # output:
    #     cevts: a cell array, each element is a vector to store the compound events'
    #     pitch
    #     cevtsix: a N * 1 array to show which cevts does nth notes belong to.
    #     index from 1. cevtsix(i) = j means the ith notes belongs to the jth
    #     cevt
    #     cevts_st: CN * 1 array to show the starting time of each cevt. CN is
    #     length(cevts)
    #     cevt_st(i) = T means ith cevt start from T sec.
    #     cevts_stix: CN*1 array to show the starting index of the cevts.
    #     cevts_stix(i) = j means the ith cevt's first note's index is j

    cevts = []
    cevtsix = np.zeros((fullnotes.shape[0], 1))
    cevts_st = []
    cevts_stix = []
    last_cevtix = 0
    # put first note in the first cell
    cevts.append([fullnotes[0, 0]])
    cevtsix[0] = last_cevtix
    cevts_st.append(fullnotes[0, 2])
    cevts_stix.append(0)

    # loop through the note matrix
    for i in range(1, fullnotes.shape[0]):
        # if the starting time is the same as prevs one, put in current cell
        if fullnotes[i, 2] == fullnotes[i-1, 2]:
            cevts[-1].append(fullnotes[i, 0])
            cevtsix[i] = last_cevtix
        # if the starting time is later than prevs one, put in a new cell
        elif fullnotes[i, 2] > fullnotes[i-1, 2]:
            cevts.append([fullnotes[i, 0]])
            last_cevtix += 1
            cevtsix[i] = last_cevtix
            cevts_st.append(fullnotes[i, 2])
            cevts_stix.append(i)
        # if the starting time is earlier, then there is a BUG
        else:
            raise ValueError('notes timing not in order')

    # cevts = np.array(cevts).T
    # cevts_st = np.array(cevts_st).T
    # cevts_stix = np.array(cevts_stix).T

    # todo(?) supposed to return the transpose of these matrices
    return cevts, cevtsix, cevts_st, cevts_stix