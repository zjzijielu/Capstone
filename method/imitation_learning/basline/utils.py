import os, fnmatch
import numpy as np

def find(pattern, path):
    result = []
    files = os.listdir(path)
    for i in range(len(files)):
        if fnmatch.fnmatch(files[i], pattern):
            result.append(files[i])
    result.sort()
    return result

def ismember(A, B):
    result = []
    for i in range(len(A)):
        a = A[i]
        if np.sum(a == B) == 1:
            result.append(i)
    return result

def setdiff(a, b):
    result = []
    for i in range(len(a)):
        if np.where(b == a[i])[0].shape[0] == 0:
            result.append(i)
    result = np.array(result)
    return result

def ismember_ix(a_vec, b_vec):
    """ MATLAB equivalent ismember function """

    bool_ind = np.isin(a_vec,b_vec)
    common = a_vec[bool_ind]
    common_unique, common_inv  = np.unique(common, return_inverse=True)     # common = common_unique[common_inv]
    b_unique, b_ind = np.unique(b_vec, return_index=True)  # b_unique = b_vec[b_ind]
    common_ind = b_ind[np.isin(b_unique, common_unique, assume_unique=True)]
    return bool_ind, common_ind[common_inv]