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