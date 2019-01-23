import os
from utils import *

def get_scorefile_ix(folder, prefix='labeled', suffix='score'):
    #  find the index of the score file within a folder by checking the suffix
    if folder[-1] != '/':
        folder += '/'
    a = find(prefix + "*.txt", folder)
    for i in range(len(a)):
        if suffix in a[i]:
            # print(a[i], i)
            return i
    return -1