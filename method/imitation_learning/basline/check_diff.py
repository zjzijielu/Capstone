import numpy as np
import os

file1 = np.loadtxt("/Users/luzijie/Desktop/Capstone/method/imitation_learning/basline/results/boy/median_mel.txt")
file2 = np.loadtxt("/Users/luzijie/Desktop/Capstone/data/polysample_boy/syn/median_mel.txt")

for i in range(len(file1)):
    if abs(file1[i, 2] - file2[i, 2]) > 0.01:
        print(i)
        print("result", file1[i, :])
        print("target", file2[i, :])