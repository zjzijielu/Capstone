from __future__ import unicode_literals, print_function, division

import os
# import spacy
import unicodedata
import string
import re
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # # this locator puts ticks at regular intervals
    # loc = ticker.MultipleLocator(base=0.2)
    # ax.yaxis.set_major_locator(loc)
    # plt.yticks(np.arange(min(points), max(points)+1, 0.1))
    plt.plot(points)

    plt.savefig("results/graph/acc.pdf")
    plt.close(fig)

def showDifPlot(init, half, last):
    plt.figure()
    fig, ax = plt.subplots()
    # # this locator puts ticks at regular intervals
    # loc = ticker.MultipleLocator(base=0.2)
    # ax.yaxis.set_major_locator(loc)
    # plt.yticks(np.arange(min(points), max(points)+1, 0.1))
    plt.plot(init, label='init')
    plt.plot(half, label='half')
    plt.plot(last, label='last')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig("results/graph/acc.pdf")
    plt.close(fig)

