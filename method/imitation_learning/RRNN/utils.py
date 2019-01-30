from __future__ import unicode_literals, print_function, division
import os, fnmatch
# import spacy
import unicodedata
import string
import re
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pretty_midi

def find(pattern, path):
    result = []
    files = os.listdir(path)
    for i in range(len(files)):
        if fnmatch.fnmatch(files[i], pattern):
            result.append(files[i])
    result.sort()
    return result

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # # this locator puts ticks at regular intervals
    # loc = ticker.MultipleLocator(base=0.2)
    # ax.yaxis.set_major_locator(loc)
    # plt.yticks(np.arange(min(points), max(points)+1, 0.1))
    plt.plot(points)

    plt.savefig("results/train_loss/train_loss.pdf")
    plt.close(fig)

def showDifPlot(loss, train_num, epoch, learning_rate, hidden_size, p, clip, model_num):
    plt.figure()
    fig, ax = plt.subplots()
    # seperate losses for different to verify the learning 
    list_idx = []

    for i in range(train_num):
        new_idx_list = list(range(i, len(loss), train_num))
        list_idx.append(new_idx_list)
    
    plot_idx = list(range(5, train_num, 7))

    for i in range(train_num):
        losses = []
        for j in range(int(len(loss) / train_num)):
            losses.append(loss[list_idx[i][j]])
        if i in plot_idx:
            plt.plot(losses, label='loss'+str(i))

    # # this locator puts ticks at regular intervals
    # loc = ticker.MultipleLocator(base=0.2)
    # ax.yaxis.set_major_locator(loc)
    # plt.yticks(np.arange(min(points), max(points)+1, 0.1))
    if model_num == 0:
        model_name = "FFNN"
    elif model_num == 1:
        model_name = 'RRNN'
    plt.ylim(0, 0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig("results/acc_e" + str(epoch) + "_lr" + str(learning_rate) + "_h" + str(hidden_size) + "_p" + str(p) + "_c" + str(clip) + "_" + model_name + ".pdf")
    plt.close(fig)

def matrix2midi(matrix, name):
    # create a PrettyMIDI file
    midi_file = pretty_midi.PrettyMIDI()

    # Create an instrument instance for a piano instrument
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)

    for i in range(len(matrix)):
        note_number = int(matrix[i][0])
        start_t = matrix[i][2]
        end_t = matrix[i][3]
        note = pretty_midi.Note(velocity=100, pitch=note_number, start=start_t, end=end_t)
        piano.notes.append(note)
    
    midi_file.instruments.append(piano)
    if 'data' in name:
        midi_file.write(name)
    else:
        midi_file.write("./results/" + name)

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