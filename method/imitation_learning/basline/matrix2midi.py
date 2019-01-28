import pretty_midi
import os, fnmatch
import numpy as np

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
    midi_file.write("./results/" + name)
