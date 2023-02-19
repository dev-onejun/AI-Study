def get_notes() -> list:
    notes = []

    import glob
    for file in glob.glob('data/midi_songs/*.mid'):
        midi = converter.parse(file)
        parts = instrument.partitionByInstrument(midi)

        notes_to_parse = None
        if parts: # file has instrument parts
            notes_to_parse = parts.parts[0].recurse()
        else: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    return notes

def preprocess_data(notes):
    sequence_length = 100

    # get all pitch names
    pitch_names = sorted(set(item for item in notes))

    # create a dictionary to map pitches to integers
    note_to_int = dict( (note, number) for number,note in enumerate(pitch_names) )

    # create input sequences and the corresponding outputs
    network_input = []
    network_output = []
    for i in range(len(notes) - sequence_length):
        sequence_in = notes[i: i+sequence_length]
        sequence_out = notes[i+sequence_length]

        network_input.append( [ note_to_int[char] for char in sequence_in ] )
        network_output.append(note_to_int[sequence_out])

    # reshape the input into a format compatible with LSTM layers
    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))

    # normalize input
    from keras.utils import np_utils

    network_input = network_input / float(n_vocab)
    network_output = np_utils.to_categorical(network_output)

    return network_input, network_output

def build_model():
    model = Sequential()

    model.add(LSTM(
        256,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))

    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))

    model.add(LSTM(256))
    model.add(Dense(256))
    model.add(Dropout(0.3))

    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

def fit_model(model):
    from keras.callbacks import ModelCheckpoint

    check_point = ModelCheckpoint(
        filepath = 'model-{epoch:02d}-{loss:.4f}.hdf5',
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [check_point]

    model.fit(network_input, network_output, epochs=200, batch_size=64, callbacks=callbacks_list)

    return model

from music21 import converter, instrument, note, chord
notes = get_notes()
n_vocab = len(set(notes))

import numpy as np
network_input, network_output = preprocess_data(notes)

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
if __name__ == '__main__':
    model = build_model()
    fit_model(model)
