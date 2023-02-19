import numpy as np
from numpy._typing import NDArray
from typing import Any

def random_seed_to_generate_music() -> NDArray[Any]:
    from train import network_input

    start = np.random.randint(0, len(network_input) - 1)
    pattern = network_input[start]

    return pattern

def decode_to_music(prediction_output):
    from music21 import instrument, note, chord

    offset = 0
    music = []

    for pitches in prediction_output:
        # if pitches is a chord
        if ('.' in pitches) or pitches.isdigit():
            notes_in_chord = pitches.split('.')

            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)

            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            music.append(new_chord)
        # pitches is a note
        else:
            new_note = note.Note(pitches)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            music.append(new_note)

        offset += 0.5

    return music

def generate_music(model, music_length=500):

    # Load the weights to each node
    model.load_weights('model-129-0.5906.hdf5')

    from train import notes, n_vocab

    # Make random start point (or you can make your own start points with the same length of `sequence_length` (100) )
    pattern = random_seed_to_generate_music()

    pitch_names = sorted(set(notes))
    int_to_note = dict( (number, note) for number, note in enumerate(pitch_names) )

    # generate 500 notes
    prediction_output = []
    for note_index in range(music_length):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input /= float(n_vocab)
        prediction = model.predict(prediction_input, verbose=0)

        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output = np.append(prediction_output, result)

        pattern = np.append(pattern, index)
        pattern = pattern[1:len(pattern)]

    music = decode_to_music(prediction_output)

    from music21 import stream
    midi_stream = stream.Stream(music)
    midi_stream.write('midi', fp='generated_music.mid')


if __name__=='__main__':
    from train import build_model
    model = build_model()

    generate_music(model)



    ''' G3.D4.A4.G5 '''
