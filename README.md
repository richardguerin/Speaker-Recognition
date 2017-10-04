# Speaker Recognition
A simple speaker recognition system.

Praat is used to extract data from a collection of mp3 files containing speech, each of which is marked as being my voice or not my voice.

This data is used to train a scikit-learn classifier, which can then take new speech data and predict if it is my voice or not.

In my experiments, with about 300 training files in total from 30 speakers, the system achieved about 80% accuracy at identifying my voice.
