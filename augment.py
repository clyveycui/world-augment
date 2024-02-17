# %%

# Plan is to augment the files using the corresponding f0s
# sampling rate for the f0s in pickles is 44100
# sampling rate for the audio files is 16000
# So may need to skip every few items
# Choose f0 based on average f0 similarity
# Should only choose f0 longer than the voice's f0

import os.path
import pickle
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite
import numpy as np
from world import main
import pickle
import math

f0_dir = os.path.join(os.path.expanduser('~'), 'kathakali_performances')
f0_c1_pickle = os.path.join(f0_dir, 'f0_c1.pkl')
f0_c2_pickle = os.path.join(f0_dir, 'f0_c2.pkl')

c1_f0s = []
c2_f0s = []

with open(f0_c1_pickle, 'rb') as f:
    c1_f0s = pickle.load(f)
    
with open(f0_c2_pickle, 'rb') as f:
    c2_f0s = pickle.load(f)

# %%
def truncate_starting_zeros(f0):
    res = []
    nonzero = False
    for f in f0:
        if (not nonzero):
            if f != 0:
                nonzero = True
            else:
                continue
        res.append(f)
    return res
        
def compute_f0_stats(f0):
    #get length and average f0
    length = len(f0)
    if length == 0:
        return None
    avg = sum(f0) / length
    return length, avg

#Computes f0 stats and returns the items sorted by length
def preprocess_singing_f0s(f0s):
    f0s_trunc = [truncate_starting_zeros(x) for x in f0s]
    f0_stats = [compute_f0_stats(x) for x in f0s_trunc]
    res = [(f0s_trunc[i], f0_stats[i][0], f0_stats[i][1]) for i in range(len(f0s))]
    return sorted(res, key=lambda x : x[1])

#Calculates f1 similarity
def f0_l1_similarity(f01, f02, length):
    diff = 0
    for i in range(length):
       diff += abs(f01[i] - f02[i]) ** 2  
    return diff

def find_closest_avg_f0(f0s, avg_f0, length):
    high = len(f0s)
    low = 0
    while (high != low):
        mid = math.floor(low + (high - low) / 2)
        if (length <= f0s[mid][1]):
            high = mid
        else:
            low = mid + 1
    smallest_f0_diff = 999999
    smallest_f0 = None
    for f0 in f0s[low : ]:
        f0_diff = abs(f0[2] - avg_f0)
        if smallest_f0_diff < f0_diff:
            continue
        else:
            smallest_f0_diff = f0_diff
            smallest_f0 = f0
    return smallest_f0

def find_closest_f0(f0s, f0_original, length):
    high = len(f0s)
    low = 0
    while (high != low):
        mid = math.floor(low + (high - low) / 2)
        if (length <= f0s[mid][1]):
            high = mid
        else:
            low = mid + 1
    smallest_f0_diff = -1
    smallest_f0 = None
    for f0 in f0s[low : ]:
        f0_diff = f0_l1_similarity(f0[0], f0_original, length)
        if smallest_f0_diff < f0_diff and smallest_f0 != None:
            continue
        else:
            smallest_f0_diff = f0_diff
            smallest_f0 = f0
    return smallest_f0

# %% 
c1_f0s_processed = preprocess_singing_f0s(c1_f0s)
c2_f0s_processed = preprocess_singing_f0s(c2_f0s)


# %%

data_dir = os.path.join(os.path.expanduser('~'), 'malayalam_speech', 'mono_male')
wav_fnames = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]

data_out_dir = os.path.join(os.path.expanduser('~'), 'malayalam_speech', 'augmented', 'mono_male_c1')

# %%
def augment(f0s, in_dir, out_dir, file_names):
    vocoder = main.World()

    for fname in file_names:
        fs, x_int16 = wavread(os.path.join(in_dir, fname))
        x = x_int16 / (2 ** 15 - 1) # to float
        dat = vocoder.encode(fs, x, f0_method='harvest')
        length, f0_avg = compute_f0_stats(dat['f0'])
        dat['f0'] = find_closest_f0(f0s, dat['f0'], length)[0][:length]
        wav = vocoder.decode(dat)
        
        wav['out'] /= 1.414
        wav['out'] *= 32767
        int16_data = wav['out'].astype(np.int16)
        out_fname = os.path.join(out_dir, fname)
        wavwrite(out_fname, rate=fs, data=int16_data)
# %%
augment(c1_f0s_processed, data_dir, data_out_dir, wav_fnames)

# %%
