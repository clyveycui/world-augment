import os.path
from scipy.io.wavfile import read as wavread
from world import main
import csv
import sys

def extract_features(fs, x):
    f0 = vocoder.encode(fs, x, f0_method='harvest')['f0']
    
    f0_max = max(f0)
    f0_min = min([x for x in f0 if x > 20])
    f0_range = f0_max - f0_min
    
    delta = 0
    for i in range(len(f0) - 1):
        delta += abs(f0[i] - f0[i + 1])
    delta /= (len(f0) - 1)
    return f0_max, f0_min, f0_range, delta

path=sys.argv[1]
outpath=sys.argv[2]

# Get the necessary files 
data_dir = os.path.join(os.path.expanduser('~'), path)
wav_fnames = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]


vocoder = main.World()

header = ['f0max', 'f0min', 'f0range', 'f0delta']
data = []

for fname in wav_fnames:
    fs, x_int16 = wavread(os.path.join(data_dir, fname))
    x = x_int16 / (2 ** 15 - 1) # to float
    c1_features = [x for x in extract_features(fs, x)]
    data.append(c1_features)



data_out_fname = os.path.join(os.path.expanduser('~'), outpath)

with open(data_out_fname, 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

    # write the data
    writer.writerows(data)
    