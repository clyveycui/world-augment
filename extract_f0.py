
import os.path
from scipy.io.wavfile import read as wavread
from world import main
import pickle

# Get the necessary files 
data_dir = os.path.join(os.path.expanduser('~'), 'kathakali_performances', "trimmed_original")
wav_fnames = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]


vocoder = main.World()
f0s_c1 = []
f0s_c2 = []
for fname in wav_fnames:
    
    fs, x_int16 = wavread(os.path.join(data_dir, fname))
    print(x_int16)
    x = x_int16 / (2 ** 15 - 1) # to float
    dat_c1 = vocoder.encode(fs, x[:, 0], f0_method='harvest')
    dat_c2 = vocoder.encode(fs, x[:, 1], f0_method='harvest')
    f0s_c1.append(dat_c1['f0'])
    f0s_c2.append(dat_c2['f0'])





data_out_dir = os.path.join(os.path.expanduser('~'), 'kathakali_performances')
data_out_fname1 = os.path.join(data_out_dir, 'f0_c1.pkl')
data_out_fname2 = os.path.join(data_out_dir, 'f0_c2.pkl')

with open(data_out_fname1, 'wb') as f:
    pickle.dump(f0s_c1, f)

with open(data_out_fname2, 'wb') as f:
    pickle.dump(f0s_c2, f)
