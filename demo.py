#%%
import os.path
from scipy.io.wavfile import read as wavread

from world import main

data_dir = os.path.join(os.path.expanduser('~'), 'download', 'Librispeech', '19')
wav_fname = os.path.join(data_dir, '19-198-0001.wav')
fs, x_int16 = wavread(wav_fname)
x = x_int16 / (2 ** 15 - 1) # to float
vocoder = main.World()
dat = vocoder.encode(fs, x, f0_method='harvest')
print (x_int16)
# %%
print("*****************************************")

dat = vocoder.scale_pitch(dat, 1.5)
dat = vocoder.scale_duration(dat, 2)

# requiem analysis
dat = vocoder.encode(fs, x, f0_method='harvest', is_requiem=True)
# %%

print(dat.keys())
# %%
dat = vocoder.encode(fs, x, f0_method='harvest')

# %%
print(dat.keys())
# %%
wav = vocoder.decode(dat)
# %%
print(wav.keys())
# %%
data_out_dir = os.path.join(os.path.expanduser('~'), 'download', 'Librispeech')
wav_out_fname = os.path.join(data_out_dir, '19-198-0001-mod.wav')



# %%
from scipy.io.wavfile import write as wavwrite
import numpy as np
wav['out'] /= 1.414
wav['out'] *= 32767
int16_data = wav['out'].astype(np.int16)
# %%
wavwrite(wav_out_fname, rate=16000, data=int16_data)
# %%
