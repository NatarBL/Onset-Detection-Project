#!/usr/bin/env python
# coding: utf-8

# # CS583 Final Project
# 

# In this experiement we will be comparing three different methods, two of which focus on novelty, to find the most accurate method of onset detection. The first method will be amplitude based onsets which determines when changes occur in amplitude. The send will be another method discussed in class, spectra detection - which is used by the librosa library. Lastly, we will evaluate the work done by Nick Collins, in "Using a Pitch Detector for Onset Detection" , for onset detection that focuses on pitch detection.

# # Part One: Implementing onset detectors

# It is important to note that amplitude and spectra are not the only step in onset detection, rather they are part of the process. Looking at how the librosa library uses onset detection, we see that spectra detection is a key factor in onset detection. The command librosa.onset.onset_dtect follows three key steps in SOURCE ONE:
#     1. Compute a spectral novelty function. (This will be our primary focus.)
#     2. Find peaks in the spectral novelty function.
#     3. Backtrack from each peak to a preceding local minimum. (This step is optional.)

# Now we will be looking deeper into spectral based onset detection. The methods used in class for spectral based onset detection are simular to the ones outline in libroa library since both depend on the use of spectrograms. 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import librosa 
import librosa.display
from IPython.display import Audio

from scipy import signal

from scipy.signal import find_peaks, windows

get_ipython().run_line_magic('matplotlib', 'inline')

# Basic audio parameters

SR            = 22050                  #  sample rate default for Librosa

# Utility functions

# Round to 4 decimal places

def round4(x):
    return np.around(x,4)  

# normalize a signal so that its max = 1.0

def normalize(x):
    return x / max(x)

import soundfile as sf
import io
import warnings

from six.moves.urllib.request import urlopen

def readSignal(name,sr=None):    
    if(name[:5] == 'https'):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X, fileSR = sf.read(io.BytesIO(urlopen(name).read()))           
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X, fileSR = librosa.load(name)
    if((sr == None) or (fileSR == sr)):
        return X
    else:
        return librosa.resample(X,fileSR,SR) 

def writeSignal(name,data,sr=SR):
    sf.write(name,data,sr)
    
# [Cell Credit: Wayne Snyder]


# In[ ]:


def realFFT(X):
    return 2*abs(np.fft.rfft(X))/len(X) 

def spectral_distance(S,Sn,kind='SF2'):      
    S = np.abs(S)          
    Sn = np.abs(Sn)
    if(kind == 'L1'):
        return np.sum(np.abs(Sn-S))
    elif(kind == 'L2'):
        return (np.sum((Sn-S)**2))**0.5
    elif(kind == 'CD'):                     
        s = np.std(S)                       
        sn = np.std(Sn)
        if(np.isclose(s,0) or np.isclose(sn,0)):
            return 0.0
        else:
            return 1.0 - (((S - np.mean(S)) @ (Sn - np.mean(Sn))) / (len(S) * s * sn))
    elif(kind == 'RL1'):
        return np.sum(np.abs(rectify(Sn-S)))
    elif(kind == 'RL2'):
        return (np.sum(rectify(Sn-S)**2))**0.5               
    else:
        return None
    
def SpectralBasedOnsets(X,window_size=512,overlap=0.5,
                        kind = None,     # distance function used, L1, L2, CD, SF1, SF2
                        filtr = None,    # filter applied before peak picking, if any
                        size = 3,        # size of kernel used in filter
                        win = None,      # apply windowing function to window
                        scale=None,      # scale factor for log, None = no log
                        height=None,     # these 3 parameters are for pick_peak,
                        prominence=None, #    any not equal to None will be applied
                        distance=None,
                        displayAll=False):
    
    N = len(X)
    
    X = X / max(X)       # normalize amplitude

    skip = int((1-overlap)*window_size)

    num_windows = (N // skip) - 1

    window_locations = skip * np.arange(num_windows)
    
    if (win == 'hann'):
        W = windows.hann(window_size)
    elif (win == "tri"):
        W = windows.triang(window_size)
    else:
        W = windows.boxcar(window_size)
        
    X_spectrogram = np.array( [ realFFT( W * X[ w : w + window_size ] ) for w in window_locations ])   
    
    if(scale == None):
        X_spectrogram_log = X_spectrogram
    else:
        X_spectrogram_log = np.log(1 + scale*X_spectrogram)

    X_spectral_novelty = np.zeros(num_windows)
    
    for k in range(1,num_windows):           #first value will be 0, length unchanged
        X_spectral_novelty[k] =               spectral_distance(X_spectrogram_log[k-1],X_spectrogram_log[k],kind)
            
    X_spectral_novelty = normalize(X_spectral_novelty)
        
    if(filtr != None):
        X_spectral_novelty = filtr(X_spectral_novelty,size)
            
    peaks,_ = find_peaks(X_spectral_novelty,height=height,prominence=prominence,distance=distance)  

    if(len(peaks)==0):
        print("No peaks found!")
        return (np.array([]), np.zeros(len(X)))
    
    onsets = peaks*skip + window_size//2

    clicks = librosa.clicks(times=onsets/SR, sr=SR, 
                        hop_length=skip, length=len(X))
    
    return (onsets,clicks)

# [Cell Credit: Wayne Snyder]


# In[ ]:


# Amplitude-based onset detection

def energy(x):
    return (x @ x) / len(x)

# Half-wave rectification

def rectify(x):
    if(isinstance(x,np.ndarray)):
        return np.maximum(x,np.zeros(len(x)))
    elif(isinstance(x,list)):
        return list(np.maximum(x,np.zeros(len(x))))
    else:
        return max(x,0)

def AmplitudeBasedOnsets(X,window_size=512,overlap=0.5,scale=10,
                         height=None,
                         prominence=None,
                         distance=None,
                         displayAll=False):
    
    N = len(X)
    X = X / max(X)       

    skip = int((1-overlap)*window_size)
    num_windows = (N // skip) - 1
    window_locations = skip * np.arange(num_windows)

    X_energy = np.array( [ energy( X[ w : (w+window_size)] ) for w in window_locations ])
    X_energy = np.array(X_energy)
    
    if(scale == None):
        X_energy_log = X_energy
    else:
        X_energy_log = np.log(1 + scale*X_energy)

    # Difference = Novelty Function
    
    # add 0 at beginning so diff alines with change in new window
    
    X_energy_log = np.concatenate([[0],X_energy_log])    

    # Take the discrete differential; watch out, diff transforms array in place

    X_energy_novelty = np.diff(list(X_energy_log)) 
    
    # standardize novelty
    
    X_energy_novelty = X_energy_novelty / max(X_energy_novelty)

    X_energy_novelty_rectified = rectify(X_energy_novelty)      

    # peak picking

    peaks,_ = find_peaks(X_energy_novelty_rectified,
                         height=height,prominence=prominence,distance=distance)   

    if(len(peaks)==0):
        print("No peaks found!")
        return (np.array([]), np.zeros(len(X)))
    
    onsets = peaks*skip + window_size//2
    
    clicks = librosa.clicks(times=onsets/SR, sr=SR, 
                        hop_length=skip, length=len(X))
    
    return (onsets,clicks)

# [Cell Credit: Wayne Snyder]


# # Part 2: Mean absolute error evaluation

# In[ ]:


#Part 2A

X = readSignal("https://www.cs.bu.edu/fac/snyder/cs583/AudioSamples/KS.guitar.C.scale.wav") 

(onset_spec,clicks_spec) = SpectralBasedOnsets(X,window_size=128,overlap=0.5,
                                    kind = 'L1',       # distance function used, L1, L2, CD, SF1, SF2
                                    scale=20,          # scale factor for log, None = no log
                                    prominence=0.4,    # any not equal to None will be applied
                                    displayAll=True)

(onset_amp,clicks_amp) = AmplitudeBasedOnsets(X,window_size=128,overlap=0.5,
                                    scale=20,
                                    prominence=0.2,
                                    displayAll=True)

onset_frames = SR*librosa.onset.onset_detect(X, sr=SR, wait=1, pre_avg=1, post_avg=1, pre_max=1, post_max=1)
onset_times = librosa.frames_to_time(onset_frames)
clicks = librosa.clicks(frames=onset_frames, sr=SR, length=len(X))


# In[ ]:


# Part 2B

spectra_trial   = np.mean((onset_spec/SR)-np.array([0.5,1.0,1.5, 2.0, 2.5, 3.0,3.5, 4.0]))    
amplitude_trial = np.mean((onset_amp/SR)-np.array([0.5,1.0,1.5, 2.0, 2.5, 3.0,3.5, 4.0]))
librosa_trial   = np.mean((onset_times/SR)-np.array([0.5,1.0,1.5, 2.0, 2.5, 3.0,3.5, 4.0]))

print("Spectra Error  :", round4(spectra_trial)  , "seconds.")
print("Amplitude Error:", round4(amplitude_trial), "seconds.")
print("Librosa Error  : ", round4(librosa_trial)  , "seconds.")


# # Part 3: Graph evaluations

# In[ ]:


# Part 3A

pots_and_pans   = readSignal('PotsAndPans.wav') 
slow_clapping   = readSignal('SlowClapping.wav') 
electric_guitar = readSignal('ElectricGuitar.wav') 
acoustic_guitar = readSignal('AcousticGuitarFingerStyle.wav') 


# In[ ]:


# Part 3B

(onset_spec,clicks_spec) = SpectralBasedOnsets(pots_and_pans,window_size=128,overlap=0.5,
                                    kind = 'L1',       # distance function used, L1, L2, CD, SF1, SF2
                                    scale=20,          # scale factor for log, None = no log
                                    prominence=0.4,    # any not equal to None will be applied
                                    displayAll=True)

(onset_amp,clicks_amp) = AmplitudeBasedOnsets(pots_and_pans,window_size=128,overlap=0.5,
                                    scale=20,
                                    prominence=0.2,
                                    displayAll=True)

onset_frames = librosa.onset.onset_detect(pots_and_pans, sr=SR, wait=1, pre_avg=1, post_avg=1, pre_max=1, post_max=1)
onset_times = librosa.frames_to_time(onset_frames)
clicks_lib = librosa.clicks(frames=onset_frames, sr=SR, length=len(pots_and_pans))


# In[ ]:


# Part 3C

plt.figure(figsize=(12,6))
plt.title("Pots & Pans - Spectral Based Onset Detection", fontsize=14, fontweight='bold')
plt.plot(pots_and_pans)
plt.plot(clicks_spec,color='r')
plt.show()

plt.figure(figsize=(12,6))
plt.title("Pots & Pans - Amplitude Based Onset Detection", fontsize=14, fontweight='bold')
plt.plot(pots_and_pans)
plt.plot(clicks_amp,color='r')
plt.show()

plt.figure(figsize=(12,6))
plt.title("Pots & Pans - Librosa Onset Detection", fontsize=14, fontweight='bold')
plt.plot(pots_and_pans)
plt.plot(clicks_lib,color='r')
plt.show()


# In[ ]:


# Part 3D

(onset_spec,clicks_spec) = SpectralBasedOnsets(slow_clapping,window_size=128,overlap=0.5,
                                    kind = 'L1',       # distance function used, L1, L2, CD, SF1, SF2
                                    scale=20,          # scale factor for log, None = no log
                                    prominence=0.4,    # any not equal to None will be applied
                                    displayAll=True)

(onset_amp,clicks_amp) = AmplitudeBasedOnsets(slow_clapping,window_size=128,overlap=0.5,
                                    scale=20,
                                    prominence=0.2,
                                    displayAll=True)

onset_frames = librosa.onset.onset_detect(slow_clapping, sr=SR, wait=1, pre_avg=1, post_avg=1, pre_max=1, post_max=1)
onset_times = librosa.frames_to_time(onset_frames)
clicks_lib = librosa.clicks(frames=onset_frames, sr=SR, length=len(slow_clapping))


# In[ ]:


# Part 3E


plt.figure(figsize=(12,6))
plt.title("Slow Clapping - Spectral Based Onset Detection", fontsize=14, fontweight='bold')
plt.plot(slow_clapping)
plt.plot(clicks_spec,color='r')
plt.show()

plt.figure(figsize=(12,6))
plt.title("Slow Clapping - Amplitude Based Onset Detection", fontsize=14, fontweight='bold')
plt.plot(slow_clapping)
plt.plot(clicks_amp,color='r')
plt.show()

plt.figure(figsize=(12,6))
plt.title("Slow Clapping - Librosa Onset Detection", fontsize=14, fontweight='bold')
plt.plot(slow_clapping)
plt.plot(clicks_lib,color='r')
plt.show()


# In[ ]:


# Part 3F

(onset_spec,clicks_spec) = SpectralBasedOnsets(electric_guitar,window_size=128,overlap=0.5,
                                    kind = 'L1',       # distance function used, L1, L2, CD, SF1, SF2
                                    scale=20,          # scale factor for log, None = no log
                                    prominence=0.4,    # any not equal to None will be applied
                                    displayAll=True)

(onset_amp,clicks_amp) = AmplitudeBasedOnsets(electric_guitar,window_size=128,overlap=0.5,
                                    scale=20,
                                    prominence=0.2,
                                    displayAll=True)

onset_frames = librosa.onset.onset_detect(electric_guitar, sr=SR, wait=1, pre_avg=1, post_avg=1, pre_max=1, post_max=1)
onset_times = librosa.frames_to_time(onset_frames)
clicks_lib = librosa.clicks(frames=onset_frames, sr=SR, length=len(electric_guitar))


# In[ ]:


# Part 3G


plt.figure(figsize=(12,6))
plt.title("Electric guitar - Spectral Based Onset Detection", fontsize=14, fontweight='bold')
plt.plot(electric_guitar)
plt.plot(clicks_spec,color='r')
plt.show()

plt.figure(figsize=(12,6))
plt.title("Electric Guitar - Amplitude Based Onset Detection", fontsize=14, fontweight='bold')
plt.plot(electric_guitar)
plt.plot(clicks_amp,color='r')
plt.show()

plt.figure(figsize=(12,6))
plt.title("Electric Guitar - Librosa Onset Detection", fontsize=14, fontweight='bold')
plt.plot(electric_guitar)
plt.plot(clicks_lib,color='r')
plt.show()


# In[ ]:


# Part 3H

(onset_spec,clicks_spec) = SpectralBasedOnsets(acoustic_guitar,window_size=128,overlap=0.5,
                                    kind = 'L1',       # distance function used, L1, L2, CD, SF1, SF2
                                    scale=20,          # scale factor for log, None = no log
                                    prominence=0.4,    # any not equal to None will be applied
                                    displayAll=True)

(onset_amp,clicks_amp) = AmplitudeBasedOnsets(acoustic_guitar,window_size=128,overlap=0.5,
                                    scale=20,
                                    prominence=0.2,
                                    displayAll=True)

onset_frames = librosa.onset.onset_detect(acoustic_guitar, sr=SR, wait=1, pre_avg=1, post_avg=1, pre_max=1, post_max=1)
onset_times = librosa.frames_to_time(onset_frames)
clicks_lib = librosa.clicks(frames=onset_frames, sr=SR, length=len(acoustic_guitar))


# In[ ]:


# Part 3I

plt.figure(figsize=(12,6))
plt.title("Acoustic guitar - Spectral Based Onset Detection", fontsize=14, fontweight='bold')
plt.plot(acoustic_guitar)
plt.plot(clicks_spec,color='r')
plt.show()

plt.figure(figsize=(12,6))
plt.title("Acoustic Guitar - Amplitude Based Onset Detection", fontsize=14, fontweight='bold')
plt.plot(acoustic_guitar)
plt.plot(clicks_amp,color='r')
plt.show()

plt.figure(figsize=(12,6))
plt.title("Acoustic Guitar - Librosa Onset Detection", fontsize=14, fontweight='bold')
plt.plot(acoustic_guitar)
plt.plot(clicks_lib,color='r')
plt.show()

