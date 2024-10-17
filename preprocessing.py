#PREPROCESSING
import librosa
import numpy as np
audio='harvard.wav'
y,sr=librosa.load(audio,sr=16000)#setiing sr=16000 resampling the audio to match the ASR model's audio
y=librosa.util.normalise(y)#normalising the audio to bring amplitude consistency
target_length=18*sr
if(len(y)>target_length):
  y=y[:target_length]#if audio is longer than target length slice it till target length
else:
  y=np.pad(y,(0,target_length-len(y)),mode='constant')#if audio less than 5 secs it gets padded with silence to reach 5 secs at the end of the file, constant tells padding is done with a constant value 0 by default
