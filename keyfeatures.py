#Spectrogram
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
audio='harvard.wav'
y,sr=librosa.load(audio,sr=22050)
D=librosa.stft(y) #converts amplitude-time to amplitude vs frequency( this breaks audio into small time windows and computes fourier trnasform for each window, giving a 2D matric rows-frequency col=time) D- contains complex no.s where magntude represents amplitudes of the frequencies at each time and phase indicates shift of those frequencies
S_db=librosa.amplitude_to_db(np.abs(D),ref=np.max) #converting complex no.s to absolute values and convert amplitude to db scale and reference value is the max amplitude so the decibel values are scaled relative to the loudest point in the audio.
plt.figure(figsize=(10,4))
librosa.display.specshow(S_db,sr=sr,x_axis='time',y_axis='log')
plt.colorbar(format='%+2.0f db')
plt.title('Spectrogram')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.plot()

#Chroma features
chroma=librosa.feature.chroma_stft(y=y,sr=sr)
plt.figure(figsize=(10,4))
librosa.display.specshow(chroma,x_axis='time',y_axis='chroma',sr=sr)
plt.colorbar()
plt.title('Chroma Features')
plt.xlabel('Time')
plt.ylabel('Chroma')
plt.plot()

#Zero-Crossing Rate
zero_crossing=librosa.feature.zero_crossing_rate(y) #2D array where row corresponds to zero-crossing rate for a specific frame of the audio.(mostly the func outputs one row of data array has one row and multiple columns)
plt.figure(figsize=(10,4))
plt.plot(zero_crossing[0])
plt.title('Zero Crossing Rate')
plt.xlabel('Time')
plt.ylabel('Rate')
plt.plot()