import librosa #for audio analysis
import librosa.display #display module of librosa for visualization
import matplotlib.pyplot as plt #for plotting

audio=r'C:\Harinee\ASR\harvard.wav'
y,sr=librosa.load(audio,sr=None)#y(1D numpy array containing amplitude values) has the audio file and stores the amplitude of sound at specific moment and sr has the sample rate,setting sr=None means the original sample of the audio is preserved. (load is used to read audio file file and convert them into a format that is used for future analysis)
plt.figure(figsize=(10,4))#creates a figure where the grpah it plotted by the next line of code
librosa.display.waveshow(y,sr=sr)#plot the waveform of the audio
plt.title('Waveform of the given audio')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

#FEATURE EXTRACTION
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13) #extracting 13 mfccs(mfccs variables has 2D numpy array with column number =no of frames and each column represents the MFCCs for a specific time frame in audio signal and no of rows=13)
plt.figure(figsize=(10,4))
librosa.display.specshow(mfccs,x_axis='time',sr=sr,cmap='coolwarm') #sr scales the x_axis time in terms of sample rate
plt.colorbar(format='%+2.0f dB')
plt.title('MFCCs of the Audio File')
plt.xlabel('Time')
plt.ylabel('MFCC coefficents')
plt.show()

