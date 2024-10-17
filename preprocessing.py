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
  
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor #first one Wav2Vec 2.0 model class and 2nd one processor for preprocessing audio inputs and converting model outputs into human-readable text
processor=Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')#loads the pre-trained Wav2Vec 2.0 processor from the specified model identifier on hugging face model hub (includes neccessary configurations to preprocess audio data)
model=Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h') #load the Wav2Vec 2.0 model fro asr(set up for ctc(connectionist temporal classification for transcribing speech))
input_values=processor(y,sampling_rate=16000,return_tensors='pt').input_values #taking y to preprocess to process it using pretrained model and store output in pytorch format(dict-key pairs)(return_tensors..pt(tensors)) input_values(actual data to be passed to model)
with torch.no_grad(): #pytorch wont track gradients (disabling gradient tracking saves memory and runs faster)
  logits=model(input_values).logits #model processes audio and raw output is called logits (represent model's confident score) higher logit- more confident thata certain token corresponds to that part of the audio
predicted_ids=torch.argmax(logits,dim=-1) #sees logits and determines which token has highest score in each step(torch.argmax: returns index position of max values along a specific dim, predicted_ids will contain most likely token id for each step based on models prediction)
transcription=processor.batch_decode(predicted_ids)[0] #converts the token id to human readable text. [0] to get first (since only one audio)and only transcription from the list
print("Transactions: ",transcription)