import whisper
model = whisper.load_model("base")  
audio_file = 'harvard.wav'
result = model.transcribe(audio_file)
result1=result['text']
print("Transcription:", result1)