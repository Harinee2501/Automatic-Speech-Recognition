import whisper
model = whisper.load_model("base")  
audio_file = 'harvard.wav'
result = model.transcribe(audio_file)
result1=result['text']
print("Transcription:", result1)

import jiwer
reference="The stale smell of old beer lingers. It takes heat to bring out the odor. A cold dip restores health and zest.A salt pickle tastes fine with ham. Tacos al pastor are my favorite. A zestful food is the hot cross bun."
wer=jiwer.wer(reference,result1)
print(f"Word Error Rate: {wer:.2f}") #0.05