# **Automatic Speech Recognition (ASR) Project**
## **🚩 Problem Statement**
The accurate transcription of speech into text is a crucial requirement across domains like accessibility, customer service, and content creation. However, traditional speech-to-text systems often struggle with background noise, diverse accents, or non-standard audio formats, limiting their usability.

## **💡 Project Idea**
This project focuses on building an Automatic Speech Recognition (ASR) pipeline that processes raw audio data, extracts meaningful features, and transcribes it into text using pre-trained models like Wav2Vec 2.0 and Whisper.

## **Key Functionalities:**
Audio Input:
Collect raw audio data in formats like .wav or .mp3.
Preprocessing:
Resample, normalize, and pad/truncate audio to make it compatible with the ASR models.
Feature Extraction:
Derive key audio features like MFCCs, spectrograms, and zero crossing rates for analysis.
Modeling:
Use pre-trained models such as Wav2Vec 2.0 and Whisper to transcribe speech into text.
Post-Processing:
Improve text accuracy by calculating metrics like Word Error Rate (WER) and refining inputs.

## **🔬 Technical Approach**
Day 1: Building the ASR Pipeline
1. Feature Extraction
MFCCs (Mel-frequency cepstral coefficients):
Extracted audio features using the librosa library to summarize audio signals into a machine-readable format.

Steps: Fourier Transform → Mel Scale Conversion → DCT Transformation.
Challenge: Execution time was high due to the large sample rate; resolved by resampling audio to reduce size.
Other Features:

Spectrogram: Analyzed frequency content over time using Short-Time Fourier Transform (STFT).
Zero Crossing Rate (ZCR): Measured rapid changes in the waveform to capture percussive sounds.
2. Visualizations:
Plotted audio signals, spectrograms, and MFCCs using matplotlib to understand their distribution and characteristics.

Day 2: Speech-to-Text Conversion
1. Preprocessing Audio Data:
Resampled to a 16 kHz sample rate for compatibility.
Normalized audio for consistent amplitude levels.
Padded or trimmed to ensure uniform input lengths.
2. Pre-Trained Models:
Wav2Vec 2.0:

Integrated using Hugging Face’s Transformers library.
Extracted logits (confidence scores) and converted them into transcriptions.
Challenge: High WER (0.98) with noisy audio.
Whisper:

Used OpenAI's Whisper model for transcription.
Handled raw audio effectively without extensive preprocessing.
Achieved low WER (0.05), even with noisy inputs.
🛠 Tech Stack
Libraries and Tools:
Librosa: Audio processing and feature extraction.
Matplotlib: Visualization of audio signals, spectrograms, and MFCCs.
Pydub: For audio manipulation like cutting and resampling.
Transformers: Hugging Face library for pre-trained models.
PyTorch: For tensor computations and deep learning.
Soundfile: For handling audio files.
Jiwer: To calculate Word Error Rate (WER).
Pre-Trained Models:
Wav2Vec 2.0: Suitable for controlled audio inputs with preprocessing.
Whisper: Versatile model with robust transcription capabilities.

## **🚧 Challenges Faced**
High Execution Time for Feature Extraction:

Processing large audio files with a high sample rate was computationally intensive.
Solution: Resampled audio to reduce computational load.
High WER with Wav2Vec 2.0:

Background noise severely impacted transcription accuracy.
Solution: Switched to the Whisper model for better performance.
Understanding MFCCs and Spectrograms:

Required in-depth research to comprehend Fourier transforms, Mel scales, and DCT.
Audio Length Variation:

Some audio files were too short or long for the models.
Solution: Applied padding and trimming to standardize input lengths.

## **🎯 Impact**
Improved Accessibility:

Enables transcription of audio into text for individuals with hearing impairments.
Enhanced Productivity:

Automates tasks like meeting transcription and content creation.
Educational Benefits:

Provides insights into audio signal processing and practical applications of machine learning.
Scalable Applications:

Forms the foundation for voice assistants, transcription services, and more.
Hands-On Experience:

Strengthened understanding of ML concepts, feature extraction, and model evaluation.

## **💡 Skills Gained**
Audio signal processing and feature engineering.
Hands-on experience with Wav2Vec 2.0 and Whisper.
Implementing advanced preprocessing techniques for noisy datasets.
Performance evaluation through metrics like WER.
Practical knowledge of transfer learning and deep learning frameworks.
Efficient timeline management and debugging complex models.
