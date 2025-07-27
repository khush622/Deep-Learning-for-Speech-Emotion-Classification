# Emotion-detection-speech-recognition
🎧 Emotion Recognition from Audio using 1D CNN
This project focuses on recognizing emotions from audio clips using MFCC features and a 1D Convolutional Neural Network (CNN).

📋 Overview
✅ Emotion Labels Extracted from File Names
Emotion categories are parsed directly from the filenames of the audio samples.

🎵 Feature Extraction
Each audio file is processed to extract 50 MFCCs (Mel-frequency cepstral coefficients), which are saved in an array along with corresponding emotion labels.

📏 Input Standardization
All feature vectors are padded to a uniform length to ensure consistent input shape for model training.

🧠 Model Architecture
A 1D Convolutional Neural Network (CNN) is used to learn emotion representations from the MFCCs and perform classification.

📁 Repository Contents
code_file.ipynb: Main Jupyter notebook with data preprocessing, model training, and evaluation.

model.h5: Trained model saved at the checkpoint with highest validation accuracy.

testing.py: Script to test the model by simply providing a directory path containing audio files.
