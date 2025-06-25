import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

test_path = "path_to_test_data"

emotion_map = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

X_test_raw, y_test_raw = [], []
for actor in sorted(os.listdir(test_path)):
    actor_path = os.path.join(test_path, actor)
    if not os.path.isdir(actor_path): continue
    for fname in os.listdir(actor_path):
        if fname.endswith(".wav"):
            parts = fname.split("-")
            emotion_id = parts[2]
            emotion = emotion_map.get(emotion_id)
            if emotion is None:
                continue
            file_path = os.path.join(actor_path, fname)
            try:
                audio, _ = librosa.load(file_path, sr=None)
                X_test_raw.append(audio)
                y_test_raw.append(emotion)
            except Exception:
                pass

def extract_mfcc_batch(waveforms, sr=22050, n_mfcc=50):
    return [librosa.feature.mfcc(y=x, sr=sr, n_mfcc=n_mfcc).T for x in waveforms]

MAX_LEN = 598

def pad_or_trim_mfccs(mfccs, max_len=MAX_LEN):
    processed = []
    for m in mfccs:
        if m.shape[0] > max_len:
            m = m[:max_len, :]
        else:
            m = np.pad(m, ((0, max_len - m.shape[0]), (0, 0)), mode='constant')
        processed.append(m)
    return np.array(processed, dtype='float32')

mfccs_test = extract_mfcc_batch(X_test_raw)
X_test_padded = pad_or_trim_mfccs(mfccs_test)

encoder = LabelEncoder()
encoder.fit(list(emotion_map.values()))
y_test_encoded = encoder.transform(y_test_raw)

model = load_model("best_model.h5")

y_pred_probs = model.predict(X_test_padded)
y_pred = np.argmax(y_pred_probs, axis=1)

print("Test Accuracy:", accuracy_score(y_test_encoded, y_pred))
print(classification_report(y_test_encoded, y_pred, target_names=encoder.classes_))
