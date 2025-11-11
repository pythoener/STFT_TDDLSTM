import os
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.signal
import math
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error
from scipy.io.wavfile import write
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from scipy.io import wavfile
from tensorflow import keras
from sklearn.model_selection import train_test_split
from scipy.signal import butter, filtfilt
from pesq import pesq
from pystoi import stoi
from tensorflow.keras.layers import Input, Conv1D, Conv1DTranspose, BatchNormalization, MaxPooling1D, LSTM, Dropout, UpSampling1D, Flatten, Dense, Reshape
from tensorflow.keras.models import Model
import tensorflow as tf
import joblib

# 1) Print whether this build has CUDA support:
print("Built with CUDA:", tf.test.is_built_with_cuda())

# 2) List all visible GPU devices:
print("GPUs visible:   ", tf.config.list_physical_devices('GPU'))

#%% helper functions unchanged

def plot_spectrogram(signal, sample_rate, title='Spectrogram'):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(signal, n_fft=1024, hop_length=128)), ref=np.max)
    plt.figure(figsize=(10, 3))
    librosa.display.specshow(D, sr=sample_rate, hop_length=128, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.ylabel('Hz')
    plt.show()

def apply_fft(signal, fs, a):
    fft_signal = np.fft.fft(signal)
    n = len(signal)
    freq = np.fft.fftfreq(n, 1/fs)
    plt.figure(figsize=(10, 3))
    plt.plot(freq, np.abs(fft_signal))
    plt.title(f'FFT Magnitude of {a}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.show()

def lowpass_filter(data, cutoff, fs=16000, order=50):
    nyquist = fs / 2
    normalized_cutoff = cutoff / nyquist
    sos = scipy.signal.butter(order, normalized_cutoff, btype='low', output='sos')
    return scipy.signal.sosfilt(sos, data)

def extract_stft(frame, sample_rate, n_fft=1024, hop_length=128, win_length=None):
    if win_length is None:
        win_length = n_fft
    S = librosa.stft(frame, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window="hamming")
    mag   = 20 * np.log10(np.abs(S) + 1e-8)
    phase = np.angle(S)
    return mag, phase

def load_audio(file_path):
    try:
        audio_signal, fs = sf.read(file_path)
    except Exception as e:
        print(f"Soundfile failed: {e}")
        try:
            audio = AudioSegment.from_file(file_path)
            audio_signal = np.array(audio.get_array_of_samples())
            fs = audio.frame_rate
        except Exception as e2:
            print(f"Pydub also failed: {e2}")
            return None, None
    return audio_signal, fs

#%% PARAMETERS

fs = 16000
n_fft = 1024
hop_length = 128
target_folder = r'E:\Raw Data\VCTK_16kHz\train\clean'

# First pass: count total frames across all files
def count_frames(file_path):
    sig, sr = load_audio(file_path)
    if sig is None:
        return 0
    frame_length = int(0.032 * sr)
    pad = (frame_length - len(sig) % frame_length) % frame_length
    sig = np.pad(sig, (0, pad), mode='constant')
    mag, _ = extract_stft(sig.astype(np.float32), sr, n_fft=n_fft, hop_length=hop_length)
    return mag.shape[1]  # number of frames

# gather wav files
wav_paths = []
for root, _, files in os.walk(target_folder):
    for f in files:
        if f.lower().endswith('.wav'):
            wav_paths.append(os.path.join(root, f))

# compute total_frames
total_frames = sum(count_frames(fp) for fp in wav_paths)
n_bins = n_fft//2 + 1

# create memmaps
low_mm  = np.memmap(r'G:\low_stft.dat',  dtype='float32', mode='w+', shape=(total_frames, n_bins))
high_mm = np.memmap(r'G:\high_stft.dat', dtype='float32', mode='w+', shape=(total_frames, n_bins))
write_idx = 0

#%% Second pass: fill memmaps

def create_dataset_stft(file_path):
    global write_idx
    audio_signal, sr = load_audio(file_path)
    if audio_signal is None or sr is None:
        return 0
    frame_length = int(0.032 * sr)
    pad_width = (frame_length - len(audio_signal) % frame_length) % frame_length
    audio_signal = np.pad(audio_signal, (0, pad_width), mode='constant')

    # high STFT
    high_mag, _ = extract_stft(audio_signal.astype(np.float32), sr, n_fft=n_fft, hop_length=hop_length)
    # low STFT
    low_audio   = lowpass_filter(audio_signal, cutoff=4000, fs=sr)
    low_mag, _  = extract_stft(low_audio.astype(np.float32), sr, n_fft=n_fft, hop_length=hop_length)

    n = high_mag.shape[1]
    # write into memmaps
    high_mm[write_idx:write_idx+n, :] = high_mag.T
    low_mm [write_idx:write_idx+n, :] = low_mag.T
    write_idx += n
    return n

# process all files
for fp in wav_paths:
    create_dataset_stft(fp)

# flush to disk
low_mm.flush()
high_mm.flush()

# ── 1) PREPARE SCALER STATS ─────────────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler

# We can fit directly on the memmaps:
feature_scaler = StandardScaler().fit(low_mm)
label_scaler   = StandardScaler().fit(high_mm)

# immediately save both scalers to disk:
joblib.dump(feature_scaler, 'feature_scaler_stft_vctk.pkl')
joblib.dump(label_scaler,   'label_scaler_stft_vctk.pkl')

mean_lo, scale_lo = feature_scaler.mean_, feature_scaler.scale_
mean_hi, scale_hi = label_scaler.mean_,   label_scaler.scale_

# ── 2) REPLAY FILE FRAME COUNTS ────────────────────────────────────────────────
# (You can collect these counts during your write pass;
# here we just re-compute with your count_frames function.)

file_frame_counts = [count_frames(fp) for fp in wav_paths]

WINDOW_SIZE = 5
n_bins      = n_fft//2 + 1

# ── 3) WINDOW GENERATOR ─────────────────────────────────────────────────────────
def window_generator():
    idx = 0
    for n in file_frame_counts:
        # slide within this file’s block of frames
        for offset in range(n - WINDOW_SIZE + 1):
            start = idx + offset
            x = low_mm [start : start+WINDOW_SIZE]   # (5, 513)
            y = high_mm[start : start+WINDOW_SIZE]
            # normalize on the fly
            x = (x - mean_lo[None, :]) / scale_lo[None, :]
            y = (y - mean_hi[None, :]) / scale_hi[None, :]
            yield x.astype(np.float32), y.astype(np.float32)
        idx += n

# ── 4) BUILD A TF.DATA PIPELINE ────────────────────────────────────────────────
import tensorflow as tf

output_signature = (
    tf.TensorSpec(shape=(WINDOW_SIZE, n_bins), dtype=tf.float32),
    tf.TensorSpec(shape=(WINDOW_SIZE, n_bins), dtype=tf.float32),
)

dataset = tf.data.Dataset.from_generator(
    window_generator,
    output_signature=output_signature
)

# split into train/val/test by counts
total_windows = sum(n - WINDOW_SIZE + 1 for n in file_frame_counts)
train_cnt = int(0.8 * total_windows)
val_cnt   = int(0.1 * total_windows)

train_ds = (
    dataset
    .take(train_cnt)
    .shuffle(10_000)
    .batch(64)
    .prefetch(tf.data.AUTOTUNE)
)
val_ds = (
    dataset
    .skip(train_cnt)
    .take(val_cnt)
    .batch(64)
    .prefetch(tf.data.AUTOTUNE)
)
test_ds = (
    dataset
    .skip(train_cnt + val_cnt)
    .batch(64)
    .prefetch(tf.data.AUTOTUNE)
)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Dense, LSTM
from tensorflow.keras.optimizers import Adam

# ── 5) DEFINE & COMPILE YOUR MODEL ─────────────────────────────────────────────
model = Sequential([
    TimeDistributed(Dense(256, activation='relu'), input_shape=(WINDOW_SIZE, n_bins)),
    TimeDistributed(Dense(128, activation='relu')),
    LSTM(128, return_sequences=True),
    TimeDistributed(Dense(128, activation='relu')),
    TimeDistributed(Dense(256, activation='relu')),
    TimeDistributed(Dense(n_bins, activation='linear')),
])
model.compile(optimizer=Adam(1e-3), loss='mse', metrics=['mae'])
model.summary()


# ── 5) TRAIN & EVALUATE YOUR MODEL ─────────────────────────────────────────────
history = model.fit(
    train_ds,
    epochs=50,
    validation_data=val_ds
)

# plot your losses (same as before)
plt.figure(figsize=(10,6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.legend(); plt.grid(True)


test_loss, test_mae = model.evaluate(test_ds)
print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

model.save('Model_TDDLSTM_VCTK.keras')
plt.show()