import recognizer.feature_extraction as fe
import scipy
import matplotlib.pyplot as plt
import numpy as np

def compute_features():
    audio_file = 'data/TEST-MAN-AH-3O33951A.wav'
    
    # Read in audio_file and call make_frames()
    sampling_rate, audio_data = scipy.io.wavfile.read(audio_file)
    frames = fe.make_frames(audio_data, sampling_rate, window_size=0.025, hop_size=0.010)
    
    audio_data = audio_data / np.max(np.abs(audio_data)) # Normalization
    frames = fe.make_frames(audio_data, sampling_rate, window_size=0.4, hop_size=0.25)

    # Discrete sampling points are converted back into scalar time values ​​(inverse of sec_to_samples())
    t = np.arange(frames.shape[1]) / sampling_rate

    # Plot first 4 frames    
    plt.figure(figsize=(12, 6))
    plt.suptitle("First 4 successive frames (normalized between -1 and 1)")

    for i in range(4):
        plt.subplot(4, 1, i+1)
        plt.plot(t, frames[i])
        plt.grid(True)
        plt.xlabel("Time in Seconds")

    plt.show()

if __name__ == "__main__":
    compute_features()
