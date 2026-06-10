import recognizer.feature_extraction as fe
import numpy as np
import matplotlib.pyplot as plt
import scipy



def compute_features():
    audio_file = 'data/TEST-MAN-AH-3O33951A.wav'
    
    # Compute features
    hop_size = 0.01
    feat = fe.compute_features(audio_file, window_size=0.025, hop_size=hop_size)
    sampling_rate, _ = scipy.io.wavfile.read(audio_file)
        
    # Convert spectrogram to logarithmic scale
    feat = 20 * np.log10(feat)
    
    # Plot spectrogram
    plt.figure(figsize=(10, 6))
    plt.imshow(feat.T, origin='lower', aspect='auto',
               extent=(0, feat.shape[0] * hop_size, 0, sampling_rate / 2))
    plt.xlabel("Time in Seconds")
    plt.ylabel("Frequency in Hz")
    plt.colorbar()
    plt.show()

    
    

if __name__ == "__main__":
    # Spectral analysis
    compute_features()
    

    
    
    
