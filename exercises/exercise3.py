import recognizer.feature_extraction as fe
import matplotlib.pyplot as plt


if __name__ == "__main__":
    
    # Plot triangular filter
    filter = fe.get_mel_filters(16000, 0.025, 24)
    plt.figure(figsize=(10, 5))
    for i in range(0,24):
        plt.plot(filter[i])
    plt.show()
    
    # Mel spectrum
    audio_file = 'data/TEST-MAN-AH-3O33951A.wav'
    hop_size = 0.01
    feat = fe.compute_features(audio_file, window_size=0.025, hop_size=hop_size,feature_type='FBANK')

    # Plot spectrum
    plt.figure(figsize=(10, 6))
    plt.imshow(feat.T, origin='lower', aspect='auto',
               extent=(0, feat.shape[0] * hop_size, 0, feat.shape[1]))
    plt.xlabel("Time in Seconds")
    plt.ylabel("Mel filter index")
    plt.colorbar()
    plt.show()