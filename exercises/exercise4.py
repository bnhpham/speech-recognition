import recognizer.feature_extraction as fe
import matplotlib.pyplot as plt

if __name__ == "__main__":
    audio_file = 'data/TEST-MAN-AH-3O33951A.wav'

    hop_size = 0.01

    # MFCC
    mfcc = fe.compute_features(audio_file, feature_type='MFCC')

    plt.figure(figsize=(10, 6))
    plt.imshow(mfcc.T, origin='lower', aspect='auto',
               extent=(0, mfcc.shape[0] * hop_size, 0, mfcc.shape[1]))
    plt.xlabel("Time in Seconds")
    plt.ylabel("MFCC index")
    plt.colorbar()
    plt.show()
    
    # MFCC+D
    mfcc_d = fe.compute_features(audio_file, feature_type='MFCC_D')

    plt.figure(figsize=(10, 6))
    plt.imshow(mfcc_d.T, origin='lower', aspect='auto',
               extent=(0, mfcc_d.shape[0] * hop_size, 0, mfcc_d.shape[1]))
    plt.xlabel("Time in Seconds")
    plt.ylabel("MFCC+D index")
    plt.colorbar()
    plt.show()

    # MFCC+D+DD
    mfcc_d_dd = fe.compute_features(audio_file, feature_type='MFCC_D_DD')
    
    plt.figure(figsize=(10, 6))
    plt.imshow(mfcc_d_dd.T, origin='lower', aspect='auto',
               extent=(0, mfcc_d_dd.shape[0] * hop_size, 0, mfcc_d_dd.shape[1]))
    plt.xlabel("Time in Seconds")
    plt.ylabel("MFCC+D+DD index")
    plt.colorbar()
    plt.show()