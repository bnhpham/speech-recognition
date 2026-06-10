import scipy
import recognizer.tools as tools
import numpy as np


# Make frames for a given audio file
def make_frames(audio_data, sampling_rate, window_size, hop_size):
    # Convert seconds to samples
    window_size = tools.dft_window_size(window_size, sampling_rate)
    hop_size = tools.sec_to_samples(hop_size, sampling_rate)

    # Zero-padding & overlapping frames
    num_frames = tools.get_num_frames(len(audio_data), window_size, hop_size)
    audio_data = np.pad(audio_data, (0, (num_frames-1) * hop_size + window_size - len(audio_data)), constant_values=(0))
    frames = np.array([audio_data[i : i + window_size] for i in range(0, len(audio_data) - window_size + 1, hop_size)])

    # Hamming
    hamming = np.hamming(window_size)
    return frames * hamming


# Return the non-redundant part of the magnitude spectrum
def compute_absolute_spectrum(frames):
    return np.array([np.abs(np.fft.rfft(f)) for f in frames])


# Main feature extraction function
def compute_features(audio_file, window_size=25e-3, hop_size=10e-3,
                     feature_type='STFT', n_filters=24, fbank_fmin=0, fbank_fmax=8000, num_ceps=13):
    
    # Audio reading
    sampling_rate, audio_data = scipy.io.wavfile.read(audio_file)
    
    # Normalization (values are within [-1,1])
    audio_data = audio_data / np.max(np.abs(audio_data))

    # Make frames
    frames = make_frames(audio_data, sampling_rate, window_size, hop_size)
    
    # Extract features depending on feature_type
    # Short-time Fourier transform
    if feature_type == 'STFT':
        return compute_absolute_spectrum(frames)
    
    # Mel filter bank
    elif feature_type == 'FBANK':
        mel_filterbank = get_mel_filters(sampling_rate, window_size, n_filters, fbank_fmin, fbank_fmax)
        abs_spectrum = compute_absolute_spectrum(frames)
        apply = apply_mel_filters(abs_spectrum, mel_filterbank)

        zero_to_min_val = np.maximum(apply, np.finfo(float).eps)
        return np.log(zero_to_min_val)
    
    # Mel-frequency cepstral coefficients
    elif feature_type == 'MFCC':
        mel_filterbank = get_mel_filters(sampling_rate, window_size, n_filters, fbank_fmin, fbank_fmax)
        abs_spectrum = compute_absolute_spectrum(frames)
        mel_spec = apply_mel_filters(abs_spectrum, mel_filterbank)

        return compute_cepstrum(mel_spec, num_ceps)
    
    # Mel-frequency cepstral coefficients supplemented by their first derivatives
    elif feature_type == 'MFCC_D':
        mel_filterbank = get_mel_filters(sampling_rate, window_size, n_filters, fbank_fmin, fbank_fmax)
        abs_spectrum = compute_absolute_spectrum(frames)
        mel_spec = apply_mel_filters(abs_spectrum, mel_filterbank)

        mfcc = compute_cepstrum(mel_spec, num_ceps)
        delta = get_delta(mfcc)

        return append_delta(mfcc, delta)
    
    # Mel-frequency cepstral coefficients supplemented by their first and second derivatives
    elif feature_type == 'MFCC_D_DD':
        mel_filterbank = get_mel_filters(sampling_rate, window_size, n_filters, fbank_fmin, fbank_fmax)
        abs_spectrum = compute_absolute_spectrum(frames)
        mel_spec = apply_mel_filters(abs_spectrum, mel_filterbank)

        mfcc = compute_cepstrum(mel_spec, num_ceps)
        delta = get_delta(mfcc)
        mfcc_d = append_delta(mfcc, delta)
        delta = get_delta(delta)

        return append_delta(mfcc_d, delta)
    else:
        # if feature_type is invalid
        return -1


# Triangular filter bank
def get_mel_filters(sampling_rate, window_size_sec, n_filters, f_min=0, f_max=8000):
    max = tools.hz_to_mel(f_max)
    min = tools.hz_to_mel(f_min)

    # Define points equidistant in the mel-frequency domain
    f_points_mel = np.linspace(min, max, n_filters + 2)

    # Frequency support points
    f_points_hz = tools.mel_to_hz(f_points_mel)

    # N = FFT size
    N = tools.dft_window_size(window_size_sec, sampling_rate)

    # f = DFT indices
    f = np.round(N * f_points_hz / sampling_rate)
    f[0] = 0
    f[n_filters+1] = N // 2

    # H = Mel filter bank / triangular filter
    H = np.zeros((n_filters, N//2 + 1))

    for m in range(1, n_filters+1):
        for k in range(0, N//2 +1):
            if k < f[m-1]:
                H[m-1,k] = 0
            elif f[m-1] <= k < f[m]:
                H[m-1,k] = (2*(k-f[m-1])) / ((f[m+1]-f[m-1])*(f[m]-f[m-1]))
            elif f[m] <= k <= f[m+1]:
                H[m-1,k] = (2*(f[m+1]-k)) / ((f[m+1]-f[m-1])*(f[m+1]-f[m]))
            elif k > f[m+1]:
                H[m-1,k] = 0
    return H


# Apply triangular filter bank to a magnitude spectrum
def apply_mel_filters(abs_spectrum, filterbank):
    return np.matmul(abs_spectrum, filterbank.T)


# Compute real cepstrum
def compute_cepstrum(mel_spectrum, num_ceps):
    # 1. |x_mel|
    mel_spectrum = np.abs(mel_spectrum)

    # 2. log()
    mel_spectrum = np.maximum(mel_spectrum, np.finfo(float).eps)
    mel_spectrum = np.log(mel_spectrum)

    # 3. DCT()
    mel_spectrum = scipy.fftpack.dct(mel_spectrum, norm='ortho')

    return mel_spectrum[: , :num_ceps]


# Compute first derivative of a given cepstrum
def get_delta(x):
    rslt = np.copy(x)

    rslt[0] = x[1] - x[0]
    rslt[x.shape[0] - 1] = x[x.shape[0] - 1] - x[x.shape[0] - 2]

    for i in range(1,x.shape[0] - 1):
        rslt[i] = 0.5 * (x[i + 1] - x[i - 1])
    
    return rslt


# Extend feature vector by its first derivative
def append_delta(x, delta):
    return np.concatenate((x, delta), axis=1)
