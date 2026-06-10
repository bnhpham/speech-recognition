import math
import numpy as np


# Convert a scalar time value x (in seconds) given a sampling rate (in Hz) into the corresponding discrete sample point
def sec_to_samples(x, sampling_rate):
    return int(x * sampling_rate)

# Return the next power of 2 greater than x, e.g. next_pow2(300) = 9
def next_pow2(x):
    log = math.log2(x)
    return int(log+1) if int(log) < log else int(log)

# Compute the number of samples based on a duration x (in sec) and the sampling rate (in Hz)
def dft_window_size(x, sampling_rate):
    return 2 ** next_pow2(sec_to_samples(x, sampling_rate))

# Compute the number of required frames
def get_num_frames(signal_length_samples, window_size_samples, hop_size_samples):
    o = window_size_samples - hop_size_samples
    k =  (signal_length_samples - o) / hop_size_samples
    return int(k+1) if int(k) < k else int(k)

# Convert a frequency value x (in Hz) into the corresponding value on the Mel scale
def hz_to_mel(x):
    return 2595 * np.log10(1 +(x / 700))

# Conversion of a frequency x from the Mel scale to the linear frequency scale (Hz)
def mel_to_hz(x):
    return 700 * (-1 + np.power(10, x / 2595))