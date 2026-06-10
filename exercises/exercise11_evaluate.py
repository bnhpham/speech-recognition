import argparse
import recognizer.hmm as HMM
import recognizer.train as train
import recognizer.tools as tools
import numpy as np
import torch
import os


def test_model(datadir, hmm, model, parameters):
    # Load test files
    lab_dir = os.path.join(datadir, "TEST", "lab")
    wav_dir = os.path.join(datadir, "TEST", "wav")
    test_files = []

    for lab_file in os.listdir(lab_dir):
        wav_file = lab_file.replace(".lab", ".wav")

        lab_path = os.path.join(lab_dir, lab_file)
        wav_path = os.path.join(wav_dir, wav_file)

        test_files.append((lab_path, wav_path))

    # As iterating through all test samples would take a long time, select 50 samples randomly to approximate the wort error rate
    np.random.seed(0)
    samples = np.random.choice(len(test_files), 50, replace=False)

    total_N = 0
    total_D = 0
    total_I = 0
    total_S = 0

    # For each sample, predict a transcription
    for i in samples:
        lab_path, wav_path = test_files[i]

        with open(lab_path, 'r') as f:
            ref_seq = f.read().lower().split()

        posteriors = train.wav_to_posteriors(model, wav_path, parameters)
        word_seq = hmm.posteriors_to_transcription(posteriors)

        N, D, I, S = tools.needlemann_wunsch(ref_seq, word_seq)
        total_N += N
        total_D += D
        total_I += I
        total_S += S

        # Print
        print('--' * 40)
        print(wav_path)
        print("REF: ", ref_seq)
        print("OUT: ", word_seq)
        print(f"I: {I}   D: {D}    S: {S}  N: {N}")
        cur_wer = 100 * ((total_D + total_I + total_S) / total_N)
        print("Current Total WER: {}".format(cur_wer))

    # Word error rate
    wer = 100 * ((total_D + total_I + total_S) / total_N)
    return wer

def get_args():
    parser = argparse.ArgumentParser()
    # get arguments from outside
    parser.add_argument('--sourcedatadir', default='./TIDIGITS-ASE', type=str, help='Dir saves the datasource information')
    parser.add_argument('--savedir', default='./trained', type=str, help='Dir to save trained model and results')
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    # parse arguments
    # data directory, e.g., /media/public/TIDIGITS-ASE
    # call:
    # python uebung10.py <data/dir>
    # e.g., python uebung11.py /media/public/TIDIGITS-ASE
    args = get_args()
    print("Arguments:")
    [print(key, val) for key, val in vars(args).items()]

    datadir = args.sourcedatadir
    savedir = args.savedir

    # Parameters for the feature extraction
    parameters = {'window_size': 25e-3,
        'hop_size': 10e-3,
        'feature_type': 'MFCC_D_DD',
        'n_filters': 40,
        'fbank_fmin': 0,
        'fbank_fmax': 8000,
        'num_ceps': 13,
        'left_context': 10,
        'right_context': 10}

    # Default HMM
    hmm = HMM.HMM()

    posteriors = np.load('data/TEST-WOMAN-BF-7O17O49A.npy')

    # Transcription for the specified probabilities
    words = hmm.posteriors_to_transcription(posteriors)
    print('Given posteriori OUT: {}'.format(words))            # OUT: [’SEVEN’, ’OH’, ’ONE’, ’SEVEN’, ’OH’, ’FOUR’, ’NINE’]

    # Trained model from exercise 7
    model_name = "bestaccmodel_Optimized"
    model_dir = os.path.join(savedir, 'model', model_name + '.pkl')
    model = torch.load(model_dir, weights_only=False)

    # Example wav File
    test_audio = './TIDIGITS-ASE/TEST/wav/TEST-WOMAN-BF-7O17O49A.wav'
    
    # Apply DNN
    posteriors = train.wav_to_posteriors(model, test_audio, parameters)
    words = hmm.posteriors_to_transcription(posteriors)

    print('OUT: {}'.format(words))  # OUT: [’SEVEN’, ’OH’, ’ONE’, ’SEVEN’, ’OH’, ’FOUR’, ’NINE’]

    # Test DNN
    wer = test_model(datadir, hmm, model, parameters)
    print('--' * 40)
    print("Total WER: {}".format(wer))
