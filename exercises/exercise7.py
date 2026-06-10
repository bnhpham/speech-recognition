import argparse
from recognizer.train import run
from recognizer.utils import *
import torch
import os

import random
SEED = 666
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def get_args():
    parser = argparse.ArgumentParser()
    # get arguments from outside
    parser.add_argument('--sourcedatadir', default='./TIDIGITS-ASE', type=str, help='Dir saves the datasource information')
    parser.add_argument('--datasdir', default='./dataset', type=str, help='Dir saves the datasource information')
    parser.add_argument('--savedir', default='./trained', type=str, help='Dir to save trained model and results')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    print("Arguments:")
    [print(key, val) for key, val in vars(args).items()]

    sourcedatadir = args.sourcedatadir
    datasetdir = args.datasdir
    savedir = args.savedir

    # If GPU on device available, use GPU to train the model
    if torch.cuda.is_available():
        device = 'cuda'     # use GPU
    else:
        device = 'cpu'      # use CPU

    # Create folders to save the trained models and evaluation results
    modeldir = os.path.join(savedir, 'model')
    resultsdir = os.path.join(savedir, 'results')
    for makedir in [modeldir, resultsdir, datasetdir]:
        if not os.path.exists(makedir):
            os.makedirs(makedir)

    # Load meta data as dictionary
    traindict, devdict, testdict = get_metadata(args.datasdir)

    # Config audio paths
    for dict in [traindict, devdict, testdict]:
        for k, v in dict.items():
            dict[k]["audiodir"] = dict[k]["audiodir"].replace('./TIDIGITS-ASE', sourcedatadir)

    # Configure hyperparameters
    config = {
        "NWORKER": 0,
        "device": device,
        "lr": 0.001,
        "batch_size": 1, 
        "epochs": 1, # 50
        "patience": 5,
        "window_size": 25e-3,
        "hop_size": 10e-3,

        "feature_type": 'MFCC_D_DD',
        "n_filters": 40,
        "fbank_fmin": 0,
        "fbank_fmax": 8000,
        "num_ceps": 13,
        "left_context": 10,
        "right_context": 10,
        "hidden_size": 512,
        "modeldir": modeldir,
        "resultsdir": resultsdir
    }
    run(config, datadicts=[traindict, devdict, testdict])