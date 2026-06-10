import numpy as np
import torch
from recognizer.utils import *
from matplotlib import pyplot as plt
import random

SEED=42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def main():
    parameters = {
        "window_size": 25e-3,
        "hop_size": 10e-3,
        "feature_type": "MFCC_D_DD",
        "n_filters": 40,
        "fbank_fmin": 0,
        "fbank_fmax": 8000,
        "num_ceps": 13,          
        "left_context": 10,
        "right_context": 10
    }
    
    # Create dataloader for development set
    traindict, devdict, testdict = get_metadata("dataset")
    dev_dataset = Dataloader(devdict, parameters)
    data_loader_dev = torch.utils.data.DataLoader(dev_dataset, shuffle=True, batch_size=1)

    # Iterate through the dataloader. However, only pick the first two samples for plotting
    loader_iter = iter(data_loader_dev)
    for i in range(len(dev_dataset)):
        feat, label, filename = next(loader_iter)

        if i == 0:
            feat_first = feat
            label_first = label
            filename_first = filename
        if i == 1:
            feat_second = feat
            label_second = label
            filename_second = filename
            break
    
    print(f"feat_first: {feat_first.shape}")
    print(f"label_first: {label_first.shape}")
    print(f"feat_second: {feat_second.shape}")
    print(f"label_second: {label_second.shape}")

    ##### Plot the labels #####

    # Remove batch dimension
    label_first = label_first.squeeze()
    label_second = label_second.squeeze()

    fig, axes = plt.subplots(2, 1, figsize=(8, 6))

    # First sample
    img_first = axes[0].imshow(label_first.T, origin='lower', aspect='auto', extent=(0, label_first.shape[0], 0, label_first.shape[1]))
    axes[0].set_xlabel("Frames")
    axes[0].set_ylabel("HMM states")
    axes[0].set_title(f"\'{filename_first[0]}\'")

    # Second sample
    img_second = axes[1].imshow(label_second.T, origin='lower', aspect='auto', extent=(0, label_second.shape[0], 0, label_second.shape[1]))
    axes[1].set_xlabel("Frames")
    axes[1].set_ylabel("HMM states")
    axes[1].set_title(f"\'{filename_second[0]}\'")

    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    main()