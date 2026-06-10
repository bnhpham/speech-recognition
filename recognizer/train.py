import os
import torch
from tqdm import tqdm
from recognizer.model import *
from recognizer.utils import *
import torch.nn.functional as F
from recognizer.hmm import *

import matplotlib.pyplot as plt
import scipy.io.wavfile

def train(dataset, model, device, odim, optimizer=None, criterion=None):

    model.to(device)
    model.train()
    
    total_correct = 0
    total_frames = 0
    
    for X, y, _ in tqdm(dataset, desc="Training", unit="samples"):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()

        # Forward Pass => [1, Frames, odim]
        outputs = model(X)

        # Since the Softmax function preserves the order of values,
        # the index of the largest logit will always be the same as the index of the largest probability.
        # To save computation time, we don't use Softmax.
        y_indices = torch.argmax(y, dim=2)
        loss = criterion(outputs.view(-1, odim), y_indices.view(-1).long())

        loss.backward()
        optimizer.step()

        predictions = torch.argmax(outputs, dim=2)
        
        total_correct += torch.sum(predictions == y_indices).item()
        total_frames += y_indices.numel()

    accuracy = total_correct / total_frames
    
    return accuracy, model

def evaluation(dataset, odim, model, device):

    model.to(device)
    model.eval()

    total_correct = 0
    total_frames = 0

    with torch.no_grad():
        for X, y, _ in tqdm(dataset, desc="Evaluation", unit="samples"):
            X, y = X.to(device), y.to(device)

            outputs = model(X)

            predictions = torch.argmax(outputs, dim=2)
            y_indices = torch.argmax(y, dim=2)

            total_correct += torch.sum(predictions == y_indices).item()
            total_frames += y_indices.numel()

    accuracy = total_correct / total_frames
                                        
    return accuracy, model

def run(config, datadicts=None):
    """
    run() trains and evaluates the model over given number of epochs.
    Input:
        config: the defined hyperparameters
        datadicts: the dictionary containing the meta-data for training, dev and test set.
    """
    traindict, devdict, testdict = datadicts  
    # Parameters for feature extraction
    feat_params = {
        "window_size": config["window_size"],
        "hop_size": config["hop_size"],
        "feature_type": config["feature_type"],
        "n_filters": config["n_filters"],
        "fbank_fmin": config["fbank_fmin"],
        "fbank_fmax": config["fbank_fmax"],
        "num_ceps": config["num_ceps"],          
        "left_context": config["left_context"],
        "right_context": config["right_context"]
    }

    # Create 3 datasets from given training, dev and test meta-data
    train_dataset = Dataloader(traindict, feat_params)
    dev_dataset = Dataloader(devdict, feat_params)
    test_dataset = Dataloader(testdict, feat_params)

    print("train_dataset: ", len(train_dataset))
    print("dev_dataset: ", len(dev_dataset))
    print(f"test_dataset: {len(test_dataset)}\n")

    resultsdir = config["resultsdir"]
    modeldir = config["modeldir"]

    # Parameters for early stopping
    evalacc_best = 0
    early_wait = config["patience"]
    run_wait = 1
    continuescore = 0
    stop_counter = 0

    # Define loss function, model and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    # calculating idim, odim for the model
    hmm = HMM()
    f_dim = 3 * config["num_ceps"]  # mal 3 wegen dem append_delta in feature_extraction.py
    c_dim = config["left_context"] + config["right_context"] + 1
    idim = f_dim * c_dim
    odim = hmm.get_num_states()

    model = DNN_Model(idim=idim, odim=odim, hidden_dim=config["hidden_size"])
    model = model.to(config["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # Pre-loading dataset
    data_loader_train = torch.utils.data.DataLoader(train_dataset,
                                                    shuffle=True,
                                                    batch_size=config["batch_size"],
                                                    num_workers=config["NWORKER"])
    data_loader_dev = torch.utils.data.DataLoader(dev_dataset, shuffle=True,
                                                      batch_size=config["batch_size"],
                                                      num_workers=config["NWORKER"])

    epochs = config["epochs"]
    for epoch in range(epochs):
        # Train model on training set
        trainscore, model = train(data_loader_train,
                                     model,
                                     config["device"],
                                     odim,
                                     optimizer=optimizer,
                                     criterion=criterion)
        # Evaluate trained model on dev set
        evalscore, model = evaluation(data_loader_dev, odim, model, config["device"])

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Acc = {trainscore:.4f} | "
              f"Dev Acc = {evalscore:.4f}.")


        # Implementation for early stopping: If the model accuracy on the dev set does not improve in
        # config["patience"] epochs, training is terminated.
        torch.cuda.empty_cache()
        if evalscore <= evalacc_best:
            stop_counter = stop_counter + 1
            print('No improvement.')
            continuescore = 0
        else:
            print('New eval score!')
            evalacc_best = evalscore
            continuescore = continuescore + 1
            OUTPUT_DIR = os.path.join(modeldir,'bestaccmodel.pkl') # save trained model
            torch.save(model, OUTPUT_DIR)

        if continuescore >= run_wait:
            stop_counter = 0
        print(f"Early Stopping Progress: {stop_counter}/{early_wait} (Best Accuracy: {evalacc_best:.4f}).")
        if stop_counter < early_wait:
            pass
        else:
            break

    # Model has trained as many epochs as specified (subject to possible early stopping).
    # Now, evaluate the model on test set:
    data_loader_test = torch.utils.data.DataLoader(test_dataset, shuffle=True,
                                                      batch_size=config["batch_size"],
                                                      num_workers=config["NWORKER"])

    # Load model
    model = torch.load(os.path.join(modeldir,
                          'bestaccmodel.pkl'),
                       map_location=config["device"],
                        weights_only=False)
    
    # Finally, evaluate the trained model on the test set.
    print("\nEvaluation of the best-trained model on the test set:")
    testscore, model = evaluation(data_loader_test, odim, model, config["device"])
    print(f"Test Acc = {testscore:.4f}.")

    # Example 
    sample_info = testdict["TEST-WOMAN-BF-7O17O49A"]
    audio_path = sample_info["audiodir"]
    target_path = sample_info["targetdir"]
    
    # Compute ground truth
    sampling_rate, _ = scipy.io.wavfile.read(audio_path)
    window_size_samples = tools.dft_window_size(config["window_size"], sampling_rate)
    hop_size_samples = tools.sec_to_samples(config["hop_size"], sampling_rate)
    
    hmm = HMM()
    ground_truth_label = tools.praat_file_to_target(target_path, sampling_rate, window_size_samples, hop_size_samples, hmm)
    
    # Compute posteriors
    predicted = wav_to_posteriors(model, audio_path, feat_params)
    
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), layout='constrained')
    fig.suptitle(f"{"TEST-WOMAN-BF-7O17O49A"}")

    vmin = min(ground_truth_label.min(), predicted.min())
    vmax = max(ground_truth_label.max(), predicted.max())

    # Plot ground truth
    im = axes[0].imshow(ground_truth_label.T, origin='lower', aspect='auto', 
                        extent=(0, ground_truth_label.shape[0], 0, ground_truth_label.shape[1]),
                        vmin=vmin, vmax=vmax)
    axes[0].set_xlabel("Frames")
    axes[0].set_ylabel("HMM states")
    axes[0].set_title("Ground-Truth-Labels")

    # Plot posterior
    axes[1].imshow(predicted.T, origin='lower', aspect='auto', 
                extent=(0, ground_truth_label.shape[0], 0, ground_truth_label.shape[1]),
                vmin=vmin, vmax=vmax)
    axes[1].set_xlabel("Frames")
    axes[1].set_ylabel("HMM states")
    axes[1].set_title("A-posteriori Wahrscheinlichkeiten")

    fig.colorbar(im, ax=axes)
    plt.subplots_adjust(hspace=0.4)
    plt.show()

# Returns the DNN output for an audio file
def wav_to_posteriors(model, audio_file, parameters):
    device = next(model.parameters()).device
    model.eval()
    
    feats = fe.compute_features_with_context(audio_file, **parameters)
    
    x = torch.FloatTensor(feats).to(device)
    
    x = x.unsqueeze(0)
    
    with torch.no_grad():
        logits = model(x)
        posteriors = F.softmax(logits, dim=2)
    
    return posteriors.numpy()


