import json
import torch
from torch.utils.data import Dataset
import recognizer.feature_extraction as fe
from . import tools
import recognizer.hmm as HMM
import scipy


def get_metadata(datadir):
    """
    get_metadata() load the meta data in dictionary. 
    Input:
        datadir: <string> the folder containing the meta data
    Return: 
        The dictionaries of the training, dev and test set
    """
    
    # Read "train.json" as traindict
    with open(datadir + '/train.json', 'r') as file:
        traindict = json.load(file)

    # Read "dev.json" as devdict
    with open(datadir + '/dev.json', 'r') as file:
        devdict = json.load(file)

    # Read "test.json" as testdict
    with open(datadir + '/test.json', 'r') as file:
        testdict = json.load(file)

    return traindict, devdict, testdict

class Dataloader(Dataset):  # For instantiating train, validation and test dataset
    def __init__(self, datadict, parameters):
        super(Dataloader).__init__()
        self.datakeys = self._get_keys(datadict)
        self.datadict = datadict
        self.parameters = parameters
    def _get_keys(self, datadict):
        keys = list(datadict.keys())
        return keys

    def __len__(self):
        return len(self.datakeys)

    def __getitem__(self, index):
        # The audio sample file name
        filename = self.datakeys[index]

        sample = self.datadict[filename]

        # Get audio sample path
        sample_path = sample["audiodir"]

        # Get path to label file
        target_dir = sample["targetdir"]

        # Extract audio features by the self-programmed feature extractor
        audio_feat = fe.compute_features_with_context(sample_path, **self.parameters)

        # Create HMM & get ground-truth label
        hmm = HMM.HMM()

        sampling_rate, _ = scipy.io.wavfile.read(sample_path)
        window_size_samples = tools.dft_window_size(self.parameters["window_size"], sampling_rate)
        hop_size_samples = tools.sec_to_samples(self.parameters["hop_size"], sampling_rate)

        label = tools.praat_file_to_target(target_dir, sampling_rate, window_size_samples, hop_size_samples, hmm)

        # Convert numpy arrays to torch tensors
        audio_feat = torch.FloatTensor(audio_feat)
        label = torch.FloatTensor(label)

        return audio_feat, label, filename
