import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import gender_classification.local.feature_extraction as fe


def get_metadata(datadir):
    """
    get_metadata() load the meta data in dictionary. 
    Input:
        datadir: <string> the folder containing the meta data
    Return: 
        The dictionaries of the training, dev and test set
    """
    
    # read "train.json" as traindict
    with open(datadir + '/train.json', 'r') as file:
        traindict = json.load(file)

    # read "dev.json" as devdict
    with open(datadir + '/dev.json', 'r') as file:
        devdict = json.load(file)

    # read "test.json" as testdict
    with open(datadir + '/test.json', 'r') as file:
        testdict = json.load(file)

    return traindict, devdict, testdict

def padding(sequences):
    '''
    To pad different sequences into a padded tensor for training. The main purpose of this function is to separate different sequence, pad them in different ways and return padded sequences.
    Input:
        sequences <list>:  The length of sequences is the same as the batch size. Each element in sequences list is a tuple with
                            a length of 3, representing the audio feature sequence in index 0, labels sequence in index 1,
                            filename sequence in index 2 (same order as the output of the sub-function def __getitem__(self, index)).
    Return:
        audio_feat_sequence <torch.FloatTensor>: The padded audio feature sequence (works with batch_size >= 1).
        label_sequence <torch.FloatTensor>: The padded label sequence (works with batch_size >= 1).
        filename_sequence <list>: The filename sequence (works with batch_size >= 1).
    '''
    
    # Save the audio features, ground-truth labels and filename
    # of each sample in one batch in the corresponding list
    audio_feat_sequence = [sequences[i][0].float() for i in range(len(sequences))]
    label_sequence = torch.FloatTensor([sequences[i][1] for i in range(len(sequences))])
    filename_sequence = [sequences[i][2] for i in range(len(sequences))]

    # Zero-padding audio features
    audio_feat_sequence = pad_sequence(audio_feat_sequence, batch_first=True)
    
    return audio_feat_sequence, label_sequence, filename_sequence

class Dataloader(Dataset):  # For instantiating train, validation and test dataset
    def __init__(self, datadict, feat_params):
        super(Dataloader).__init__()
        self.datakeys = self._get_keys(datadict)
        self.datadict = datadict                        # The Python-dictionary which imported from the json file
        self.window_size = feat_params[0]               # The hyper-parameters for feature extraction
        self.hop_size = feat_params[1]
        self.feature_type = feat_params[2]
        self.n_filters = feat_params[3]
        self.fbank_fmin = feat_params[4]
        self.fbank_fmax = feat_params[5]
        self.max_len = feat_params[6]
    def _get_keys(self, datadict):
        keys = list(datadict.keys())
        return keys

    def __len__(self):
        return len(self.datakeys)

    def __getitem__(self, index):
        filename = self.datakeys[index]                                     # The audio sample file name

        sample = self.datadict[filename]

        # Get audio sample path
        sample_path = sample["sampledir"]

        # Get label and move label to FloatTensor
        label = torch.FloatTensor([sample["label"]])

        # Extract audio features by the self-programmed feature extractor
        audio_feat = fe.compute_features(sample_path, window_size=self.window_size, hop_size=self.hop_size, feature_type=self.feature_type,
                                        n_filters=self.n_filters, fbank_fmin=self.fbank_fmin, fbank_fmax=self.fbank_fmax)

        # Move numpy audio features to FloatTensor, the tensor has dimension of [s_length, 60], where s_length is the sequence length. 
        # Crop the audio features, if the frame length longer than defined max_len
        audio_feat = torch.from_numpy(audio_feat)[:self.max_len].float()

        return audio_feat, label, filename
