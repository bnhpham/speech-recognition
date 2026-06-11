# Grundlagen der automatischen Spracherkennung WiSe 2025 Programmierpraktikum

This project was carried out as a programming practical as part of the course "Grundlagen der automatischen Spracherkennung" (Foundations of Automatic Speech Recognition) at TU Berlin. It implements a gender classification network distinguishing between male and female voices as well as hybrid DNN/HMMs for automatic digit transcription.

## Key Features
- **Handcrafted Feature Extraction**: Implementation of speech processing algorithms including Short-Time Fourier Transform (STFT), Mel filterbanks, and Mel-Frequency Cepstral Coefficients (MFCC) supplemented with first and second derivatives as well as temporal context windowing.
- **Gender Classification**: Deep neural network optimized to distinguish binary vocal traits and verified across the VoxCeleb dataset.
- **DNN/HMM for digit recognition**: Hybrid deep neural network / hidden Markov model trained and tested on the TIDIGITS dataset.
- **BLSTM/HMM for digit recognition**: Bidirectional Long Short-Term Memory / hidden Markov model trained and tested on the TIDIGITS dataset.
- **Viterbi Decoding**: Implementation of a dynamic programming search algorithm to track the most likely sequence of HMM states.
- **Performance Benchmarking**: Evaluation of model performance using frame-level accuracy and Word Error Rate (WER) via the Needleman–Wunsch algorithm.
- **Exercises**: Visualization and step-by-step application of each audio processing part of this project including spectrogram plotting, model training and WER evaluation.

## Evaluation
The models were trained and tested using the VoxCeleb (gender classification) and TIDIGITS (digit transcription) datasets.

| Model | Accuracy | WER |
| :--- | :--- | :--- |
| **Gender Classifier** | 95.80% | — |
| **DNN/HMM** | 64.22%  | 3.27% |
| **BLSTM/HMM** | 75.11% | 1.19% |

## Setup & Installation
Install required frameworks:
```
pip install torch numpy scipy tqdm matplotlib praatio
```

This project maps old properties belonging to praatio 4.x. If a WrongPraatioVersion exception occurs, run:
```
pip uninstall praatio
pip install "praatio<5"
```
