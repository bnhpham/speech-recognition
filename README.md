# Grundlagen der automatischen Spracherkennung WiSe 2025 Programmierpraktikum

This project was carried out as a programming practical as part of the course "Grundlagen der automatischen Spracherkennung" (Foundations of Automatic speech Recognition) at TU Berlin. It implements a gender classification network distinguishing between male and female voices as well as a hybrid DNN/HMMs for automatic digit transcription.

## Key Features
- **Handcrafted Feature Extraction**: Implementation of speech processing algorithms including Short-Time Fourier Transform (STFT), Mel filterbanks, and Mel-Frequency Cepstral Coefficients (MFCC) supplemented with first and second derivatives as well as temporal context windowing.
- **Gender Classification**: Deep neural network optimized to distinguish binary vocal traits and verified across the VoxCeleb dataset.
- **DNN/HMM for digit recognition**: Hybrid deep neural network / Hidden Markov Model trained and tested the TIDIGITS dataset.
- **BLSTM/HMM for digit recognition**: Bidirectional Long Short-Term Memory / Hidden Markov Model trained and tested the TIDIGITS dataset.
- **Viterbi Decoding**: Implementation of a dynamic programming search algorithm to track the most likely sequence of HMM states.
- **Performance Benchmarking**: Computing the Word Error Rate (WER) to evaluate model performance using the Needleman–Wunsch algorithm.
- **Exercises**: Showcase, visualization and explanation of each audio processing part of this project including spectrogram plotting, model training and WER evaluation.

## Evaluation
| Model | Accuracy | WER |
| :--- | :--- | :--- |
| **Gender Classifier** | 0.9580 | - |
| **DNN/HMM** | 0.6422  | 3.27 |
| **BSTLM/HMM** | 0.7511 | 1.19 |

## Setup & Installation
Install required frameworks:
```
pip install torch numpy scipy tqdm matplotlib praatio
```

This project maps old properties belonging to praatio 4.x. If a WrongPraatioVersion exception triggers, run:
```
pip uninstall praatio
pip install "praatio<5"
```

