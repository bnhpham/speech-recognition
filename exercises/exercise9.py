import recognizer.hmm as HMM
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    # default HMM
    hmm = HMM.HMM()

    #statesequence = [0, 0, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 0, 31, 32, 33, 34, 35, 36, 0, 37, 38, 39, 40, 41, 42, 43, 44, 45, 0]
    #statesequence = [0, 1, 1, 2, 2, 3]
    #statesequence = [1, 2, 3, 3, 31, 32, 33, 34, 35, 36, 36, 1, 2, 3, 0]
    statesequence = [31, 32, 33, 34, 35, 36, 0, 1, 2, 3, 1, 2, 3, 31, 32, 33, 34, 35, 36]
    
    words = hmm.getTranscription(statesequence)
    print(words) # ['ONE', 'TWO', 'THREE']

    for i in range(hmm.logA.shape[0]):
        sum = 0.0
        for j in range(hmm.logA.shape[0]):
            sum += np.exp(hmm.logA[i, j])
            
        if not np.isclose(sum, 1.0):
            print(f"sum != 1.0 in i: {i}")

    sum_pi = 0.0
    for log_val in hmm.logPi:
        sum_pi += np.exp(log_val)
    
    if sum_pi != 1.0:
        print("sum_pi != 1.0")


    plt.imshow(np.exp(hmm.logA))
    plt.xlabel('nach Zustand j')
    plt.ylabel('von Zustand i')
    plt.colorbar()
    plt.show()
