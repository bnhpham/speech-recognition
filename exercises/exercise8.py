import numpy as np


def viterbi( logLike, logPi, logA ):
  """
  logLike: (T, N)
  logPi: (N)
  logA: (N, N)
  """

  N = logPi.shape[0]      # Number of states
  T = logLike.shape[0]    # Number of timsteps

  # Initialization
  phi = np.zeros((T, N))
  psi = -np.ones((T, N))
  phi[0] = logPi + logLike[0]

  # Iteration
  for t in range(1, T):
     best_path_prob = np.max(phi[t-1][:, None] + logA, axis=0)
     phi[t] = best_path_prob + logLike[t]
     psi[t] = np.argmax(phi[t-1][:, None] + logA, axis=0)

  # Termination
  best_path = np.zeros((T), dtype=np.uint32)
  best_path[T-1] = np.argmax(phi[T-1], axis=0)
  pStar = phi[T-1][best_path[T-1]]

  # Backtracking
  for t in range(T-1, 0, -1):
     best_path[t-1] = psi[t][best_path[t]]

  return best_path, pStar


def limLog(x):
    """
    Log of x.

    :param x: numpy array.
    :return: log of x.
    """
    MINLOG = 1e-100
    return np.log(np.maximum(x, MINLOG))



if __name__ == "__main__":
    # Vector of initial state probabilities
    logPi = limLog([ 0.9, .0, 0.1 ])

    # Matrix of state transition probabilities
    logA  = limLog([
      [ 0.8,  .0, 0.2 ], 
      [ 0.4, 0.4, 0.2 ], 
      [ 0.3, 0.2, 0.5 ] 
    ]) 

    # Observation probabilities for "Rain", "Sun", "Snow" 
    # B = [
    #     {  2: 0.1,  3: 0.1,  4: 0.2,  5: 0.5,  8: 0.1 },
    #     { -1: 0.1,  1: 0.1,  8: 0.2, 10: 0.2, 15: 0.4 },
    #     { -3: 0.2, -2: 0.0, -1: 0.8,  0: 0.0 }
    # ]




    # Measured temperatures (observation sequence): [ 2, -1, 8, 8 ] yields the following state log-likelihoods
    logLike = limLog([
      [ 0.1,  .0,  .0 ],
      [  .0, 0.1, 0.8 ],
      [ 0.1, 0.2,  .0 ],
      [ 0.1, 0.2,  .0 ]
    ])

    # Expected result: [0, 2, 1, 1], -9.985131541576637
    print( viterbi( logLike, logPi, logA ) )


    # Extending the observation sequence by one additional observation with the measured temperature 4
    # New observation sequence: [ 2, -1, 8, 8, 4 ]
    logLike = np.vstack( ( logLike, limLog([ 0.2, 0, 0 ]) ) )

    # Expected result: [0, 2, 0, 0, 0], -12.105395077776727
    print( viterbi( logLike, logPi, logA ) )
