import numpy as np
import recognizer.tools as tools

# Default HMM
WORDS = {
    'name': ['sil', 'oh', 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'],
    'size': [1, 3, 15, 12, 6, 9, 9, 9, 12, 15, 6, 9],
    'gram': [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
}


class HMM:  

    words = {}

    def __init__(self, words=WORDS):
        """
        Constructor of HMM class. Inits with provided structure words
        :param input: word of the defined HMM.
        """
        self.words = words

        num_words = len(WORDS['name'])
        sizes = WORDS['size']
        N = np.sum(sizes)                           # total number of states
        start_states = np.cumsum([0] + sizes[:-1])  # indices of start states of each word

        # Initialize logPi
        # Start in the first state of any word with uniform probability
        # Use -np.inf instead of 0 since log(0) is not defined
        self.logPi = -np.inf * np.ones(N)
        for s in start_states:
            self.logPi[s] = np.log(1 / num_words)

        # Initialize logA
        self.logA = -np.inf * np.ones((N, N))
    
        # Linear HMM topology (Each state has only a self-loop and a transition to the next state within the word model)
        state_offset = 0
        for word_size in sizes:
            for i in range(word_size):
                s = state_offset + i

                # self-loop
                self.logA[s, s] = np.log(0.5)

                # next-state (if not last state)
                if i < word_size - 1:
                    self.logA[s, s + 1] = np.log(0.5)

            state_offset += word_size

        # From the last state of each word, allow transition to the first state of any word
        state_offset = 0
        for word_size in sizes:
            last_state = state_offset + word_size - 1

            for start_state in start_states:
                if last_state == start_state:
                    # special case in which a state is both the last and start state
                    # => probability of repetition 0.5 + (0.5 / num_words) probability that it comes from the beginning
                    self.logA[last_state, start_state] = np.log(0.5 + (0.5 / num_words))
                else:
                    self.logA[last_state, start_state] = np.log(0.5 / num_words)

            state_offset += word_size


    def get_num_states(self):
        """
        Returns the total number of states of the defined HMM.
        :return: number of states.
        """
        return sum(self.words['size'])

    def input_to_state(self, input):
        """
        Returns the state sequenze for a word.
        :param input: word of the defined HMM.
        :return: states of the word as a sequence.
        """
        if input not in self.words['name']:
            raise Exception('Undefined word/phone: {}'.format(input))

        # start index of each word
        start_idx = np.insert(np.cumsum(self.words['size']), 0, 0)

        # returns index for input's last state
        idx = self.words['name'].index(input) + 1

        start_state = start_idx[idx - 1]
        end_state = start_idx[idx]

        return [n for n in range(start_state, end_state) ]
    

    # Convert a given state sequence into the corresponding word sequence (e.g., 'oh two oh oh')
    def getTranscription(self, stateSequence):
        names = self.words['name']
        sizes = self.words['size']
        
        transcription = []

        # Remove 0's in state sequence ('sil' (silence))
        stateSequence = np.array(stateSequence)
        mask = stateSequence != 0
        stateSequence = stateSequence[mask]
        
        # Auxiliary variables
        prev_word_index = -1
        prev_state = -1

        for state in stateSequence:
            word_index = -1
            
            # Look for the word this state belongs to
            range = 0
            for i, size in enumerate(sizes):
                if range <= state < range + size:
                    word_index = i
                    break
                range += size
            
            #Now that we have the word, we need to check if it is a new word in the `stateSequence`.
            # If the `word_index` has changed, then we have a new one.
            if word_index != prev_word_index:
                transcription.append(names[word_index])
            else:
                # If the state is at the same word_index as before, the word might be repeating --> state < prev_state
                if state < prev_state:
                    transcription.append(names[word_index])

            prev_word_index = word_index
            prev_state = state

        return transcription
    
    # Generate a transcription from a matrix of posteriors
    def posteriors_to_transcription(self, posteriors):

        # Remove batch dimension
        if posteriors.ndim == 3:
            posteriors = posteriors.squeeze(0)

        # Compute most likely HMM state sequence using Viterbi
        logLike = tools.limLog(posteriors)
        best_path, pStar = tools.viterbi(logLike, self.logPi, self.logA)

        # Convert state sequence into transcription
        transcription = self.getTranscription(best_path)
        return transcription
