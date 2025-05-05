# Hidden Markov Model (HMM) and Viterbi Algorithm Implementation

This project implements a Hidden Markov Model (HMM) with the Viterbi algorithm for Part-of-Speech (POS) tagging. The implementation uses log probabilities for numerical stability and includes smoothing to handle unseen words and transitions.

## Project Structure

```
HMM_and_Viterbi/
├── data/
│   ├── POS_train.pos    # Training data with POS tags
│   ├── POS_dev.pos      # Development data with POS tags
│   ├── POS_dev.words    # Development data without POS tags
│   ├── POS_test.words   # Test data without POS tags
│   └── README.txt       # Data format description
├── main.py             # Main implementation of HMM and Viterbi
├── scorer.py           # Evaluation script
└── my_test.pos         # Generated predictions for test data
```

## Features

- **HMM Training with Smoothing**: Implements a smoothed HMM training process using Laplace smoothing
- **Log Probability Space**: All calculations are performed in log space to prevent numerical underflow
- **Unknown Word Handling**: Special handling for unseen words using smoothing
- **Efficient Viterbi Implementation**: Optimized implementation of the Viterbi algorithm for POS tagging
- **Evaluation Script**: Included scorer for measuring tagging accuracy

## Implementation Details

### HMM Components

1. **Initial Probabilities (π)**: Probability of starting with a particular POS tag
2. **Transition Probabilities (A)**: Probability of transitioning between POS tags
3. **Emission Probabilities (B)**: Probability of observing a word given a POS tag

### Key Functions

- `train_hmm_smoothed()`: Trains the HMM model with smoothing
- `viterbi_log()`: Implements the Viterbi algorithm in log space
- `tag_file()`: Tags words in a file using the trained model
- `get_sentences()`: Utility function for reading sentence data

## Usage

1. Train the model on training data:
```python
train_sents = get_sentences(['data/POS_train.pos'])
pi, A, B, tagset, vocab, word_tag = train_hmm_smoothed(train_sents)
```

2. Tag new sentences:
```python
tag_file('input.words', 'output.pos', pi, A, B, tagset, vocab, word_tag)
```

3. Evaluate results:
```bash
python scorer.py gold_standard.pos predicted.pos
```

## Data Format

The input data files should be formatted as follows:
- Each line contains a word and its POS tag, separated by a tab
- Sentences are separated by empty lines
- Example:
```
The     DT
quick   JJ
brown   JJ
fox     NN
```

## Performance

The implementation includes:
- Laplace smoothing (α=0.1) for handling unseen words and transitions
- Log-space calculations for numerical stability
- Efficient handling of unknown words
- Support for large vocabulary sizes

## Requirements

- Python 3.x
- No external dependencies required

## Author

This implementation was created as part of an NLP assignment, demonstrating practical application of HMMs and the Viterbi algorithm for POS tagging. 