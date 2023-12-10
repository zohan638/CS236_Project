import os
import re
import argparse
import json
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a simple auto-regressive recurrent LM')
    parser.add_argument('--sentence_or_article', type=str, default='sentence', help='Whether to use sentences or articles')
    parser.add_argument('--percent_dataset', type=float, default=0.0005, help='Percent of dataset to use')
    parser.add_argument('--redo', type=bool, default=False, help='Whether articles/sentences list file should be reconstructed from dataset')
    
    args = parser.parse_args()
    args = vars(args)
    return args

def reduce_dataset(file_path, percent_dataset):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    return lines[:int(percent_dataset * len(lines))]

def tokenize_sequences(sequences):    
    return [word_tokenize(line.lower()) for line in tqdm(sequences, desc="Tokenizing")]

def build_vocab(tokenized_sequences, min_word_freq=3):
    # Count word frequencies
    word_freq = {}
    for sequence in tqdm(tokenized_sequences, desc="Counting word frequencies"):
        for word in sequence:
            word_freq[word] = word_freq.get(word, 0) + 1

    # Initialize vocabulary with special tokens
    vocab = {'<unk>': 0, '<pad>': 1}

    # Assign indices to words above the frequency threshold
    index = len(vocab)  # Start indexing from the next available index
    for word, freq in tqdm(word_freq.items(), desc="Building vocab"):
        if freq >= min_word_freq:
            vocab[word] = index
            index += 1

    return vocab

def vectorize_sequences(tokenized_sequences, vocab):
    vectorized_sequences = []
    for sequence in tqdm(tokenized_sequences, desc="Vectorizing"):
        vectorized_sequence = [vocab.get(word, vocab['<unk>']) for word in sequence]
        vectorized_sequences.append(vectorized_sequence)
    return vectorized_sequences

def pad_sequences(vectorized_sequences, desired_length):
    padded_sequences = np.zeros((len(vectorized_sequences), desired_length), dtype=int)
    for i, sequence in tqdm(enumerate(vectorized_sequences), desc="Padding"):
        length = min(desired_length, len(sequence))
        padded_sequences[i, :length] = sequence[:length]
    return padded_sequences

def extract_sentences(dataset):
    return sent_tokenize(dataset)

def build_datset(sentence_or_article):
    # Load dataset
    dataset = load_dataset('cnn_dailymail', '3.0.0')
    articles = dataset['train']['article']
    print('Loaded dataset')

    # Remove punctuation except periods
    no_punct_articles = []
    for article in tqdm(articles, desc='Removing punctuation'):
        no_punct_articles.append(re.sub(r'[^\w\s.]', '', article))

    # Save articles/sentences
    if(sentence_or_article == 'sentence'):
        joined_articles = ''.join(no_punct_articles)
        sentences = extract_sentences(joined_articles)
        with open(f'data/vrnn_dailymail_cnn.list', 'wb') as sentences_file:
            sentences_file.write('\n'.join(sentences).encode('utf-8'))
    else:
        with open(f'data/vrnn_dailymail_cnn_articles.list', 'w') as sentences_file:
            for article in tqdm(no_punct_articles, desc='Writing articles to file'):
                sentences_file.write(article + '\n')

# Parse arguments
args = parse_arguments()
sentence_or_article = args['sentence_or_article']
percent_dataset = args['percent_dataset']
redo = args['redo']

# File path
file_path = 'data/vrnn_dailymail_cnn.list'
if sentence_or_article == 'article':
    file_path = 'data/vrnn_dailymail_cnn_articles.list'

# Build dataset file if needed
if redo == True or not os.path.exists(file_path):
    build_datset(sentence_or_article)

# Extract specified percent of dataset from list file
reduced_dataset = reduce_dataset(file_path, percent_dataset)

# Tokenize sequences
tokenized_sequences = tokenize_sequences(reduced_dataset)

# Build vocabulary
vocab = build_vocab(tokenized_sequences)

# Vectorize articles/sentences
vectorized_sequences = vectorize_sequences(tokenized_sequences, vocab)

# Find longest sequence
max_sequence_length = max([len(sentence) for sentence in tokenized_sequences])

# Pad sentences
padded_sequences = pad_sequences(vectorized_sequences, max_sequence_length)

# Save padded sentences and vocabulary
if(sentence_or_article == 'sentence'):
    np.save('data/vrnn_padded_sentences.npy', padded_sequences)
    with open('data/vrnn_vocabulary_sentences.json', 'w') as vocab_file:
        json.dump(vocab, vocab_file)
else:
    np.save('data/vrnn_padded_articles.npy', padded_sequences)
    with open('data/vrnn_vocabulary_articles.json', 'w') as vocab_file:
        json.dump(vocab, vocab_file)