import nltk
from nltk.tokenize import word_tokenize
import json
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from datasets import load_dataset
nltk.download('punkt')
import os

def tokenize_sentences(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    truncated_lines = lines[:int(0.0025 * len(lines))]
    return [word_tokenize(line.lower()) for line in tqdm(truncated_lines, desc="Tokenizing")]

def build_vocab(tokenized_sentences, min_word_freq=1):
    word_freq = {}
    for sentence in tqdm(tokenized_sentences):
        for word in sentence:
            if word not in word_freq:
                word_freq[word] = 1
            else:
                word_freq[word] += 1

    vocab = {word: index + 2 for index, (word, freq) in enumerate(word_freq.items()) if freq >= min_word_freq}
    vocab['<unk>'] = 0  # Unknown words
    vocab['<pad>'] = 1  # Padding token

    return vocab

def vectorize_sentences(tokenized_sentences, vocab):
    vectorized_sentences = []
    for sentence in tqdm(tokenized_sentences, desc="Vectorizing"):
        vectorized_sentence = [vocab.get(word, vocab['<unk>']) for word in sentence]
        print(vectorized_sentence)
        vectorized_sentences.append(vectorized_sentence)
    return vectorized_sentences

import numpy as np

def pad_sentences(vectorized_sentences, desired_length):
    padded_sentences = np.zeros((len(vectorized_sentences), desired_length), dtype=int)
    for i, sentence in tqdm(enumerate(vectorized_sentences), desc="Padding"):
        length = min(desired_length, len(sentence))
        padded_sentences[i, :length] = sentence[:length]
    return padded_sentences

def extract_sentences(dataset):
    return sent_tokenize(dataset)

def remove_from_list(lst, denominator):
    # Calculate the positions
    begin = len(lst) // denominator
    #end = 2 * len(lst) // denominator

    # Remove elements from first_third to second_third
    #del lst[:begin]
    return lst[:begin]

file_path = 'data/vrnn_dailymail_cnn.list'  # Path to your file
if not os.path.exists(file_path):
    # CNN/Daily Mail dataset
    dataset = load_dataset('cnn_dailymail', '3.0.0')

    # Load dataset
    articles = remove_from_list(dataset['train']['article'], 3)
    joined_articles = ''.join(articles)

    # Get sentences
    sentences = extract_sentences(joined_articles)

    # save sentences to a .list file
    with open(f'data/vrnn_dailymail_cnn.list', 'wb') as sentences_file:
        sentences_file.write('\n'.join(sentences).encode('utf-8'))

# Tokenize sentences
tokenized_sentences = tokenize_sentences(file_path)

# Build vocabulary
vocab = build_vocab(tokenized_sentences)

# Vectorize sentences
vectorized_sentences = vectorize_sentences(tokenized_sentences, vocab)

# Pad sentences
desired_length = 100  # Change this to your desired length
padded_sentences = pad_sentences(vectorized_sentences, desired_length)
np.save('data/vrnn_padded_sentences.npy', padded_sentences)
with open('data/vrnn_vocabulary.json', 'w') as vocab_file:
    json.dump(vocab, vocab_file)