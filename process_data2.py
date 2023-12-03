import re
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
#nltk.download('punkt')
from datasets import load_dataset
#from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import Pool
import numpy as np

# CNN/Daily Mail dataset
dataset = load_dataset('cnn_dailymail', '3.0.0')

# Step 1: Find the top words in the dataset
def get_top_words(dataset, top_n=int(10000)):
    dataset = re.sub(r'\d', '', dataset)
    words = re.findall(r'\w+', dataset.lower())
    words = set(words)
    words = list(words)
    print(len(words))
    '''
    word_freq = Counter(tqdm(words, desc="Counting word frequencies"))
    top_words = [word for word, _ in word_freq.most_common(top_n)]
    # top_words += ['.',',','!', '?','+', '-', '(', ')', '$', '%', '&', ':', ';', '=', '@', '^', '_', '~']
    '''
    return words

# Step 2: Extract sentences
'''
def extract_sentences(dataset, num_processes=16):
    with Pool(processes=num_processes) as pool:
        sentences = pool.map(sent_tokenize, [dataset])
    return sentences[0]
'''
def extract_sentences(dataset):
    return sent_tokenize(dataset)

# Step 3: Replace words other than vocab with <UNK>
'''
def replace_with_unk(text, vocab, num_processes=16):
    with Pool(processes=num_processes) as pool:
        words = text.lower().split()
        results = pool.map(word if word in vocab else '<UNK>', words)
    return ' '.join(results)
'''
'''
def replace_with_unk(text, vocab, num_processes=16):
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        words = text.lower().split()
        results = list(executor.map(lambda word: word if word in vocab else '<UNK>', words))
    return ' '.join(results)
'''
'''
def replace_with_unk(text, vocab):
    return ''.join(word if word in vocab else '<UNK>' for word in text.lower().split())
'''
'''
def replace_with_unk(text, vocab):
    processed_text = []
    for word in tqdm(text.lower().split(), desc="Replacing words with <UNK>"):
        processed_text.append(word if word in vocab else '<UNK>')
    return ''.join(processed_text)
'''
'''
def replace_with_unk(text, vocab):
    # processed_text = []
    words = np.array(text.lower().split())
    mask = np.isin(words, vocab, invert=True)
    words[mask] = '<UNK>'
    # for word in tqdm(text.lower().split(), desc="Replacing words with <UNK>"):
    #     processed_text.append(word if word in vocab else '<UNK>')
    return ''.join(words)
'''
def replace_with_unk(text, vocab):
    return ' '.join(word if word in vocab else '<UNK>' for word in text.lower().split())

# Step 4: Append sentences with <s> and </s>
def append_tags(sentences, vocab):
    processed_sentences = []
    for sentence in tqdm(sentences):
        # processed_sentence = ''
        # symbols = r'[.,!?+\-()$%&:;=@^_~]'
        # modified_sentence = re.sub(f'({symbols})', r' \1 ', sentence)
        processed_sentence = ' '.join(word if word in vocab else '<UNK>' for word in sentence.lower().split())
        processed_sentences.append('<s> ' + processed_sentence + ' </s>')
    return processed_sentences

def remove_from_list(lst, denominator):
    # Calculate the positions
    begin = len(lst) // denominator
    #end = 2 * len(lst) // denominator

    # Remove elements from first_third to second_third
    #del lst[:begin]
    return lst[:begin]

# Get Training Data & Convert to String
articles = remove_from_list(dataset['train']['article'], 3)
print(len(articles))
joined_articles = ''.join(articles)
#print(joined_articles)

# Step 1: Create vocabulary
print('********Step 1: Creating vocabulary*********')
top_words = get_top_words(joined_articles)
print('Step 1 Complete')

# Save vocab to a .voc file
data_fpath = 'dailymail_cnn_all'
with open(f'data/{data_fpath}.voc', 'wb') as vocab_file:
    # json.dump('\n'.join(top_words), vocab_file)
    vocab_file.write('\n'.join(top_words).encode('utf-8'))

# Step 2: Extract sentences from training data
print('********Step 2: Extracting Sentences*********')
sentences = extract_sentences(joined_articles)
print('Step 2 Complete')

# Step 3: Replace unknown words with <UNK>
print('********Step 3: Replace with UNK*********')
vocab_dict = {}
for top_word in top_words:
    vocab_dict[top_word] = True
processed_sentences = replace_with_unk(joined_articles, vocab_dict)
print('Step 3 Complete')

# Step 4: Tag Sentences
print('********Step 4: Tag Sentences*********')
vocab_dict = {}
for top_word in top_words:
    vocab_dict[top_word] = True
tagged_sentences = append_tags(sentences, vocab_dict)
print('Step 4 Complete')

# Save processed sentences to a .list file
with open(f'data/{data_fpath}.list', 'wb') as sentences_file:
    sentences_file.write('\n'.join(tagged_sentences).encode('utf-8'))
