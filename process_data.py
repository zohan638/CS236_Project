import re
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize
import tqdm
#nltk.download('punkt')

# Step 1: Find the top words in the dataset
def get_top_words(dataset, top_n=int(1000)):
    dataset = re.sub(r'\d', '', dataset)
    words = re.findall(r'\w+', dataset.lower())
    word_freq = Counter(words)
    top_words = [word for word, _ in word_freq.most_common(top_n)]
    # top_words += ['.',',','!', '?','+', '-', '(', ')', '$', '%', '&', ':', ';', '=', '@', '^', '_', '~']
    return top_words

# # Step 2: Replace words other than vocab with <UNK>
def replace_with_unk(text, vocab):
    return ' '.join(word if word in vocab else '<UNK>' for word in text.lower().split())

# Step 3: Extract sentences
def extract_sentences(dataset):
    return sent_tokenize(dataset)

# Step 4: Append sentences with <s> and </s>
def append_tags(sentences, vocab):
    processed_sentences = []
    for sentence in tqdm.tqdm(sentences):
        # processed_sentence = ''
        # for word in sentence.lower().split():
        #     if vocab[word]:
        #         processed_sentence += ' '.join(word)
        #     else:
        #         processed_sentence += ' '.join('<UNK>')
        symbols = r'[.,!?+\-()$%&:;=@^_~]'
        modified_sentence = re.sub(f'({symbols})', r' \1 ', sentence)
        processed_sentence = ' '.join(word if word in vocab else '<UNK>' for word in modified_sentence.lower().split())
        processed_sentences.append('<s> ' + processed_sentence + ' </s>')
    return processed_sentences

# Example usage
with open('data/enwik8', 'r', encoding='utf-8') as file:
    dataset = file.read()

# Step 1
top_words = get_top_words(dataset)

# Step 2
# processed_dataset = replace_with_unk(dataset, top_words)

# Step 3
sentences = extract_sentences(dataset)

# processed_sentences = replace_with_unk(sentences, top_words)

# Step 4
vocab_dict = {}
for top_word in top_words:
    vocab_dict[top_word] = True
tagged_sentences = append_tags(sentences, vocab_dict)

# Save vocab to a .voc file
with open('data/text8_vocab.voc', 'wb') as vocab_file:
    # json.dump('\n'.join(top_words), vocab_file)
    vocab_file.write('\n'.join(top_words).encode('utf-8'))

# Save processed sentences to a .list file
with open('data/text8_sentences.list', 'wb') as sentences_file:
    sentences_file.write('\n'.join(tagged_sentences).encode('utf-8'))