import os
import argparse
from tqdm import tqdm
import json
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models.BI_RNN import BIDIRECTIONAL_RNN
from process_cnn_dailymail import tokenize_sequences, vectorize_sequences
from evaluate_roberta import get_roberta_score
import warnings
warnings.filterwarnings('ignore')

def make_folder_for_file(fileName):
    folder = os.path.dirname(fileName)
    if folder != '' and not os.path.isdir(folder):
        os.makedirs(folder)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a bidirectional variational RNN')

    # Model hyperparameters
    parser.add_argument('--rnn_type', type=str, default='GRU', help='Type of RNN to use (LSTM or GRU)')
    parser.add_argument('--embed_dim', type=int, default=128, help='Size of the embedding layer')
    parser.add_argument('--h_dim', type=int, default=256, help='Size of the hidden recurrent layers')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of recurrent layers')
    parser.add_argument('--learning_rate', type=float, default=.001, help='Learning rate')

    # Training options
    parser.add_argument('--sentence_or_article', type=str, default='article', help='Whether to use sentences or articles')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size')
    parser.add_argument('--seed', type=float, default=128, help='Random seed')
    parser.add_argument('--clip', default=2.0, type=int, help='Gradient clipping')

    # Sample options
    parser.add_argument('--start_sequence', default='Today the president of the United States announced', type=str, help='Starting sequence for sampling')
    parser.add_argument('--num_samples', default=1, type=int, help='Number of samples to generate after every epoch')
    parser.add_argument('--sample_length', default=100, type=int, help='Length of samples')

    # Save and plot options
    parser.add_argument('--save_samples', default=False, type=bool, help='Whether to save samples')
    parser.add_argument('--save_every', default=10, type=int, help='Save model every n epochs')
    parser.add_argument('--print_every', default=1, type=int, help='Print every n batches')
    
    args = parser.parse_args()
    args = vars(args)
    return args

class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
def calculate_lengths(batch, padding_idx):
    lengths = (batch != padding_idx).sum(dim=1)
    return lengths

def get_file_index():
    # Get integer for output file name
    output_dir = 'output/birnn/intermediate/'
    largest_integer = 0
    if os.path.exists(output_dir):
        # Get a list of files in the directory
        files = os.listdir(output_dir)
        
        for file in files:
            if file.startswith(tuple(map(str, range(10)))):
                # Extract the integer from the file name
                file_integer = int(file.split('_')[0])
                
                # Update the largest integer if necessary
                if file_integer > largest_integer:
                    largest_integer = file_integer
        
        # Increment the largest integer by 1
        largest_integer += 1

    return largest_integer

def generate_samples(model, rnn_type, num_samples, sample_length, start_sequence):
    # Tokenize and vectorize the start sequence
    sequence = start_sequence
    sequence = tokenize_sequences([sequence])
    sequence = vectorize_sequences(sequence, model.vocab)
    sequence = torch.tensor(sequence).long().to(model.device)

    samples = []
    for i in range(num_samples):
        sample = model.sample(rnn_type, sample_length, sequence)
        samples.append([sample])
        print('Sample {}: {}\n'.format(i+1, sample))
    return samples

def train_model(model, training_options, sample_options, data_loader, device):
    rnn_type = training_options['rnn_type']
    epochs = training_options['epochs']
    optimizer = torch.optim.Adam(model.parameters(), lr=training_options['learning_rate'])

    model.train()
    out_dict = {}
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch = batch.to(device)
            batch = batch.squeeze().transpose(0, 1)
            optimizer.zero_grad()
            lengths = calculate_lengths(batch, 1)
            
            _, loss = model(rnn_type, batch, lengths=lengths) 
            loss.backward()
            optimizer.step()

            nn.utils.clip_grad_norm_(model.parameters(), args['clip'])
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {round(total_loss / len(data_loader), 2)}")

        # save model
        save_every = 10
        if epoch % save_every == 0:
            fn = f"output/birnn/intermediate/{sample_options['file_int']}_birnn_{rnn_type}_{sample_options['sentence_or_article']}_{epoch}.pth"
            torch.save(model, fn)

        # Generate samples
        samples = generate_samples(model, rnn_type, sample_options['num_samples'], sample_options['sample_length'], sample_options['start_sequence'])
    
    # Save output dictionary
    if sample_options['save_samples'] == True:
        with open(f"plots/output_dicts/birnn/{sample_options['file_int']}_birnn_{rnn_type}_{sample_options['sentence_or_article']}.pkl", 'wb') as file:
            pickle.dump(out_dict, file)

    # Save final model
    fn = f"output/birnn/{sample_options['file_int']}_birnn_{rnn_type}_{sample_options['sentence_or_article']}_final.pth"
    torch.save(model, fn)

if __name__ == '__main__':
    args = parse_arguments()

    # Initialisations
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed(args['seed'])
    torch.backends.cudnn.deterministic = True
    args['device'] = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))

    # Load files for sentences or articles
    if args['sentence_or_article'] == 'sentence':
        padded_sequences = np.load('data/vrnn_padded_sentences.npy')
        print("Using Sentences - Padded sequences shape: ", padded_sequences.shape)
        with open('data/vrnn_vocabulary_sentences.json', 'r') as f:
            vocab = json.load(f)
    elif args['sentence_or_article'] == 'article':
        padded_sequences = np.load('data/vrnn_padded_articles.npy')
        print("Using Articles - Padded sequences shape: ", padded_sequences.shape)
        with open('data/vrnn_vocabulary_articles.json', 'r') as f:
            vocab = json.load(f)  

    # Convert numpy array to PyTorch tensor
    padded_sequences_tensor = torch.from_numpy(padded_sequences).long()

    # Create data loader
    dataset = TextDataset(padded_sequences_tensor)
    data_loader = DataLoader(dataset, batch_size=args['batch_size'], shuffle=True)

    # Model parameters
    model_parameters = {
        'vocab': vocab,
        'rnn_type': args['rnn_type'],
        'embed_dim': args['embed_dim'],
        'h_dim': args['h_dim'],
        'n_layers': args['n_layers'],
        'bias': False,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }
    print("Model parameters: ", {'rnn_type': args['rnn_type'], 'embed_dim': args['embed_dim'], 'h_dim': args['h_dim'], 'n_layers': args['n_layers'], 'bias': False})

    # Training Options
    training_options = {
        "rnn_type": args['rnn_type'],
        "epochs": args['epochs'],
        "learning_rate": args['learning_rate'],
    }
    print("Training options: ", {'epochs': training_options['epochs'], 'batch size': args['batch_size'], 'learning rate': training_options['learning_rate']})
    print('\n')

    # Sample options
    sample_options = {
        'num_samples': args['num_samples'],
        'sample_length': args['sample_length'],
        'save_samples': args['save_samples'],
        'start_sequence': args['start_sequence'],
        'sentence_or_article': args['sentence_or_article'],
        'file_int': get_file_index()
    }

    # Train model
    model = BIDIRECTIONAL_RNN(model_parameters)
    model.to(args['device'])
    train_model(model, training_options, sample_options, data_loader, args['device'])