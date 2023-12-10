import sys
import os
import random
from tqdm import tqdm
import argparse
import gc
from utils.rnnlm_func import load_data
from utils.lm_func import read_vocabulary, count_sequences
import torch
import torch.nn as nn
import warnings
import numpy as np
import json
from models.BIVRNN import BidirectionalVRNN
from evaluate_roberta import get_roberta_score
from torch.utils.data import Dataset, DataLoader
warnings.filterwarnings('ignore')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a simple auto-regressive recurrent LM')
    parser.add_argument('--rnn_type', type=str, default='GRU', help='Type of RNN to use (LSTM or GRU)')
    parser.add_argument('--input_file', metavar='FILE', default='data/dailymail_cnn.list', help='Full path to a file containing normalised sentences')
    parser.add_argument('--output_file', metavar='FILE', default='dailymail_cnn(0)', help='Full path to the output model file as a torch serialised object')
    parser.add_argument('--revise_sequence', metavar='FILE', default='test_sequences/revise1.txt', help='Full path to the sequence to be revised')
    parser.add_argument('--vocabulary',metavar='FILE',default='data/dailymail_cnn.voc',help='Full path to a file containing the vocabulary words')
    parser.add_argument('--n_epochs',type=int,default=25,help='Number of epochs to train')
    parser.add_argument('--batch_size',type=int,default=8,help='Batch size')
    parser.add_argument('--embedding_size',type=int,default=128,help='Size of the embedding layer')
    parser.add_argument('--h_dim',type=int,default=256,help='Size of the hidden recurrent layers')
    parser.add_argument('--n_layers',type=int,default=1,help='Number of recurrent layers')
    parser.add_argument('--learning_rate',type=float,default=.001,help='Learning rate')
    parser.add_argument('--seed',type=float,default=128,help='Random seed')
    parser.add_argument('--max_length',type=int,default=20,help='Maximum length of sequences to use (longer sequences are discarded)')
    parser.add_argument('--start_token',type=str,default='<s>',help='Word token used at the beginning of a sentence')
    parser.add_argument('--end_token',type=str,default='<\s>',help='Word token used at the end of a sentence')
    parser.add_argument('--unk_token',type=str,default='<UNK>',help='Word token used for out-of-vocabulary words')
    parser.add_argument('--verbose',default=2,type=int,choices=[0,1,2],help='Verbosity level (0, 1 or 2)')
    parser.add_argument('--z_dim',default=20,type=int,help='Dimensionality of the latent variable')
    parser.add_argument('--clip',default=2.0,type=int,help='Gradient clipping')
    parser.add_argument('--save_every',default=10,type=int,help='Save model every n epochs')
    parser.add_argument('--print_every',default=1000,type=int,help='Print every n batches')
    parser.add_argument('--cv_percentage',default=0.1,help='Percentage Valid Set')
    
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

def train_bidirectional_vrnn(model, rnn_type, data_loader, device, learning_rate, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    total_loss = 0

    for epoch in range(epochs):
        for batch in tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch = batch.to(device)
            batch = batch.squeeze().transpose(0, 1)
            optimizer.zero_grad()

            # Forward pass
            lengths = calculate_lengths(batch, 1)
            kld_loss, recon_loss, _, _ = model(batch, lengths=lengths)

            # Combined loss
            loss = kld_loss + recon_loss

            # Backward pass
            loss.backward()
            optimizer.step()

            # Grad clipping
            nn.utils.clip_grad_norm_(model.parameters(), args['clip'])
            
            total_kld_loss += kld_loss.item()
            total_recon_loss += recon_loss.item()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, KLD Loss: {round(total_kld_loss / len(data_loader), 2)}, Recon Loss: {round(total_recon_loss / len(data_loader), 2)}, Loss: {round(total_loss / len(data_loader), 2)}")

        # save model
        save_every = 1
        if epoch % save_every == 0:
            fn = f'output/bidirectional/intermediate/vrnn_{args["output_file"]}_{epoch}.pth'
            torch.save(model.state_dict(), fn)
            print('Saved model to '+fn)

    
    # Save final model
    fn = f'output/bidirectional/vrnn_{args["output_file"]}_final.pth'
    torch.save(model.state_dict(), fn)

if __name__ == '__main__':
    args = parse_arguments()

    # Load test sequence
    with open(args['revise_sequence'], 'r') as file:
        test_sequences = file.readlines()

    #Initialisations
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed(args['seed'])
    torch.backends.cudnn.deterministic = True
    args['device'] = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))
    if os.path.exists('output/intermediate') is False:
        os.makedirs('output/intermediate')

    # Load padded sentences
    padded_sentences = np.load('data/vrnn_padded_paragraphs.npy')

    # Convert numpy array to PyTorch tensor
    padded_sentences_tensor = torch.from_numpy(padded_sentences).long()
    
    # Load vocabulary
    with open('data/vrnn_vocabulary_paragraphs.json', 'r') as f:
        vocab = json.load(f)

    # Create data loader
    dataset = TextDataset(padded_sentences_tensor)
    data_loader = DataLoader(dataset, batch_size=20, shuffle=True)

    preset_args = {
        'vocab': vocab,
        'rnn_type': args['rnn_type'],
        'embedding_size': 128,
        'h_dim': 256,
        'z_dim': 32,
        'n_layers': 2,
        'bias': False,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }

    # Create model
    model = BidirectionalVRNN(preset_args)
    model.to(args['device'])
    train_bidirectional_vrnn(model, args['model_type'], data_loader, args['device'], args['learning_rate'], epochs=20)