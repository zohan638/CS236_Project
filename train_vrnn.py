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
from models.VRNN import VRNN
import numpy as np
import json
warnings.filterwarnings('ignore')

os.environ['COMMANDLINE_ARGS'] = '--no-half'

def make_folder_for_file(fileName):
    folder = os.path.dirname(fileName)
    if folder != '' and not os.path.isdir(folder):
        os.makedirs(folder)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a simple auto-regressive recurrent LM')
    parser.add_argument('--input_file', metavar='FILE', default='data/dailymail_cnn.list', help='Full path to a file containing normalised sentences')
    parser.add_argument('--output_file', metavar='FILE', default='dailymail_cnn(1)', help='Full path to the output model file as a torch serialised object')
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

from torch.utils.data import Dataset, DataLoader

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

def train_model(model, data_loader, device, learning_rate, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total_kld_loss = 0
        total_nll_loss = 0
        for batch in tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch = batch.to(device)
            batch = batch.squeeze().transpose(0, 1)
            optimizer.zero_grad()
            lengths = calculate_lengths(batch, 1)
            kld_loss, nll_loss, _, _ = model(batch, lengths=lengths) 
            loss = 10 * kld_loss + nll_loss
            loss.backward()
            optimizer.step()

            nn.utils.clip_grad_norm_(model.parameters(), args['clip'])
            # print(f"KLD Loss: {kld_loss.item()}, NLL Loss: {nll_loss.item()}")
            total_kld_loss += kld_loss.item()
            total_nll_loss += nll_loss.item()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, KLD Loss: {round(total_kld_loss / len(data_loader), 2)}, Recon Loss: {round(total_nll_loss / len(data_loader))}, Loss: {round(total_loss / len(data_loader))}")

        # save model
        save_every = 1
        if epoch % save_every == 0:
            fn = f'output/intermediate/vrnn_{args["output_file"]}_{epoch}.pth'
            torch.save(model.state_dict(), fn)
            print('Saved model to '+fn)

        for i in range(2):
            random_integer = random.randint(40, 60)
            sample_embedding, sample_string = model.sample(random_integer)
            print(f"Sample {i+1}: {sample_string}")

        print('\n')

    # Save final model
    fn = f'output/vrnn_{args["output_file"]}_final.pth'
    torch.save(model.state_dict(), fn)

if __name__ == '__main__':
    args = parse_arguments()

    #Initialisations
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed(args['seed'])
    torch.backends.cudnn.deterministic = True
    args['device'] = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))
    if os.path.exists('output/intermediate') is False:
        os.makedirs('output/intermediate')
    # Load padded sentences
    padded_sentences = np.load('data/vrnn_padded_sentences.npy')

    # Convert numpy array to PyTorch tensor
    padded_sentences_tensor = torch.from_numpy(padded_sentences).long()
    
    # Load vocabulary
    with open('data/vrnn_vocabulary.json', 'r') as f:
        vocab = json.load(f)

    # Create data loader
    dataset = TextDataset(padded_sentences_tensor)
    data_loader = DataLoader(dataset, batch_size=500, shuffle=True)

    preset_args = {
        'vocab': vocab,
        'embedding_size': 128,
        'h_dim': 256,
        'z_dim': 32,
        'n_layers': 2,
        'bias': False,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }

    # Create model
    model = VRNN(preset_args)
    model.to(args['device'])
    train_model(model, data_loader, args['device'], args['learning_rate'], epochs=100)  # Adjust epochs as needed


'''
def train(epoch, trainset_batches, args):
    train_loss = 0
    for batch_idx, data in enumerate(trainset_batches):

        #transforming data
        data = data.to(args['device'])
        # data = data.squeeze().transpose(0, 1) # (seq, batch, elem)
        # data = (data - data.min()) / (data.max() - data.min())
        # data = data.permute(1, 0)
        # print(data.shape)
        # exit()
        
        #forward + backward + optimize
        optimizer.zero_grad()
        lengths = [data.shape[0] for _ in range(data.shape[1])]
        kld_loss, nll_loss, _, _ = model(data, lengths)
        loss = kld_loss + nll_loss
        loss.backward()
        optimizer.step()

        #grad norm clipping, only in pytorch version >= 1.10
        nn.utils.clip_grad_norm_(model.parameters(), args['clip'])

        #printing
        if batch_idx % args['print_every'] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t KLD Loss: {:.6f} \t NLL Loss: {:.6f}'.format(
                epoch, batch_idx * args['batch_size'], args['batch_size'] * (len(trainset_batches)//args['batch_size']),
                100. / len(trainset_batches),
                kld_loss / args['batch_size'],
                nll_loss / args['batch_size']))
            
            # sample = model.sample(torch.tensor(28, device=args['device']))

        train_loss += loss.item()

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(trainset_batches)))

if __name__ == '__main__':
    args = parse_arguments()

    #Initialisations
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed(args['seed'])
    torch.backends.cudnn.deterministic = True
    args['device'] = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))

    #Read vocabulary, sentences and load data
    args['vocab'],args['characters'] = read_vocabulary(**args)
    args['num_seq'],args['max_words'] = count_sequences(**args)

    trainset_exists = os.path.exists('data/trainset_dailymail_cnn.pth')
    validset_exists = os.path.exists('data/validset_dailymail_cnn.pth')

    if trainset_exists and validset_exists:
        trainset = torch.load('data/trainset_dailymail_cnn.pth')
        validset = torch.load('data/validset_dailymail_cnn.pth')
    else:
        trainset,validset = load_data(cv=True, **args)

    num_batches = trainset.shape[1] // args['batch_size']
    trainset_batches = torch.chunk(trainset, num_batches, dim=1)
    # trainset_batches = [torch.tensor(batch, dtype=torch.float) for batch in trainset_batches]
    # del trainset

    # print('--------------------------------------------')
    # print(trainset.shape)
    # print(validset.shape)
    # print(trainset_batches[0].shape)
    # print(trainset_batches[10].shape)
    # print('--------------------------------------------')
    # torch.save(trainset, 'trainset_dailymail_cnn.pth')
    # torch.save(validset, 'validset_dailymail_cnn.pth')

    if args['verbose'] >= 1:
        print('Number of training sequences: {0:d}'.format(trainset.shape[1]))
        print('Number of cross-validaton sequences: {0:d}'.format(validset.shape[1]))

    #manual seed
    torch.manual_seed(args['seed'])

    #init model + optimizer + datasets
    model = VRNN(args)
    model = model.to(args['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])

    for epoch in range(1, args['n_epochs'] + 1):
        #training + testing
        train(epoch, trainset_batches, args)
        # test(epoch)

        #saving model
        if epoch % args['save_every'] == 1:
            fn = 'saves/vrnn_state_dict_'+str(epoch)+'.pth'
            torch.save(model.state_dict(), fn)
            print('Saved model to '+fn)
'''