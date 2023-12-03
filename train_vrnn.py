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
import models.VRNN as VRNN
warnings.filterwarnings('ignore')

#import multiprocessing as mp
#mp.Queue(1000)

os.environ['COMMANDLINE_ARGS'] = '--no-half'

def make_folder_for_file(fileName):
    folder = os.path.dirname(fileName)
    if folder != '' and not os.path.isdir(folder):
        os.makedirs(folder)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a simple auto-regressive recurrent LM')
    parser.add_argument('--input_file', metavar='FILE', default='data/dailymail_cnn_shrey.list', help='Full path to a file containing normalised sentences')
    parser.add_argument('--output_file', metavar='FILE', default='output/dailymail_cnn_shrey.pytorch', help='Full path to the output model file as a torch serialised object')
    parser.add_argument('--vocabulary',metavar='FILE',default='data/dailymail_cnn_shrey.voc',help='Full path to a file containing the vocabulary words')
    parser.add_argument('--n_epochs',type=int,default=25,help='Number of epochs to train')
    parser.add_argument('--batch_size',type=int,default=8,help='Batch size')
    parser.add_argument('--embedding_size',type=int,default=128,help='Size of the embedding layer')
    parser.add_argument('--h_dim',type=int,default=100,help='Size of the hidden recurrent layers')
    parser.add_argument('--n_layers',type=int,default=1,help='Number of recurrent layers')
    parser.add_argument('--learning_rate',type=float,default=1e-3,help='Learning rate')
    parser.add_argument('--seed',type=float,default=128,help='Random seed')
    parser.add_argument('--max_length',type=int,default=20,help='Maximum length of sequences to use (longer sequences are discarded)')
    parser.add_argument('--start_token',type=str,default='<s>',help='Word token used at the beginning of a sentence')
    parser.add_argument('--end_token',type=str,default='<\s>',help='Word token used at the end of a sentence')
    parser.add_argument('--unk_token',type=str,default='<UNK>',help='Word token used for out-of-vocabulary words')
    parser.add_argument('--verbose',default=2,type=int,choices=[0,1,2],help='Verbosity level (0, 1 or 2)')
    parser.add_argument('--z_dim',default=16,type=int,help='Dimensionality of the latent variable')
    parser.add_argument('--clip',default=10,type=int,help='Gradient clipping')
    parser.add_argument('--save_every',default=10,type=int,help='Save model every n epochs')
    parser.add_argument('--print_every',default=1000,type=int,help='Print every n batches')
    
    args = parser.parse_args()
    args = vars(args)
    return args

def train(epoch, trainset, args):
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):

        #transforming data
        data = data.to(args['device'])
        data = data.squeeze().transpose(0, 1) # (seq, batch, elem)
        data = (data - data.min()) / (data.max() - data.min())
        
        #forward + backward + optimize
        optimizer.zero_grad()
        kld_loss, nll_loss, _, _ = model(data)
        loss = kld_loss + nll_loss
        loss.backward()
        optimizer.step()

        #grad norm clipping, only in pytorch version >= 1.10
        nn.utils.clip_grad_norm_(model.parameters(), args['clip'])

        #printing
        if batch_idx % args['print_every'] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t KLD Loss: {:.6f} \t NLL Loss: {:.6f}'.format(
                epoch, batch_idx * args['batch_size'], args['batch_size'] * (len(train_loader.dataset)//args['batch_size']),
                100. * batch_idx / len(train_loader),
                kld_loss / args['batch_size'],
                nll_loss / args['batch_size']))
            
            sample = model.sample(torch.tensor(28, device=args['device']))

        train_loss += loss.item()

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))

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
    trainset,validset = load_data(cv=True, **args)
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
        train(epoch, trainset, args)
        test(epoch)

        #saving model
        if epoch % args['save_every'] == 1:
            fn = 'saves/vrnn_state_dict_'+str(epoch)+'.pth'
            torch.save(model.state_dict(), fn)
            print('Saved model to '+fn)

