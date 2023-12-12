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
from models.VRNN import VRNN
from evaluate_roberta import get_roberta_score
import warnings
warnings.filterwarnings('ignore')

os.environ['COMMANDLINE_ARGS'] = '--no-half'

def make_folder_for_file(fileName):
    folder = os.path.dirname(fileName)
    if folder != '' and not os.path.isdir(folder):
        os.makedirs(folder)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a bidirectional variational RNN')

    # Model hyperparameters
    parser.add_argument('--rnn_type', type=str, default='GRU', help='Type of RNN to use (LSTM or GRU)')
    parser.add_argument('--embed_dim', type=int, default=128, help='Size of the embedding layer')
    parser.add_argument('--z_dim', default=100, type=int, help='Dimensionality of the latent variable')
    parser.add_argument('--h_dim', type=int, default=256, help='Size of the hidden recurrent layers')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of recurrent layers')
    parser.add_argument('--learning_rate', type=float, default=.001, help='Learning rate')

    # Training options
    parser.add_argument('--sentence_or_article', type=str, default='sentence', help='Whether to use sentences or articles')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=200, help='Batch size')
    parser.add_argument('--seed', type=float, default=128, help='Random seed')
    parser.add_argument('--kl_annealing', type=str, default='linear', help='Type of KL Annealing to Use')
    parser.add_argument('--kl_annealing_start', type=float, default=0.0, help='The starting value for the KL weight')
    parser.add_argument('--kl_annealing_growth_rate', type=float, default=0.01, help='KL Annealing growth rate for linear/exponential annealing')
    parser.add_argument('--kl_annealing_epoch', type=int, default=25, help='The end/middle epoch for KL weight, depending on type of annealing')
    parser.add_argument('--clip', default=2.0, type=int, help='Gradient clipping')

    # Sample options
    parser.add_argument('--num_samples', default=50, type=int, help='Number of samples to generate after every epoch')
    parser.add_argument('--sample_length', default=100, type=int, help='Length of samples')

    # Save and plot options
    parser.add_argument('--save_samples', default=True, type=bool, help='Whether to save samples')
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
    output_dir = 'output/vrnn/intermediate/'
    largest_integer = 0
    if os.path.exists(output_dir):
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

def generate_samples(model, rnn_type, num_samples, sample_length):
    samples = []
    for i in range(num_samples):
        sample = model.sample(rnn_type, sample_length)
        samples.append([sample])
        # roberta_score = get_roberta_score(sample)[0][0]
        # print('Sample {} - Roberta Score {}:\n {}\n'.format(i+1, roberta_score, sample))
        # print('Sample {}: {}\n'.format(i+1, sample))
    print('Sample {}: {}\n'.format(i+1, sample))
    print("num samples: ", len(samples))
    return samples

def calculate_annealing_weight(epoch, kl_annealing_type, kl_annealing_start, kl_annealing_growth_rate, kl_annealing_epoch):
    if kl_annealing_type == 'linear':
        kl_weight = kl_annealing_start + kl_annealing_growth_rate * epoch
        kl_weight = min(kl_weight, 1)
    elif kl_annealing_type == 'sigmoid':
        kl_weight = 1 / (1 + np.exp(-1 * (epoch - kl_annealing_epoch / 2)))
    elif kl_annealing_type == 'exponential':
        kl_weight = kl_annealing_start * np.exp(kl_annealing_growth_rate * epoch)
        kl_weight = min(kl_weight, 1)
    elif kl_annealing_type == 'step':
        kl_weight = 1 if epoch >= kl_annealing_epoch else 0
    else:
        kl_weight = 1
    return kl_weight

def train_model(model, training_options, sample_options, data_loader, device):
    rnn_type = training_options['rnn_type']
    epochs = training_options['epochs']
    optimizer = torch.optim.Adam(model.parameters(), lr=training_options['learning_rate'])

    model.train()
    out_dict = {}
    for epoch in range(epochs):
        # KL Annealing
        kl_weight = calculate_annealing_weight(epoch, training_options['kl_annealing'], training_options['kl_annealing_start'], training_options['kl_annealing_growth_rate'], training_options['kl_annealing_epoch'])

        total_loss = 0
        total_kld_loss = 0
        total_recon_loss = 0
        for batch in tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch = batch.to(device)
            batch = batch.squeeze().transpose(0, 1)
            optimizer.zero_grad()
            lengths = calculate_lengths(batch, 1)
            kld_loss, recon_loss, _, _ = model(rnn_type, batch, lengths=lengths)
            loss = kl_weight * kld_loss + recon_loss
            loss.backward()
            optimizer.step()

            nn.utils.clip_grad_norm_(model.parameters(), args['clip'])
            total_kld_loss += kld_loss.item()
            total_recon_loss += recon_loss.item()
            total_loss += kld_loss.item() + recon_loss.item()

        print(f"Epoch {epoch+1}, KLD Loss: {round(total_kld_loss / len(data_loader), 2)}, Recon Loss: {round(total_recon_loss / len(data_loader))}, Loss: {round(total_loss / len(data_loader))}")

        # save model
        save_every = 1
        if epoch % save_every == 0:
            fn = f"output/vrnn/intermediate/{sample_options['file_int']}_vrnn_{rnn_type}_{sample_options['sentence_or_article']}_{epoch}.pth"
            torch.save(model, fn)

        # Generate samples
        samples = generate_samples(model, rnn_type, sample_options['num_samples'], sample_options['sample_length'])

        # Add to output dictionary
        out_dict[f"epoch{epoch+1}"] = {"kld" : total_kld_loss / len(data_loader),
                                        "rec" : total_recon_loss / len(data_loader),
                                        "loss": total_loss / len(data_loader),
                                        "sample" : samples}

    # Save output dictionary
    pickle_fn = f"plots/output_dicts/vrnn/{sample_options['file_int']}_vrnn_{rnn_type}_{sample_options['sentence_or_article']}.pkl"
    print("Dumping Pickle to", pickle_fn)
    with open(pickle_fn, 'wb') as file:
        pickle.dump(out_dict, file)

    # Save final model
    fn = f"output/vrnn/{sample_options['file_int']}_vrnn_{rnn_type}_{sample_options['sentence_or_article']}_final.pth"
    print("Saved Final Model to", fn)
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
        'z_dim': args['z_dim'],
        'h_dim': args['h_dim'],
        'n_layers': args['n_layers'],
        'bias': False,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }
    print("Model parameters: ", {'rnn_type': args['rnn_type'], 'embed_dim': args['embed_dim'], 'z_dim': args['z_dim'], 'h_dim': args['h_dim'], 'n_layers': args['n_layers'], 'bias': False})

    # Training Options
    training_options = {
        "rnn_type": args['rnn_type'],
        "epochs": args['epochs'],
        "learning_rate": args['learning_rate'],
        "kl_annealing": args['kl_annealing'],
        "kl_annealing_start": args['kl_annealing_start'],
        "kl_annealing_growth_rate": args['kl_annealing_growth_rate'],
        "kl_annealing_epoch": args['kl_annealing_epoch']
    }
    print("Training options: ", {'epochs': training_options['epochs'], 'learning rate': training_options['learning_rate'], 'KL Annealing Type': training_options['kl_annealing']})
    print('\n')

    # Sample options
    sample_options = {
        'num_samples': args['num_samples'],
        'sample_length': args['sample_length'],
        'save_samples': args['save_samples'],
        'sentence_or_article': args['sentence_or_article'],
        'file_int': get_file_index()
    }

    # Train model
    model = VRNN(model_parameters)
    model.to(args['device'])
    train_model(model, training_options, sample_options, data_loader, args['device'])


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