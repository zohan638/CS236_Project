from torchviz import make_dot
from models.BI_VRNN import BIDIRECTIONAL_VRNN
import argparse
import torch
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.onnx
from torch.utils.tensorboard import SummaryWriter
import hiddenlayer as hl

transforms = [ hl.transforms.Prune('Constant') ]

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a bidirectional variational RNN')

    # Model hyperparameters
    parser.add_argument('--rnn_type', type=str, default='GRU', help='Type of RNN to use (LSTM or GRU)')
    parser.add_argument('--embed_dim', type=int, default=128, help='Size of the embedding layer')
    parser.add_argument('--z_dim', default=56, type=int, help='Dimensionality of the latent variable')
    parser.add_argument('--h_dim', type=int, default=256, help='Size of the hidden recurrent layers')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of recurrent layers')
    parser.add_argument('--learning_rate', type=float, default=.001, help='Learning rate')

    # Training options
    parser.add_argument('--sentence_or_article', type=str, default='sentence', help='Whether to use sentences (sentence) or articles (article)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    # parser.add_argument('--batch_size', type=int, default=50, help='Batch size')
    parser.add_argument('--batch_size', type=int, default=200, help='Batch size')
    parser.add_argument('--seed', type=float, default=128, help='Random seed')
    parser.add_argument('--kl_annealing', type=str, default='linear', help='Type of KL Annealing to Use')
    # parser.add_argument('--kl_annealing_start', type=float, default=0.05, help='The starting value for the KL weight')
    # parser.add_argument('--kl_annealing_growth_rate', type=float, default=0.05, help='KL Annealing growth rate for linear/exponential annealing')
    parser.add_argument('--kl_annealing_start', type=float, default=0.0, help='The starting value for the KL weight')
    parser.add_argument('--kl_annealing_growth_rate', type=float, default=0.01, help='KL Annealing growth rate for linear/exponential annealing')
    parser.add_argument('--kl_annealing_epoch', type=int, default=25, help='The end/middle epoch for KL weight, depending on type of annealing')
    parser.add_argument('--clip', default=2.0, type=int, help='Gradient clipping')

    # Sample options
    parser.add_argument('--num_samples', default=1, type=int, help='Number of samples to generate after every epoch')
    parser.add_argument('--sample_length', default=100, type=int, help='Length of samples')

    # Save and plot options
    parser.add_argument('--save_samples', default=True, type=bool, help='Whether to save samples & losses')
    parser.add_argument('--save_every', default=10, type=int, help='Save model every n epochs')
    parser.add_argument('--print_every', default=1, type=int, help='Print every n batches')
    
    args = parser.parse_args()
    args = vars(args)
    return args

args = parse_arguments()

if args['sentence_or_article'] == 'sentence':
        padded_sequences = np.load('data/vrnn_padded_sentences.npy')[0:2, :]
        print("Using Sentences - Padded sequences shape: ", padded_sequences.shape)
        with open('data/vrnn_vocabulary_sentences.json', 'r') as f:
            vocab = json.load(f)
elif args['sentence_or_article'] == 'article':
    padded_sequences = np.load('data/vrnn_padded_articles.npy')[0:2, :]
    print("Using Articles - Padded sequences shape: ", padded_sequences.shape)
    with open('data/vrnn_vocabulary_articles.json', 'r') as f:
        vocab = json.load(f)  

padded_sequences_tensor = torch.from_numpy(padded_sequences).long()

def calculate_lengths(batch, padding_idx):
    lengths = (batch != padding_idx).sum(dim=1)
    return lengths

class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

dataset = TextDataset(padded_sequences_tensor)
data_loader = DataLoader(dataset, batch_size=args['batch_size'], shuffle=True)

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

model = BIDIRECTIONAL_VRNN(model_parameters)
model.load_state_dict(torch.load("/home/jaymartin/project_files/CS236_Project/output/bivrnn/72_bivrnn_GRU_sentence_final.pth", map_location=model_parameters['device']))
model.eval()
# model.to(model_parameters['device'])
# model.load_state_dict
inputs = next(iter(data_loader))
inputs = inputs.to(model_parameters['device'])
inputs = inputs.transpose(0, 1)
lengths = calculate_lengths(inputs, 1)
# writer = SummaryWriter("torchlogs/")
# writer.add_graph(model, list([model_parameters['rnn_type'], inputs, lengths]))
# writer.close()
# onnx_path = "model.onnx"
# torch.onnx.export(model, (model_parameters['rnn_type'], inputs, lengths), onnx_path, export_params=False, do_constant_folding=True, input_names = ['modelInput'], output_names = ['modelOutput'])
# kld_loss, recon_loss, enc_stats, dec_stats = model(model_parameters['rnn_type'], inputs, lengths)
# make_dot((kld_loss, recon_loss), params=dict(model.named_parameters())).render("model_graph", format="png", cleanup=True)
graph = hl.build_graph(model, (model_parameters['rnn_type'], inputs, lengths), transforms=transforms)