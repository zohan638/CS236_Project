import sys
import os
from utils.lm_func import read_vocabulary, count_sequences

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
    parser.add_argument('--epochs',type=int,default=10,help='Number of epochs to train')
    parser.add_argument('--batch_size',type=int,default=32,help='Batch size')
    parser.add_argument('--embedding_size',type=int,default=128,help='Size of the embedding layer')
    parser.add_argument('--hidden_size',type=int,default=128,help='Size of the hidden recurrent layers')
    parser.add_argument('--latent_size',type=int,default=128,help='Dimension of the latent layers')
    parser.add_argument('--num_layers',type=int,default=1,help='Number of recurrent layers')
    parser.add_argument('--learning_rate',type=float,default=0.1,help='Learning rate')
    parser.add_argument('--seed',type=float,default=0,help='Random seed')
    parser.add_argument('--bptt',type=int,default=sys.maxsize,help='Length of segments for Truncated Back-Propagation Through Time')
    parser.add_argument('--max_length',type=int,default=20,help='Maximum length of sequences to use (longer sequences are discarded)')
    parser.add_argument('--ltype',type=str,default='rnn',help='Type of hidden layers to use ("rnn", "gru", "lstm")')
    parser.add_argument('--nonlinearity',type=str,default='relu',help='Non-linear function used in the recurrent layersi (for RNN)')
    parser.add_argument('--start_token',type=str,default='<s>',help='Word token used at the beginning of a sentence')
    parser.add_argument('--end_token',type=str,default='<\s>',help='Word token used at the end of a sentence')
    parser.add_argument('--unk_token',type=str,default='<UNK>',help='Word token used for out-of-vocabulary words')
    parser.add_argument('--verbose',default=2,type=int,choices=[0,1,2],help='Verbosity level (0, 1 or 2)')
    
    args = parser.parse_args()
    args = vars(args)
    return args

def train_lm(args):
    # Loss function
    def loss_function(recon_x, x, mu, logvar):
        BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    # Initializations
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed(args['seed'])
    torch.backends.cudnn.deterministic = True
    args['device'] = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))

    # Get vocab
    args['vocab'], args['characters'] = read_vocabulary(**args)

    # Model instantiation
    model = VariationalRNN(self.vocab, args['hidden_size'], vocabulary=args['vocabulary'], args['latent_size']).to(args['device']) # in dimension is calculated as size of vocabulary + 1

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop (simplified)
    for epoch in tqdm(range(args['epochs']+1)):
        for data in dataloader:
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()

if __name__ == '__main__':
    args = parse_arguments()
    train_lm(args)