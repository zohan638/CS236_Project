import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class VariationalRNN(nn.Module):
    def __init__(self, **kwargs, bidirectional=False):
        super(VariationalRNN, self).__init__()

        # Define number of directions
        self.num_directions = 2 if bidirectional else 1

        # Base variables
        self.vocab = kwargs['vocab']
        self.in_dim = len(self.vocab) + 1
        self.start_token = kwargs['start_token']
        self.end_token = kwargs['end_token']
        self.unk_token = kwargs['unk_token']
        self.characters = kwargs['characters']
        self.embed_dim = kwargs['embedding_size']
        self.hid_dim = kwargs['hidden_size']
        self.n_layers = kwargs['num_layers']
        self.nonlinearity = kwargs['nonlinearity']

        #Define the embedding layer
        self.embed = nn.Embedding(input_size=self.embed_dim,self.embed_dim,padding_idx=self.in_dim)

        # Latent variables
        self.latent_dim = kwargs['latent_dim']
        
        # Define RNN layer
        # Note: We removed batch_first=True
        self.rnn = nn.RNN(input_size=self.embed_dim, hidden_size=self.hid_dim, num_layers=self.n_layers, 
                        nonlinearity=self.nonlinearity.lower(), bidirectional=bidirectional)
        
        # Define layers for variational part
        self.hidden_to_mean = nn.Linear(self.hid_dim * self.num_directions, self.latent_dim)
        self.hidden_to_logvar = nn.Linear(self.hid_dim * self.num_directions, self.latent_dim)
        self.latent_to_hidden = nn.Linear(self.latent_dim, self.hid_dim * self.num_directions)

        # Define output layer
        self.hidden_to_output = nn.Linear(self.hid_dim * self.num_directions, output_size=self.in_dim)
        
        #Define the softmax layer
        self.softmax = nn.LogSoftmax(dim=1)

    def init_weights(self):
        #Randomly initialise all parameters
        torch.nn.init.xavier_uniform_(self.embed.weight)
        # TODO: Consider if this needs to change for bidirectional
        if self.num_directions > 1:
            raise NotImplementedError
        for i in range(self.n_layers):
            torch.nn.init.xavier_uniform_(getattr(self.rnn,'weight_hh_l'+str(i)))
            torch.nn.init.xavier_uniform_(getattr(self.rnn,'weight_ih_l'+str(i)))
            torch.nn.init.uniform_(getattr(self.rnn,'bias_hh_l'+str(i)))
            torch.nn.init.uniform_(getattr(self.rnn,'bias_ih_l'+str(i)))
        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.uniform_(self.linear.bias)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, lengths):
        batch_size = x.size(1)

        # Embedding layer
        emb = self.embed(x)

        # Pack the sequences for RNN
        packed = torch.nn.utils.rnn.pack_padded_sequence(emb, lengths)

        # RNN forward
        out, self.hidden = self.rnn(packed, self.hidden)

        # Unpack the sequences
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out)

        # Multiple layers
        hidden = hidden.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)
        hidden = hidden[-1]  # Take the last layer
        hidden = hidden.transpose(0, 1).contiguous().view(batch_size, -1) # Reshape to (batch_size, hidden_size * num_directions)
        
        # Variational part
        mu = self.hidden_to_mean(hidden)
        logvar = self.hidden_to_logvar(hidden)
        z = self.reparameterize(mu, logvar)
        hidden = self.latent_to_hidden(z)

        # Output layer
        out = self.hidden_to_output(hidden)

        return out, mu, logvar
