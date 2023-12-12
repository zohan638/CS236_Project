import numpy as np
import torch
import torch.nn as nn
import sys
from tqdm import tqdm
from nltk.tokenize import word_tokenize

# EPS = torch.finfo(torch.float).eps # numerical logs
EPS = 1e-8

class BIDIRECTIONAL_RNN(nn.Module):
    def __init__(self, kwargs, bias=False):
        
        super(BIDIRECTIONAL_RNN, self).__init__()
        
        # Base variables
        self.vocab = kwargs['vocab']
        self.embed_dim = kwargs['embed_dim']
        self.input_dim = len(self.vocab)
        self.hid_dim = kwargs['h_dim']
        self.n_layers = kwargs['n_layers']
        self.bias = bias
        self.device = kwargs['device']

        # Embedding layer
        self.embed = nn.Embedding(self.input_dim+2, self.embed_dim, padding_idx=self.vocab['<pad>'])

        # Recurrence
        if kwargs['rnn_type'] == 'GRU':
            self.rnn = nn.GRU(self.embed_dim, self.hid_dim, self.n_layers, self.bias, bidirectional=True)
        elif kwargs['rnn_type'] == 'LSTM':
            self.rnn = nn.LSTM(self.embed_dim, self.hid_dim, self.n_layers, self.bias, bidirectional=True)

        # Output layer
        self.out = nn.Linear(2 * self.hid_dim, self.input_dim+2)

    def forward(self, rnn_type, inputs, lengths):
        if rnn_type == 'GRU':
            return self.forward_GRU(inputs)
        elif rnn_type == 'LSTM':
            return self.forward_LSTM(inputs)

    def forward_GRU(self, inputs):
        # Forward embedding layer
        x = self.embed(inputs)
        batch_size = x.shape[1]
        
        # Initialize hidden state
        h = torch.zeros(self.n_layers * 2, batch_size, self.hid_dim, device=self.device)

        # Recurrence
        out, h = self.rnn(x, h)

        # Output logits
        logits = self.out(out)

        # Calculate loss
        loss = self.calculate_recon_loss(logits, inputs)

        return logits, loss
    
    def forward_LSTM(self, inputs):
        # Forward embedding layer
        x = self.embed(inputs)
        batch_size = x.shape[1]
        
        # Initialize hidden state
        h = torch.zeros(self.n_layers * 2, batch_size, self.hid_dim, device=self.device)
        c = torch.zeros(self.n_layers * 2, batch_size, self.hid_dim, device=self.device)

        # Recurrence
        out, (h, c) = self.rnn(x, (h,c))

        # Output logits
        # out = out.reshape(out.size(0)*out.size(1), self.hidden_size*2)
        logits = self.out(out)

        # Calculate loss
        loss = self.calculate_recon_loss(logits, inputs)

        return logits, loss
    
    def sample(self, rnn_type, seq_len, start_sequence):
        if rnn_type == 'GRU':
            return self.sample_GRU(seq_len, start_sequence)
        elif rnn_type == 'LSTM':
            return self.sample_LSTM(seq_len, start_sequence)
    
    def sample_GRU(self, num_tokens_to_generate, start_sequence):
        sequence = start_sequence
        sequence = sequence.transpose(0, 1)
        print("Sequence Shape: ", sequence.shape)

        # Initialize hidden state
        h = torch.zeros(2 * self.n_layers, 1, self.hid_dim, device=self.device)

        for _ in range(num_tokens_to_generate):
            # Embed
            embedded_sequence = self.embed(sequence)
            print("Embedded Sequence Shape: ", embedded_sequence.shape)

            # Recurrence
            out, h = self.rnn(embedded_sequence, h)
            print("Out Shape: ", out.shape)
            print("h Shape: ", h.shape)

            # Output logits
            logits = self.out(out)
            last_logits = logits[-1]
            print("Logits Shape: ", logits.shape)

            probs = torch.softmax(logits, dim=1)
            distribution = torch.distributions.Categorical(probs)
            word_index = distribution.sample()  
            word = self.find_closest_word(word_index, self.vocab)

            # Update sequence
            sequence = sequence.append(word)

        # Convert the generated words to a string
        sample_string = ' '.join(sequence)

        return sample_string
    
    def sample_LSTM(self, start_sequence, num_tokens_to_generate):
        sequence = start_sequence

        # Initialize hidden state and cell state
        h = torch.zeros(2*self.n_layers, 1, self.hid_dim, device=self.device)
        c = torch.zeros(2*self.n_layers, 1, self.hid_dim, device=self.device)

        for _ in range(num_tokens_to_generate):
            # Embed
            embedded_sequence = self.embed(sequence)

            # Recurrence
            _, (h, c) = self.rnn(embedded_sequence, (h, c))

            # Output logits
            logits = self.out(h)
            probs = torch.softmax(logits, dim=1)
            distribution = torch.distributions.Categorical(probs)
            word_index = distribution.sample()
            word = self.find_closest_word(word_index, self.vocab)

            # Update sequence
            sequence = sequence.append(word)

        # Convert the generated words to a string
        sample_string = ' '.join(sequence)

        return sample_string

    def find_closest_word(self, word_index, vocab):
        index_to_word = {index: word for word, index in vocab.items()}
        return index_to_word[word_index.item()]


    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)


    def _init_weights(self, stdv):
        pass

    def calculate_recon_loss(self, output_logits, targets):
        criterion = nn.CrossEntropyLoss(reduction='mean')

        output_logits = output_logits.reshape(-1, output_logits.shape[2])
        targets = targets.reshape(-1)

        reconstruction_loss = criterion(output_logits, targets)
        return reconstruction_loss

