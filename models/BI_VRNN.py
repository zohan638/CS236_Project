import numpy as np
import torch
import torch.nn as nn
import sys
from tqdm import tqdm
from nltk.tokenize import word_tokenize

# EPS = torch.finfo(torch.float).eps # numerical logs
EPS = 1e-8

class BIDIRECTIONAL_VRNN(nn.Module):
    def __init__(self, kwargs, bias=False):
        
        super(BIDIRECTIONAL_VRNN, self).__init__()
        
        # Base variables
        self.vocab = kwargs['vocab']
        self.embed_dim = kwargs['embed_dim']
        self.input_dim = len(self.vocab)
        self.hid_dim = kwargs['h_dim']
        self.n_layers = kwargs['n_layers']
        self.z_dim = kwargs['z_dim']
        self.bias = bias
        self.device = kwargs['device']

        # Embedding layer
        self.embed = nn.Embedding(self.input_dim+1, self.embed_dim, padding_idx=self.vocab['<pad>'])
        
        # Feature-extracting transformations
        self.phi_x = nn.Sequential(
            nn.Linear(self.embed_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU())
        self.phi_z = nn.Sequential(
            nn.Linear(self.z_dim, self.hid_dim),
            nn.ReLU())

        # Adjust hidden dimension for bidirectional GRU
        bidir_hid_dim = self.hid_dim * 2

        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(self.hid_dim + bidir_hid_dim, bidir_hid_dim),
            nn.ReLU(),
            nn.Linear(bidir_hid_dim, bidir_hid_dim),
            nn.ReLU())
        self.enc_mean = nn.Linear(bidir_hid_dim, self.z_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(bidir_hid_dim, self.z_dim),
            nn.Softplus())

        # Prior
        self.prior = nn.Sequential(
            nn.Linear(bidir_hid_dim, bidir_hid_dim),
            nn.ReLU())
        self.prior_mean = nn.Linear(bidir_hid_dim, self.z_dim)
        self.prior_std = nn.Sequential(
            nn.Linear(bidir_hid_dim, self.z_dim),
            nn.Softplus())

        # Decoder
        self.dec = nn.Sequential(
            nn.Linear(self.hid_dim + bidir_hid_dim, bidir_hid_dim),
            nn.ReLU(),
            nn.Linear(bidir_hid_dim, bidir_hid_dim),
            nn.ReLU())
        self.dec_std = nn.Sequential(
            nn.Linear(bidir_hid_dim, self.embed_dim),
            nn.Softplus())
        self.dec_mean = nn.Sequential(
            nn.Linear(bidir_hid_dim, self.embed_dim),
            nn.Sigmoid())

        # Recurrence
        if kwargs['rnn_type'] == 'GRU':
            self.rnn = nn.GRU(self.hid_dim + self.hid_dim, self.hid_dim, self.n_layers, self.bias, bidirectional=True)
        elif kwargs['rnn_type'] == 'LSTM':
            self.rnn = nn.LSTM(self.hid_dim + self.hid_dim, self.hid_dim, self.n_layers, self.bias, bidirectional=True)

        # Output layer
        self.out = nn.Linear(bidir_hid_dim, self.input_dim)

    def forward(self, rnn_type, inputs, lengths):
        if rnn_type == 'GRU':
            return self.forward_GRU(inputs, lengths)
        elif rnn_type == 'LSTM':
            return self.forward_LSTM(inputs, lengths)

    def forward_GRU(self, inputs, lengths):
        # Forward embedding layer
        x = self.embed(inputs)
        batch_size = x.shape[1]

        # Pack the sequences for RNN
        # x = torch.nn.utils.rnn.pack_padded_sequence(emb, lengths).data

        # Initialize loss variables
        all_enc_mean, all_enc_std = [], []
        all_dec_mean, all_dec_std = [], []
        all_output_logits = []
        kld_loss = 0.0
        recon_loss = 0.0
        
        # Initialize hidden state
        h = torch.zeros(self.n_layers * 2, batch_size, self.hid_dim, device=self.device)

        for t in range(x.shape[0]):
            # Phi_x
            phi_x_t = self.phi_x(x[t])

            # Forward and backward hidden state
            h_fwd = h[-2]
            h_bwd = h[-1]
            h_combined = torch.cat([h_fwd, h_bwd], dim=1)

            # Encoder
            enc_t = self.enc(torch.cat([phi_x_t, h_combined], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t) 

            # Prior
            prior_t = self.prior(h_combined)
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # Sampling and reparameterization
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)

            # Decoder
            dec_t = self.dec(torch.cat([phi_z_t, h_combined], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_std_t = self.dec_std(dec_t)

            # Recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            # Output logits
            output_logits_t = self.out(dec_t)
            all_output_logits.append(output_logits_t)

            # Reconstruction loss
            recon_loss += self.calculate_recon_loss(output_logits_t, inputs[t])

            # KL Divergence loss
            kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)

            all_enc_std.append(enc_std_t)
            all_enc_mean.append(enc_mean_t)
            all_dec_mean.append(dec_mean_t)
            all_dec_std.append(dec_std_t)

        return kld_loss, recon_loss, \
            (all_enc_mean, all_enc_std), \
            (all_dec_mean, all_dec_std)
    
    def forward_LSTM(self, inputs, lengths):
        # Forward embedding layer
        x = self.embed(inputs)
        batch_size = x.shape[1]

        # Pack the sequences for RNN
        # x = torch.nn.utils.rnn.pack_padded_sequence(emb, lengths).data

        # Initialize loss variables
        all_enc_mean, all_enc_std = [], []
        all_dec_mean, all_dec_std = [], []
        all_output_logits = []
        kld_loss = 0.0
        recon_loss = 0.0

        # Initialize hidden state
        h = torch.zeros(self.n_layers * 2, batch_size, self.hid_dim, device=self.device)
        c = torch.zeros(self.n_layers * 2, batch_size, self.hid_dim, device=self.device)

        for t in range(x.shape[0]):
            # Phi_x
            phi_x_t = self.phi_x(x[t])

            # Forward and backward hidden state
            h_fwd = h[-2]
            h_bwd = h[-1]
            h_combined = torch.cat([h_fwd, h_bwd], dim=1)

            # Encoder
            enc_t = self.enc(torch.cat([phi_x_t, h_combined], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t) 

            # Prior
            prior_t = self.prior(h_combined)
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # Sampling and reparameterization
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)

            # Decoder
            dec_t = self.dec(torch.cat([phi_z_t, h_combined], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_std_t = self.dec_std(dec_t)

            # Recurrence
            _, (h, c) = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), (h, c))

            # Output logits
            output_logits_t = self.out(dec_t)
            all_output_logits.append(output_logits_t)

            # Reconstruction loss
            recon_loss += self.calculate_recon_loss(output_logits_t, inputs[t])

            # KL Divergence loss
            kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)

            all_enc_std.append(enc_std_t)
            all_enc_mean.append(enc_mean_t)
            all_dec_mean.append(dec_mean_t)
            all_dec_std.append(dec_std_t)

        return kld_loss, recon_loss, \
            (all_enc_mean, all_enc_std), \
            (all_dec_mean, all_dec_std)
    
    def sample(self, rnn_type, seq_len):
        if rnn_type == 'GRU':
            return self.sample_GRU(seq_len)
        elif rnn_type == 'LSTM':
            return self.sample_LSTM(seq_len)
    
    def sample_GRU(self, seq_len):
        sample_words = []

        # Initialize hidden state for the forward and backward RNNs
        h_fwd = torch.zeros(2*self.n_layers, 1, self.hid_dim, device=self.device)
        h_bwd = torch.zeros(2*self.n_layers, 1, self.hid_dim, device=self.device)

        for _ in range(seq_len):
            # Prior
            prior_t = self.prior(torch.cat([h_fwd[-1], h_bwd[-1]], dim=1))
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # Sampling and reparameterization
            z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
            phi_z_t = self.phi_z(z_t)

            # Decoder
            dec_t = self.dec(torch.cat([phi_z_t, h_fwd[-1], h_bwd[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)

            # Output logits
            logits = self.out(dec_t)
            probs = torch.softmax(logits, dim=1)
            distribution = torch.distributions.Categorical(probs)
            word_index = distribution.sample()  
            word = self.find_closest_word(word_index, self.vocab)
            sample_words.append(word)

            # Update phi_x_t for the next timestep
            phi_x_t = self.phi_x(dec_mean_t)

            # Recurrence for forward RNN
            _, h_fwd = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h_fwd)

            # Recurrence for backward RNN
            _, h_bwd = self.rnn(torch.cat([phi_x_t.flip(0), phi_z_t.flip(0)], 1).unsqueeze(0), h_bwd)

        # Convert the generated words to a string
        sample_string = ' '.join(sample_words)

        return sample_string
    
    def sample_LSTM(self, seq_len):
        sample_words = []

        # Initialize hidden state for the forward and backward RNNs
        h_fwd = torch.zeros(2*self.n_layers, 1, self.hid_dim, device=self.device)
        h_bwd = torch.zeros(2*self.n_layers, 1, self.hid_dim, device=self.device)

        c_fwd = torch.zeros(2*self.n_layers, 1, self.hid_dim, device=self.device)
        c_bwd = torch.zeros(2*self.n_layers, 1, self.hid_dim, device=self.device)

        for _ in range(seq_len):
            # Prior
            prior_t = self.prior(torch.cat([h_fwd[-1], h_bwd[-1]], dim=1))
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # Sampling and reparameterization
            z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
            phi_z_t = self.phi_z(z_t)

            # Decoder
            dec_t = self.dec(torch.cat([phi_z_t, h_fwd[-1], h_bwd[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)

            # Output logits
            logits = self.out(dec_t)
            probs = torch.softmax(logits, dim=1)
            distribution = torch.distributions.Categorical(probs)
            word_index = distribution.sample()  
            word = self.find_closest_word(word_index, self.vocab)
            sample_words.append(word)

            # Update phi_x_t for the next timestep
            phi_x_t = self.phi_x(dec_mean_t)

            # Recurrence for forward RNN
            _, (h_fwd, c_fwd) = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), (h_fwd, c_fwd))

            # Recurrence for backward RNN
            _, (h_bwd, c_bwd) = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), (h_bwd, c_bwd))

        # Convert the generated words to a string
        sample_string = ' '.join(sample_words)

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
        reconstruction_loss = criterion(output_logits.view(-1, self.input_dim), targets.view(-1))
        return reconstruction_loss


    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""

        kld_element =  (2 * torch.log(std_2 + EPS) - 2 * torch.log(std_1 + EPS) + 
            (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
            std_2.pow(2) - 1)
        return	0.5 * torch.mean(kld_element)

    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.empty(size=std.size(), device=self.device, dtype=torch.float).normal_()
        return eps.mul(std).add_(mean)

    def _nll_bernoulli(self, theta, x):
        return - torch.sum(x*torch.log(theta + EPS) + (1-x)*torch.log(1-theta-EPS))


    def _nll_gauss(self, mean, std, x):
        return torch.sum(torch.log(std + EPS) + np.log(2*torch.pi)/2 + (x - mean).pow(2)/(2*std.pow(2)))
    

    def revise_text(self, input_sequence):
        # Convert input_sequence to tensor
        input_tensor = self.prepare_input_sequence(input_sequence)
        input_tensor = input_tensor.transpose(0, 1)

        # Get sequence length
        seq_len = input_tensor.shape[0]

        # Embedding the input sequence
        x = self.embed(input_tensor)

        # Initialize hidden states for bidirectional RNN
        h = torch.zeros(self.n_layers * 2, 1, self.hid_dim, device=self.device)
        all_h_combined = []

        # Forward pass through entire text sequence to get hidden states
        for t in range(seq_len):
            # Phi_x
            phi_x_t = self.phi_x(x[t])

            # Forward and backward hidden state
            h_fwd = h[-2]
            h_bwd = h[-1]
            h_combined = torch.cat([h_fwd, h_bwd], dim=1)
            # h_combined = h_combined.squeeze(0) # Remove batch dimension

            # Encoder
            enc_t = self.enc(torch.cat([phi_x_t, h_combined], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t) 

            # Prior
            prior_t = self.prior(h_combined)
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # Sampling and reparameterization
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)

            # Update RNN hidden state
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            # Store combined hidden states
            h_fwd = h[-2]
            h_bwd = h[-1]
            h_combined = torch.cat([h_fwd, h_bwd], dim=1)
            all_h_combined.append(h_combined)
        
        # Initialize revised sequence
        revised_sequence = []

        # Iterate through every word in input sequence to regenerate
        for t in range(seq_len):
            # Use the stored hidden states to generate the revised word
            h_combined = all_h_combined[t]

            # Prior
            prior_t = self.prior(h_combined)
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # Sampling and reparameterization
            z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
            phi_z_t = self.phi_z(z_t)

            # Decoder
            dec_t = self.dec(torch.cat([phi_z_t, h_combined], 1))
            logits = self.out(dec_t)
            probs = torch.softmax(logits, dim=1)
            distribution = torch.distributions.Categorical(probs)
            word_index = distribution.sample()  

            # Retrieve word from index
            word = self.find_closest_word(word_index, self.vocab)
            revised_sequence.append(word)

        # Reconstruct as string
        revised_string = ' '.join(revised_sequence)
        return revised_string

    def prepare_input_sequence(self, input_sequences):
        # Tokenize sequences
        tokenized_sequences = self.tokenize_words(input_sequences)

        # Vectorize sequences
        vectorized_sequences = []
        for tokens in tokenized_sequences:
            vectorized_sentence = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
            vectorized_sequences.append(vectorized_sentence)

        # Padding, maybe

        # Convert to tensor
        tensor = torch.tensor(vectorized_sequences, device=self.device)

        return tensor

    def tokenize_words(self, input_sequences):
        tokenized_sequences = []
        for sequence in input_sequences:
            tokens = word_tokenize(sequence.lower())
            tokenized_sequences.append(tokens)
        return tokenized_sequences

    def sample_with_revision(self, seq_len, n):
        sample = torch.zeros(seq_len, self.embed_dim, device=self.device)
        sample_words = []
        temp_storage = []

        # Forward and backward hidden states
        h_forward = torch.zeros(self.n_layers, 1, self.hid_dim, device=self.device)
        h_backward = torch.zeros(self.n_layers, 1, self.hid_dim, device=self.device)

        for t in range(seq_len):
            if t % n == 0 and t > 0:
                # Perform backward pass and regenerate the last n words
                h_backward = torch.zeros(self.n_layers, 1, self.hid_dim, device=self.device)
                
                for i in range(n):
                    _, h_backward = self.rnn_backward(self.phi_x_backward(temp_storage[i][0]).unsqueeze(0), h_backward)
                    h_combined = torch.cat([temp_storage[i][1], h_backward[-1]], 1)
                    temp_storage[i] = self.regenerate_word(h_combined)
                    sample_words[-n+i] = temp_storage[i][0]

            # Reset last_n_words for the next segment
            temp_storage = []

            # Combine forward and backward hidden states
            h_combined = torch.cat([h_forward[-1], h_backward[-1]], 1)

            # Prior
            prior_t = self.prior(h_combined)
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # Sampling and reparameterization
            z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
            phi_z_t = self.phi_z(z_t)

            # Decoder
            dec_t = self.dec(torch.cat([phi_z_t, h_combined], 1))
            dec_mean_t = self.dec_mean(dec_t)

            # Output logits
            logits = self.out(dec_t)

            # Sample word
            probs = torch.softmax(logits, dim=1)
            distribution = torch.distributions.Categorical(probs)
            word_index = distribution.sample()
            word = self.find_closest_word(word_index, self.vocab)
            sample_words.append(word)
            phi_x_t = self.phi_x(dec_mean_t)

            # Recurrence
            _, h_forward = self.rnn_forward(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h_forward)

        sample_string = ' '.join(sample_words)
        return sample, sample_string

    def regenerate_word(self, h_combined):
        # Prior
        prior_t = self.prior(h_combined)
        prior_mean_t = self.prior_mean(prior_t)
        prior_std_t = self.prior_std(prior_t)

        # Sampling and reparameterization
        z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
        phi_z_t = self.phi_z(z_t)

        # Decoder
        dec_t = self.dec(torch.cat([phi_z_t, h_combined], 1))
        dec_mean_t = self.dec_mean(dec_t)

        # Output logits
        logits = self.out(dec_t)

        # Sample word
        probs = torch.softmax(logits, dim=1)
        distribution = torch.distributions.Categorical(probs)
        word_index = distribution.sample()
        word = self.find_closest_word(word_index, self.vocab)

        return word, h_combined


