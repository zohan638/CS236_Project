import numpy as np
import torch
import torch.nn as nn
import sys
from scipy.spatial.distance import cosine

# EPS = torch.finfo(torch.float).eps # numerical logs
EPS = 1e-8

class VRNN(nn.Module):
    def __init__(self, kwargs, bias=False):
        
        super(VRNN, self).__init__()
        #Base variables
        self.vocab = kwargs['vocab']
        #self.start_token = kwargs['start_token']
        #self.end_token = kwargs['end_token']
        #self.unk_token = kwargs['unk_token']
        #self.characters = kwargs['characters']
        self.embed_dim = kwargs['embedding_size']
        self.input_dim = len(self.vocab)
        self.hid_dim = kwargs['h_dim']
        self.n_layers = kwargs['n_layers']
        self.z_dim = kwargs['z_dim']
        self.bias = bias
        self.device = kwargs['device']

        #Define the embedding layer
        # self.embed = nn.Embedding(self.input_dim+1, self.embed_dim, padding_idx=self.input_dim)
        self.embed = nn.Embedding(self.input_dim+1, self.embed_dim, padding_idx=self.vocab['<pad>'])
        
        #feature-extracting transformations
        self.phi_x = nn.Sequential(
            nn.Linear(self.embed_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU())
        '''
        self.phi_x = nn.Sequential(
            nn.Linear(self.input_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU())
        '''
        self.phi_z = nn.Sequential(
            nn.Linear(self.z_dim, self.hid_dim),
            nn.ReLU())

        #encoder
        self.enc = nn.Sequential(
            nn.Linear(self.hid_dim + self.hid_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU())
        self.enc_mean = nn.Linear(self.hid_dim, self.z_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(self.hid_dim, self.z_dim),
            nn.Softplus())

        #prior
        self.prior = nn.Sequential(
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU())
        self.prior_mean = nn.Linear(self.hid_dim, self.z_dim)
        self.prior_std = nn.Sequential(
            nn.Linear(self.hid_dim, self.z_dim),
            nn.Softplus())

        #decoder
        self.dec = nn.Sequential(
            nn.Linear(self.hid_dim + self.hid_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU())
        self.dec_std = nn.Sequential(
            nn.Linear(self.hid_dim, self.embed_dim),
            nn.Softplus())
        #self.dec_std = nn.Linear(self.hid_dim, self.embed_dim)
        #self.dec_mean = nn.Linear(self.hid_dim, self.embed_dim)
        self.dec_mean = nn.Sequential(
            nn.Linear(self.hid_dim, self.embed_dim),
            nn.Sigmoid())
        '''
        self.dec = nn.Sequential(
            nn.Linear(self.hid_dim + self.hid_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU())
        self.dec_std = nn.Sequential(
            nn.Linear(self.hid_dim, self.input_dim),
            nn.Softplus())
        # self.dec_std = nn.Linear(self.hid_dim, self.in_dim)
        self.dec_mean = nn.Linear(self.hid_dim, self.input_dim)
        # self.dec_mean = nn.Sequential(
        #     nn.Linear(self.hid_dim, self.in_dim),
        #     nn.Sigmoid())
        '''

        #recurrence
        self.rnn = nn.GRU(self.hid_dim + self.hid_dim, self.hid_dim, self.n_layers, self.bias)

        # output layer
        self.out = nn.Linear(self.hid_dim, self.input_dim)

    def forward(self, inputs, lengths):
        #Forward embedding layer
        x = self.embed(inputs)

        #Pack the sequences for RNN
        # x = torch.nn.utils.rnn.pack_padded_sequence(emb, lengths).data

        all_enc_mean, all_enc_std = [], []
        all_dec_mean, all_dec_std = [], []
        all_output_logits = []
        kld_loss = 0.0
        nll_loss = 0.0
        # h = torch.zeros(self.n_layers, x.shape[1], self.hid_dim, device=self.device)
        h = torch.zeros(self.n_layers, x.shape[1], self.hid_dim, device=self.device)
        for t in range(x.shape[0]):
            # Phi_x
            phi_x_t = self.phi_x(x[t])

            # Encoder
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t) 

            # Prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # Sampling and reparameterization
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)

            # Decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_std_t = self.dec_std(dec_t)

            # Recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            # Output logits
            output_logits_t = self.out(dec_t)
            all_output_logits.append(output_logits_t)

            # Reconstruction loss
            nll_loss += self.calculate_recon_loss(output_logits_t, inputs[t])
            # nll_loss += self._nll_gauss(dec_mean_t, dec_std_t, x[t])
            # nll_loss += self._nll_bernoulli(dec_mean_t, x[t])

            # KL Divergence loss
            kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)

            all_enc_std.append(enc_std_t)
            all_enc_mean.append(enc_mean_t)
            all_dec_mean.append(dec_mean_t)
            all_dec_std.append(dec_std_t)

        return kld_loss, nll_loss, \
            (all_enc_mean, all_enc_std), \
            (all_dec_mean, all_dec_std), \
            all_output_logits
    
    def sample(self, seq_len):

        sample = torch.zeros(seq_len, self.embed_dim, device=self.device)
        sample_words = []

        h = torch.zeros(self.n_layers, 1, self.hid_dim, device=self.device)
        for t in range(seq_len):

            #prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            #sampling and reparameterization
            z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
            phi_z_t = self.phi_z(z_t)

            #decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)

            # output logits
            logits = self.out(dec_t)
            probs = torch.softmax(logits, dim=1)
            distribution = torch.distributions.Categorical(probs)
            word_index = distribution.sample()  
            word = self.find_closest_word(word_index, self.vocab)
            sample_words.append(word)

            # Update phi_x_t for next timestep
            phi_x_t = self.phi_x(dec_mean_t)

            #recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)
        
        sample_string = ' '.join(sample_words)

        return sample, sample_string

    def find_closest_word(self, word_index, vocab):
        index_to_word = {index: word for word, index in vocab.items()}
        return index_to_word[word_index.item()]
    
    '''
    def find_closest_word(self, word_vector, vocab):
        word_vector = word_vector / torch.norm(word_vector)  # Normalize the embedding
        min_dist = float('inf')
        closest_word_idx = -1

        index_to_word = {index: word for word, index in vocab.items()}

        for idx, word_embedding in enumerate(self.embed.weight):
            # Skip the padding_idx
            if idx == self.vocab['<pad>']:
                continue

            word_embedding = word_embedding / torch.norm(word_embedding)  # Normalize word_embedding
            dist = cosine(word_vector.detach().cpu().numpy(), word_embedding.detach().cpu().numpy())
            if dist < min_dist:
                min_dist = dist
                closest_word_idx = idx

        return index_to_word[closest_word_idx]
    '''


    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)


    def _init_weights(self, stdv):
        pass


    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.empty(size=std.size(), device=self.device, dtype=torch.float).normal_()
        return eps.mul(std).add_(mean)


    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""

        kld_element =  (2 * torch.log(std_2 + EPS) - 2 * torch.log(std_1 + EPS) + 
            (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
            std_2.pow(2) - 1)
        return	0.5 * torch.mean(kld_element)


    def _nll_bernoulli(self, theta, x):
        return - torch.sum(x*torch.log(theta + EPS) + (1-x)*torch.log(1-theta-EPS))


    def _nll_gauss(self, mean, std, x):
        return torch.sum(torch.log(std + EPS) + np.log(2*torch.pi)/2 + (x - mean).pow(2)/(2*std.pow(2)))

    def calculate_recon_loss(self, output_logits, targets):
        criterion = nn.CrossEntropyLoss(reduction='mean')
        reconstruction_loss = criterion(output_logits.view(-1, self.input_dim), targets.view(-1))
        return reconstruction_loss


