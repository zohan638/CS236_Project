import torch
from torch import nn, optim
from torch.autograd import Variable
 
class VRNN_Small(nn.Module):
    def __init__(self):
        super(VRNNCell,self).__init__()
        self.phi_x = nn.Sequential(nn.Embedding(128,64), nn.Linear(64,64), nn.ELU())
        self.encoder = nn.Linear(128,64*2) # output hyperparameters
        self.phi_z = nn.Sequential(nn.Linear(64,64), nn.ELU())
        self.decoder = nn.Linear(128,128) # logits
        self.prior = nn.Linear(64,64*2) # output hyperparameters
        self.rnn = nn.GRUCell(128,64)

    def forward(self, x, hidden):
        x = self.phi_x(x)
        z_prior = self.prior(hidden)
        z_infer = self.encoder(torch.cat([x,hidden], dim=1))
        z = Variable(torch.randn(x.size(0),64))*z_infer[:,64:].exp()+z_infer[:,:64]
        z = self.phi_z(z)
        x_out = self.decoder(torch.cat([hidden, z], dim=1))
        hidden_next = self.rnn(torch.cat([x,z], dim=1),hidden)
        return x_out, hidden_next, z_prior, z_infer

    def sample_vrnn(self, seq_length, start_token):
        """
        Generate a sequence of data from the VRNN model.

        :param model: The trained VRNN model
        :param seq_length: Length of the sequence to generate
        :param start_token: Starting token for sequence generation
        :return: Generated sequence
        """
        self.eval()  # Set the model to evaluation mode

        generated_sequence = []
        hidden = torch.zeros(1, 64)  # Assuming hidden size is 64
        x = start_token

        for _ in range(seq_length):
            x = self.phi_x(x)
            z_prior = self.prior(hidden)
            mu_prior, log_sigma_prior = z_prior[:, :64], z_prior[:, 64:]
            z = Variable(torch.randn(1, 64)) * log_sigma_prior.exp() + mu_prior
            x_out = self.decoder(torch.cat([hidden, z], dim=1))
            hidden = self.rnn(torch.cat([x, z], dim=1), hidden)

            # Assuming the output x_out is the next token (e.g., in a language model)
            # For continuous data, you might sample from a distribution parameterized by x_out
            next_token = torch.argmax(x_out, dim=1)
            generated_sequence.append(next_token.item())
            x = next_token

        return generated_sequence

    def calculate_loss(self, x, hidden):
        x_out, hidden_next, z_prior, z_infer = self.forward(x, hidden)

        # 1. logistic regression loss
        loss1 = nn.functional.cross_entropy(x_out, x)

        # 2. KL Divergence between Multivariate Gaussian
        mu_infer, log_sigma_infer = z_infer[:,:64], z_infer[:,64:]
        mu_prior, log_sigma_prior = z_prior[:,:64], z_prior[:,64:]
        loss2 = (2*(log_sigma_infer-log_sigma_prior)).exp() \
                + ((mu_infer-mu_prior)/log_sigma_prior.exp())**2 \
                - 2*(log_sigma_infer-log_sigma_prior) - 1
        loss2 = 0.5*loss2.sum(dim=1).mean()
        return loss1, loss2, hidden_next