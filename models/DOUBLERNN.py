import torch
import torch.nn as nn
from models.VRNN_copy import VRNN

class BidirectionalVRNN(nn.Module):
    def __init__(self, kwargs, bias=False):
        super(BidirectionalVRNN, self).__init__()
        self.forward_vrnn = VRNN(kwargs)
        self.backward_vrnn = VRNN(kwargs)

    def forward(self, batch, lengths):
        # Calculate sequence lengths for the reversed sequence
        reversed_lengths = lengths.clone()

        # Forward VRNN
        forward_kld_loss, forward_recon_loss, forward_states, forward_output_logits = self.forward_vrnn(batch, lengths)

        # Reverse the input and lengths and pass through backward VRNN
        reversed_input = torch.flip(batch, [0])  # Reversing the sequence
        backward_kld_loss, backward_recon_loss, backward_states, backward_output_logits = self.backward_vrnn(reversed_input, reversed_lengths)

        # Convert forward and backward states to tensors
        forward_states = (torch.stack(forward_states[0]), torch.stack(forward_states[1]))
        backward_states = (torch.stack(backward_states[0]), torch.stack(backward_states[1]))

        # Concatenate the outputs from forward and backward VRNNs
        combined_kld_loss = forward_kld_loss + backward_kld_loss
        combined_recon_loss = forward_recon_loss + backward_recon_loss
        combined_states = (torch.cat((forward_states[0], backward_states[0]), dim=-1), 
                           torch.cat((forward_states[1], backward_states[1]), dim=-1))
        combined_output_logits = torch.cat((forward_output_logits, backward_output_logits), dim=-1)

        return combined_kld_loss, combined_recon_loss, combined_states, combined_output_logits
