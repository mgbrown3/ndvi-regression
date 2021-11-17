import torch
import torch.nn as nn
import torch.nn.functional as F

# Here we define our model as a class
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 32, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]) 
        # out.size() --> 100, 10
        return out


# Re-write CNN model from Meredith with Pytorch
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        #input [B, C=1, L=6]
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=2)
        #[B, 32, 4]
        
        self.fc1 = nn.Linear(32*5, 64)
        #[B, 64]
        
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        
        x = x.flatten(1)
        
        x = F.relu(self.fc1(x))
        
        x = self.fc2(x)
        return x
