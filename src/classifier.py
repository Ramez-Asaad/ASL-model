
import torch
import torch.nn as nn

class ASLClassifier(nn.Module):
    def __init__(self, input_size=42, hidden_size=64, num_layers=2, num_classes=10):
        """
        A simple LSTM-based classifier for ASL signs.
        Args:
            input_size (int): Number of input features per time step (21 * 2 = 42).
            hidden_size (int): Number of features in the hidden state.
            num_layers (int): Number of recurrent layers.
            num_classes (int): Number of output classes (vocabulary size).
        """
        super(ASLClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # BatchFirst=True means input shape is (batch, seq, feature)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        
        # Initialize hidden and cell states
        # Default initialization (zeros) is usually fine, but explicit here for clarity
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        # out shape: (batch_size, seq_length, hidden_size)
        out = out[:, -1, :]
        
        out = self.fc(out)
        return out
