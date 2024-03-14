import torch
import torch.nn as nn

class Net(nn.Module):
    """Simple neural network to forecast token trends"""
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.init_weights()

    def forward(self, x):
        x = torch.relu(self.fc1())
        x = torch.relu(self.fc2())
        x = torch.sigmoid(self.fc3())

        return x
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)