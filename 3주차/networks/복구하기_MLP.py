import torch
import torch.nn as nn

class myMLP(nn.Modeul):
    def __init__(self, hidden_size, num_classes):
        super(myMLP,self).__init__()
        self.fc1 = (28*28, hidden_size)
        self.fc1 = (hidden_size, hidden_size)
        self.fc1 = (hidden_size, hidden_size)
        self.fc1 = (hidden_size, num_classes)

    def forward(self, x):
        x = x.reshape(-1,28,28)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        return x