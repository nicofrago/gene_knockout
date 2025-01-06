import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(
            self, 
            output_size: int = 6, 
            input_size: int = 512, 
            dropout_ratio: float = 0.4
    ):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_ratio)
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_ratio)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
    
class SimpleModelV2(nn.Module):
    def __init__(
            self, 
            output_size: int = 6, 
            input_size: int = 512, 
            dropout_ratio: float = 0.4
    ):
        super(SimpleModelV2, self).__init__()
        self.fc = nn.Linear(input_size, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_ratio)
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_ratio)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x