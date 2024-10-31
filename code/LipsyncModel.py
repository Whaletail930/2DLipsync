import torch.nn as nn


class LipsyncModel(nn.Module):
    def __init__(self, input_size, num_visemes):
        super(LipsyncModel, self).__init__()
        self.lstm = nn.LSTM(input_size, 200, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(200, num_visemes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # Use the last valid output
        out = self.fc(out)
        return out