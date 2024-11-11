import torch.nn as nn


class DecodingLayer(nn.Module):
    def __init__(self, label_mapping):
        super(DecodingLayer, self).__init__()
        self.label_mapping = label_mapping

    def forward(self, input_tensor):
        decoded_labels = [
            [self.label_mapping[int(label)] for label in time_step]
            for time_step in input_tensor
        ]
        return decoded_labels


class LipsyncModel(nn.Module):
    def __init__(self, input_size, num_visemes, label_mapping, dropout_ratio=0.5):
        super(LipsyncModel, self).__init__()
        self.lstm = nn.LSTM(input_size, 200, batch_first=True)
        self.dropout = nn.Dropout(dropout_ratio)
        self.fc = nn.Linear(200, num_visemes)
        self.decoder = DecodingLayer(label_mapping)  # Decoding layer

    def forward(self, input_tensor, decode=False):
        """
        :param input_tensor: The input tensor for the model
        :param decode: Set to False during training, True during inference
        """

        out, _ = self.lstm(input_tensor)
        out = self.dropout(out)
        out = self.fc(out)

        if decode:
            out = self.decoder(out.argmax(dim=2))
        return out

