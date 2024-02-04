import torch
import torch.nn as nn


class CNN_RNN_Classifier(nn.Module):
    def __init__(self, in_channels=None, out_size=None, rnn_hidden_size=64, rnn_num_layers=1):
        super(CNN_RNN_Classifier, self).__init__()

        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers

        # CNN layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 22 * 22, 512)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, stride=2)

        # RNN layer
        self.rnn = nn.LSTM(64 * 22 * 22, rnn_hidden_size, rnn_num_layers, batch_first=True)

        # Fully connected layer for classification
        self.fc2 = nn.Linear(rnn_hidden_size, out_size)

        self.dropout = nn.Dropout(0.0)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()  # (batch,seq,3,96,96)
        c_in = x.view(batch_size * seq_len, c, h, w).float()

        x = self.relu(self.conv1(c_in))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool1(x)  # (batch size * seq, 64, 22, 22)
        x = x.reshape(batch_size * seq_len, -1)
        # rnn_input = self.relu(self.dropout(self.fc1(x)))
        rnn_input = self.relu(self.dropout(x))
        rnn_input = rnn_input.view(batch_size, seq_len, -1)
        rnn_out, _ = self.rnn(rnn_input)

        output = self.fc2(rnn_out)  # [batch, seq, 4]

        return output


class CNNClassifier(nn.Module):  # Architecture
    def __init__(self, in_channels=None, out_size=None):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 22 * 22, 512)
        self.fc2 = nn.Linear(512, out_size)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.0)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool1(x)
        x = x.view(x.size(0), -1)

        x = self.relu(self.dropout(self.fc1(x)))
        x = self.fc2(x)

        return x
