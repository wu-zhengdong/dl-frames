from torch import nn

'''
This file include the base model of deep learning.
'''


class conv_bn_net(nn.Module):
    def __init__(self, conv_bn, linear_layers):
        super(conv_bn_net, self).__init__()

        self.conv_layer = conv_bn

        self.predict = linear_layers

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.shape[0], -1)
        out = self.predict(x)
        return out


class lstm_network(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout, activate_function):
        super(lstm_network, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.predict = nn.Sequential(
            activate_function,
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x):
        x, (h, o) = self.rnn(x, None)
        # 将最后一个时间片扔进 Linear
        out = self.predict(x[:, -1, :])
        return out