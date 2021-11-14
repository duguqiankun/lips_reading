import torch


class BidirectionalLSTM(torch.nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = torch.nn.LSTM(nIn, nHidden, bidirectional=True, batch_first=True)
        self.embedding = torch.nn.Linear(nHidden * 2, nOut)

    def forward(self, inputs):
        recurrent, _ = self.rnn(inputs)
        T, b, h = recurrent.size()
        t_rec = recurrent.reshape(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.reshape(T, b, -1)
        output = torch.nn.functional.softmax(output, dim=-1)

        return output


class videoModel(torch.nn.Module):
    def __init__(self, number_classes=28, max_len=6):
        """

        :param number_classes:
        our char dictionary is:
        0: <blank>
        1: a
        2: b
        3: c
        ...
        26: z
        27: <eos>
        :param max_len: max_len = 6,
        Suppose we said abcde,
        the the label should be abcde<eos>
        abc -> abc<eos><blank><blank>
        number_classes = 28, 26 characters + <eos> + <blank>
        """
        super(videoModel, self).__init__()
        self.number_classes = number_classes
        self.max_len = max_len
        self.mlp1= self.mlp(491520, 2000)
        self.mlp2 = self.mlp(2000,2000)
        self.mlp3 = self.mlp(2000, 1000*6)

        self.lstm_decoder = BidirectionalLSTM(nIn=1000,
                                              nHidden=256,
                                              nOut=number_classes)

    def _conv_block(self, input_c, output_c):
        conv_block = torch.nn.Sequential(
            torch.nn.Conv3d(input_c, output_c, kernel_size=(3, 3, 2), padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv3d(output_c, output_c, kernel_size=(3, 3, 2), padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool3d((2, 2, 2))
        )
        return conv_block


    def mlp(self, input_size, output_size):
        conv_block = torch.nn.Sequential(
            torch.nn.Linear(input_size, output_size),
            torch.nn.ReLU(),
            torch.nn.Linear(output_size, output_size),
            torch.nn.ReLU(),
        )
        return conv_block

    def forward(self, x):
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        shape = x.size()
        # bs, 256, 3, 3, 14
        x = x.view(shape[0], self.max_len, -1)  # bs, max_len, rest
        x = self.lstm_decoder(x)
        return x


if __name__ == '__main__':
    # input video,
    # batch_size, 2457600)
    batch_size = 5

    x = torch.rand(batch_size, 491520)

    model = videoModel()

    y = model(x)
    print(y.size())  # [5, 6, 28]
