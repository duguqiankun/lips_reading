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

        return output


class VideoModel(torch.nn.Module):
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
        super(VideoModel, self).__init__()
        self.number_classes = number_classes
        self.max_len = max_len
        self.conv_block_1 = self._conv_block(3, 32)
        self.conv_block_2 = self._conv_block(32, 64)
        self.conv_block_3 = self._conv_block(64, 128)
        self.conv_block_4 = self._conv_block(128, 256)
        self.lstm_decoder = BidirectionalLSTM(nIn=5376,
                                              nHidden=256,
                                              nOut=number_classes)

    def _conv_block(self, input_c, output_c):
        conv_block = torch.nn.Sequential(
            torch.nn.Conv3d(input_c, output_c, kernel_size=(3, 3, 2), padding=1),
            torch.nn.LeakyReLU(),
            # torch.nn.BatchNorm3d(output_c),
            torch.nn.Conv3d(output_c, output_c, kernel_size=(3, 3, 2), padding=1),
            torch.nn.LeakyReLU(),
            # torch.nn.BatchNorm3d(output_c),
            torch.nn.MaxPool3d((2, 2, 2))
        )
        return conv_block

    def forward(self, x):
        x = x.permute(dims=(0, 2, 3, 4, 1))
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        shape = x.size()
        # bs, 256, 3, 3, 14
        x = x.view(shape[0], self.max_len, -1)  # bs, max_len, rest
        x = self.lstm_decoder(x)
        return x


if __name__ == '__main__':
    # input video,
    # batch_size, channel, height, width, frames (pad all video to same frames)
    batch_size = 1
    channel = 3
    fixed_height, fixed_width = 60, 60
    fixed_max_frame = 200
    # batch size, fixed_max_frame, channel, fixed_height, fixed_width
    x = torch.rand(batch_size, fixed_max_frame, channel, fixed_height, fixed_width)

    model = VideoModel()

    y = model(x)
    print(y.size())  # [5, 6, 28]
    # 5 is the batch size, 6 is the max_char_len, 28 is the char_number,
    # we can use greedy search to find the character with max score for each position

