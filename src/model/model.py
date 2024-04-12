from torch import nn
from .convolution import Conv_2d
import torchaudio


class Model(nn.Module):
    def __init__(self, num_classes,
                 sample_rate=22050,
                 num_channels=16,
                 n_fft=1024,
                 f_min=0.0,
                 f_max=11025.0,
                 num_mels=128):
        super(Model, self).__init__()

        # Mel spectrogram transformation
        self.melspec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                            n_fft=n_fft,
                                                            f_min=f_min,
                                                            f_max=f_max,
                                                            n_mels=num_mels)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        self.input_bn = nn.BatchNorm2d(1)

        # Convolutional layers
        self.layer1 = Conv_2d(1, num_channels, pooling=(2, 3))
        self.layer2 = Conv_2d(num_channels, num_channels, pooling=(3, 4))
        self.layer3 = Conv_2d(num_channels, num_channels * 2, pooling=(2, 5))
        self.layer4 = Conv_2d(num_channels * 2, num_channels * 2, pooling=(3, 3))
        self.layer5 = Conv_2d(num_channels * 2, num_channels * 4, pooling=(3, 4))

        # Dense layers
        self.dense1 = nn.Linear(num_channels * 4, num_channels * 4)
        self.dense_bn = nn.BatchNorm1d(num_channels * 4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.dense2 = nn.Linear(num_channels * 4, num_classes)

    def forward(self, wav):
        """
        Forward pass through the model

        Args:
            wav (torch.Tensor): Input waveform tensor

        Returns:
            torch.Tensor: Output logits tensor
        """
        # Input preprocessing
        out = self.melspec(wav)
        out = self.amplitude_to_db(out)

        # Input batch normalization
        out = out.unsqueeze(1)
        out = self.input_bn(out)

        # Convolutional layers
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        out = out.reshape(len(out), -1)

        # Dense layers
        out = self.dense1(out)
        out = self.dense_bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.dense2(out)

        return out
