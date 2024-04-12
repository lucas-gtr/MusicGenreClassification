from torch import nn


class Conv_2d(nn.Module):
    """
    Custom convolutional layer module for 2D data

    Args:
        input_channels (int): Number of input channels
        output_channels (int): Number of output channels
        shape (int): Kernel size for convolution operation (default is 3)
        pooling (int): Kernel size for max pooling operation (default is 2)
        dropout_rate (float): Dropout probability (default is 0.1)
    """
    def __init__(self, input_channels: int, output_channels: int, shape: int = 3, pooling: tuple = (2, 2),
                 dropout_rate: float = 0.1):
        super(Conv_2d, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, shape, padding=shape // 2)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(pooling)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Forward pass of the convolutional layer

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor after passing through the convolutional layer
        """
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.max_pool(out)
        out = self.dropout(out)

        return out
