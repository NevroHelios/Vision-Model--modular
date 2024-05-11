# /content/going_modular/model_builder.py
"""
"""
import torch
import torch.nn as nn

class TinyVGG(nn.Module):
  """
    TinyVGG is a simple convolutional neural network architecture inspired by VGGNet.
    It consists of two convolutional blocks followed by a fully connected classifier.

    Args:
        input_shape (int): The number of input channels.
        hidden_units (int): The number of channels in the convolutional layers.
        output_shape (int): The number of output classes.

    Attributes:
        conv_block_1 (torch.nn.Sequential): The first convolutional block.
        conv_block_2 (torch.nn.Sequential): The second convolutional block.
        classifier (torch.nn.Sequential): The fully connected classifier.
    """
  def __init__(self, input_shape, hidden_units, output_shape):
    super().__init__()
    self.conv_block_1 = nn.Sequential(
        nn.Conv2d(in_channels=input_shape,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=0),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )
    self.conv_block_2 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=0),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=hidden_units*13*13,
                  out_features=output_shape)
    )

  def forward(self, x):
    x = self.conv_block_1(x)
    x = self.conv_block_2(x)
    return self.classifier(x)
