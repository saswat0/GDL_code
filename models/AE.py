import numpy as np
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_x_to_determine_size, input_channels, enc_output_channels, enc_kernel_size, enc_strides, enc_padding, dec_output_channels, dec_kernel_size, dec_strides, dec_padding, dec_op_padding, z_dim):

        # Encoder Network
        enc_conv_layers = []
        enc_output_channels.insert(0, input_channels)

        for input_channels, output_channels, kernel_size, stride, pad in zip(enc_output_channels[0:], enc_output_channels[1:], enc_kernel_size, enc_strides, enc_padding):
            enc_conv_layer = []
            enc_conv_layer.append(nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding=pad))
            enc_conv_layer.extend([nn.LeakyReLU(), nn.BatchNorm2d(output_channels), nn.Dropout(.25)])
            enc_conv_layers.append(nn.Sequential(*enc_conv_layer))

        # Decoder Network
        dec_conv_layers = []
        dec_output_channels.insert(0, output_channels)
        n_layers_decoder = range(len(dec_output_channels))

        for input_channels, output_channels, kernel_size, stride, i, pad, op_pad in zip(dec_output_channels[0:], dec_output_channels[1:], dec_kernel_size, dec_strides, n_layers_decoder, dec_padding, dec_op_padding):

            dec_conv_layer = []
            dec_conv_layer.append(nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding=pad, output_padding=op_pad))
            if i == len(dec_output_channels) - 2:
                dec_conv_layer.append(nn.Sigmoid())
            else:
                dec_conv_layer.extend([nn.LeakyReLU(), nn.BatchNorm2d(output_channels), nn.Dropout(.25)])
            dec_conv_layers.append(nn.Sequential(*dec_conv_layer))

        # Dynamically determine the sizes
        x = nn.Sequential(*enc_conv_layers)(input_x_to_determine_size)
        pre_flatten_shape = x.shape
        x = nn.Flatten()(x)

        enc_conv_layers.append(nn.Flatten())
        enc_conv_layers.append(nn.Linear(x.shape[1], z_dim))
        self.enc_conv_layers = nn.Sequential(*enc_conv_layers)
        dec_conv_layers.insert(0, nn.Linear(z_dim, np.prod(pre_flatten_shape[1:])))
        dec_conv_layers.insert(1, nn.View(pre_flatten_shape))
        self.dec_conv_layers = nn.Sequential(*dec_conv_layers)

    def forward(self, x):
        x = self.enc_conv_layers(x)
        return self.dec_conv_layers(x)