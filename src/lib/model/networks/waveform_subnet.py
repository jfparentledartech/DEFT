import torch
from torch import nn
from lib.model import layers

BN_MOMENTUM = 0.1

class WaveformSubnet(nn.Module):
    def __init__(self):
        super().__init__()

        self.waveforms_block_1 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(32, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.AvgPool3d(kernel_size=(8,1,1), stride=(8,1,1)),
            nn.ConvTranspose3d(in_channels=32, out_channels=32, kernel_size=(1,2,2),
                                        stride=(1,2,2), padding=0)
        )
        self.waveforms_block_temp_1 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.waveforms_block_temp_2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(128, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.waveforms_block_temp_3 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.waveforms_block_temp_4 = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(32, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.waveforms_block_2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.AvgPool3d(kernel_size=(8,1,1), stride=(8,1,1)),
            nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=(1,5,5),
                                        stride=(1,5,5), padding=0)
        )
        self.waveforms_block_3 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(32, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.AvgPool3d(kernel_size=(12,1,1), stride=(12,1,1)),
            nn.ConvTranspose3d(in_channels=32, out_channels=32, kernel_size=(1,2,2),
                                        stride=(1,2,2), padding=0)
        )

    def forward(self, trace):
        trace = self.waveforms_block_1(trace)
        trace = self.waveforms_block_temp_1(trace)
        trace = self.waveforms_block_temp_2(trace)
        trace = self.waveforms_block_temp_3(trace)
        trace = self.waveforms_block_temp_4(trace)
        trace = self.waveforms_block_2(trace)
        trace = self.waveforms_block_3(trace)
        trace = trace.squeeze(2)
        return trace


class WaveformDenseSubnet(nn.Module):
    def __init__(self):
        super().__init__()

        self.decapitation_1 = layers.Decapitation(in_channels=16, out_channels=16, kernel=(3,3,7))
        self.dense_block_1 = layers.DenseBlock(in_channels=16, growth=2, n_layers=2, kernel=(3,3,7))
        # self.dense_block_1 = layers.DenseBlock(in_channels=16, growth=16, n_layers=2, kernel=(3,3,7))
        self.compressor_1 = layers.Compressor(in_channels=self.dense_block_1.out_channels, out_channels=24, kernel=(1,1,3), shrink=(1,1,8))

        self.decapitation_2 = layers.Decapitation(in_channels=self.compressor_1.out_channels, out_channels=32, kernel=(3,3,5))
        self.upscale_decap_2 = layers.Upscale(in_channels=self.decapitation_2.out_channels, out_channels=32, kernel=(4,4), up_factor=(4,4))

        # self.dense_block_2 = layers.DenseBlock(in_channels=self.compressor_1.out_channels, growth=20, n_layers=4, kernel=(3,3,5))
        self.dense_block_2 = layers.DenseBlock(in_channels=self.compressor_1.out_channels, growth=4, n_layers=2, kernel=(3,3,5))
        self.compressor_2 = layers.Compressor(in_channels=self.dense_block_2.out_channels, out_channels=80, kernel=(1,1,3), shrink=(1,1,4))

        self.decapitation_3 = layers.Decapitation(in_channels=self.compressor_2.out_channels, out_channels=48, kernel=(3,3,3))
        self.upscale_decap_3 = layers.Upscale(in_channels=self.decapitation_3.out_channels, out_channels=48, kernel=(2,2), up_factor=(2,2))

        # self.dense_block_3 = layers.DenseBlock(in_channels=self.compressor_2.out_channels, growth=20, n_layers=8, kernel=(3,3,3))
        self.dense_block_3 = layers.DenseBlock(in_channels=self.compressor_2.out_channels, growth=2, n_layers=4, kernel=(3,3,3))
        self.compressor_3 = layers.Compressor(in_channels=self.dense_block_3.out_channels, out_channels=200, kernel=(1,1,3), shrink=(1,1,8))

        nb_decapitated_channels = self.decapitation_1.out_channels + self.decapitation_2.out_channels + self.decapitation_3.out_channels
        self.compressor_4 = layers.Compressor(in_channels=self.compressor_3.out_channels + nb_decapitated_channels, out_channels=160, kernel=(1,1), shrink=(1,1))

        # self.dense_block_4 = layers.DenseBlock(in_channels=self.compressor_4.out_channels, growth=10, n_layers=12, kernel=(3,3))
        self.dense_block_4 = layers.DenseBlock(in_channels=self.compressor_4.out_channels, growth=2, n_layers=4, kernel=(3,3))
        self.upscale_1 = layers.Upscale(in_channels=self.dense_block_4.out_channels, out_channels=120, kernel=(2,2), up_factor=(2,2))

        self.dense_block_5 = layers.DenseBlock(in_channels=self.upscale_1.out_channels + self.upscale_decap_3.out_channels, growth=2, n_layers=2, kernel=(3,3))
        # self.dense_block_5 = layers.DenseBlock(in_channels=self.upscale_1.out_channels + self.upscale_decap_3.out_channels, growth=12, n_layers=4, kernel=(3,3))
        self.upscale_2 = layers.Upscale(in_channels=self.dense_block_5.out_channels, out_channels=80, kernel=(2,2), up_factor=(2,2))

        # self.dense_block_6 = layers.DenseBlock(in_channels=self.upscale_2.out_channels + self.upscale_decap_2.out_channels, out_channels=16, growth=6, n_layers=1, kernel=(3,3))
        self.dense_block_6 = layers.DenseBlock(in_channels=self.upscale_2.out_channels + self.upscale_decap_2.out_channels, out_channels=16, growth=2, n_layers=1, kernel=(3,3))
        self.upscale_3 = layers.Upscale(in_channels=self.dense_block_6.out_channels, out_channels=80, kernel=(5,5), up_factor=(5,5))

        # self.dense_block_out = layers.DenseBlock(in_channels=self.upscale_3.out_channels, out_channels=16, growth=6, n_layers=1, kernel=(3,3))
        self.dense_block_out = layers.DenseBlock(in_channels=self.upscale_3.out_channels, out_channels=16, growth=2, n_layers=1, kernel=(3,3))


    def forward(self, x):
        x = x.permute(0,1,3,4,2)
        xd1 = self.decapitation_1(x)
        x = self.dense_block_1(x)
        x = self.compressor_1(x)

        xd2 = self.decapitation_2(x)
        x = self.dense_block_2(x)
        x = self.compressor_2(x)

        xd3 = self.decapitation_3(x)
        x = self.dense_block_3(x)
        x = self.compressor_3(x)

        x = x.squeeze(-1)
        x = torch.cat((x,xd1,xd2,xd3), dim=1)
        x = self.compressor_4(x)

        x = self.dense_block_4(x)
        x = self.upscale_1(x)

        xd3_up = self.upscale_decap_3(xd3)
        x = torch.cat((x,xd3_up), dim=1)

        x = self.dense_block_5(x)
        x = self.upscale_2(x)

        xd2_up = self.upscale_decap_2(xd2)
        x = torch.cat((x,xd2_up), dim=1)

        x = self.dense_block_6(x)
        x = self.upscale_3(x)

        x = self.dense_block_out(x)
        # from matplotlib import pyplot as plt
        # plt.imshow(x.detach().cpu()[0,0])
        x = torch.nn.functional.interpolate(x, (160,720))
        return x