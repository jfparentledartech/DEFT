from torch import nn

BN_MOMENTUM = 0.1

class WaveformSubnet(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.waveforms_block_1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(32, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.AvgPool3d(kernel_size=(8,1,1), stride=(8,1,1)),
            nn.ConvTranspose3d(in_channels=32, out_channels=32, kernel_size=(1,4,4),
                                        stride=(1,4,4), padding=0)
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
            nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(16, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.AvgPool3d(kernel_size=(12,1,1), stride=(12,1,1)),
            nn.ConvTranspose3d(in_channels=16, out_channels=16, kernel_size=(1,2,2),
                                        stride=(1,2,2), padding=0)
        )
        self.half_features = nn.Sequential(
            nn.Conv3d(32, channels[0], kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

    def forward(self, trace):
        trace = self.waveforms_block_1(trace.unsqueeze(1))
        trace = self.waveforms_block_2(trace)
        trace = self.waveforms_block_3(trace)
        return trace