import torch
from torch import nn, Tensor

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, height_width):
        super(SqueezeExcitation, self).__init__()
        self.channels = channels
        self.ratio = 8
        self.n: int = height_width

        self.se = nn.Sequential(
            nn.AvgPool2d(self.n),
            nn.Linear(self.channels, self.channels // self.ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.channels // self.ratio, self.channels, bias=False),
            nn.Hardswish(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:  # x = BN Block
        assert x.shape[2] == x.shape[3]
        return x * self.se(x)


class AttentiveShuffleNetUnit(nn.Module):
    def __init__(self, in_channel, out_channel, height_width, stride):
        super(AttentiveShuffleNetUnit, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride  # basic unit (stride = 1) or unit for spatial down sampling (stride=2)

        branch_features = out_channel // 2
        assert (self.stride != 1) or (in_channel == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(in_channel, in_channel, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(in_channel),
                nn.Conv2d(in_channel, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()  # why?

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channel if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )
        if self.stride == 1:
            self.se = SqueezeExcitation(branch_features, height_width)

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            branch2 = self.branch2(x2)
            branch2 = self.se(branch2)
            out: Tensor = torch.cat((x1, branch2), dim=1)
            # out: Tensor = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out: Tensor = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class EmbeddedYolo(nn.Module):
    def __init__(self):
        super(EmbeddedYolo, self).__init__()
        asu = AttentiveShuffleNetUnit
        stages_repeats = [3, 7, 3]  # oder 4,8,4 wie im ShuffleNetv2?
        self._stage_out_channels = [116, 232, 464]
        self.height_width = [52, 26, 13]

        self.conv1 = nn.Conv2d(3, 24, kernel_size=3, stride=2)  # bias = False? laut shufflenetv2 repo
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)  # laut EY paper keine BN und relu Schicht hier

        input_channels = 24
        stage_names = [f'stage{i}' for i in [2, 3, 4]]
        for name, repeats, output_channels, height_width in zip(stage_names, stages_repeats, self._stage_out_channels,
                                                                self.height_width):
            seq = [asu(input_channels, output_channels, height_width, stride=2)]
            for i in range(repeats - 1):
                seq.append(asu(output_channels, output_channels, height_width, stride=1))
            setattr(self, name, nn.Sequential(*seq))  # Seq sorgt dafür dass beim späteren durchreichen die outputs an
            # die nächste repeat schicht weitergegeben wird
            input_channels = output_channels

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        x1 = self.stage2(x)
        x2 = self.stage3(x1)
        x3 = self.stage4(x2)
        # x1,x2,x3 in spp layer überführen


if __name__ == "__main__":
    model = EmbeddedYolo().to(device)
    print(model)
