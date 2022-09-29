from typing import List, Tuple, Any

import torch
from torch import nn, Tensor
from torch.nn import functional
from loss import FCOSLoss
from postprocess import FCOSPostprocessor

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using {device} device")
device = "cpu"


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

        self.global_avg_pool = nn.AvgPool2d(self.n)
        self.dense1 = nn.Linear(self.channels, self.channels // self.ratio, bias=False)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(self.channels // self.ratio, self.channels, bias=False)
        self.hardswish = nn.Hardswish()

    def forward(self, x: Tensor) -> Tensor:  # x = BN ReLU Block
        block: Tensor = x
        x = self.global_avg_pool(x)
        x = x.view(x.shape[0], self.channels)
        x = self.relu(self.dense1(x))
        x = self.hardswish(self.dense2(x))
        x = x[:, :, None, None]

        return x * block


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
            self.branch1 = nn.Sequential()  # why? vmtl nur für die model() Ausgabe

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
        else:
            out: Tensor = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class SPP(nn.Module):
    def __init__(self):
        super(SPP, self).__init__()
        self.maxpool5 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.maxpool9 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
        self.maxpool13 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)

    def forward(self, x):
        # spielt die Reihenfolge eine Rolle?
        return torch.cat((self.maxpool13(x), self.maxpool9(x), self.maxpool5(x), x), dim=1)


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        asu = AttentiveShuffleNetUnit
        stages_repeats = [3, 7, 3]  # oder 4,8,4 wie im ShuffleNetv2?
        self._stage_out_channels = [116, 232, 464]
        self.height_width = [52, 26, 13]

        self.conv1 = nn.Conv2d(3, 24, kernel_size=3, stride=2)  # bias = False? laut shufflenetv2 repo
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)  # laut EY paper keine BN und relu Schicht hier

        input_channels = 24
        stage_names = [f'stage{i}' for i in [2, 3, 4]]
        spp_names = [f'spp{i}' for i in range(2, 5)]
        for name, spp, repeats, output_channels, height_width in zip(stage_names, spp_names,
                                                                     stages_repeats,
                                                                     self._stage_out_channels,
                                                                     self.height_width):
            seq = [asu(input_channels, output_channels, height_width, stride=2)]
            for _ in range(repeats - 1):
                seq.append(asu(output_channels, output_channels, height_width, stride=1))
            setattr(self, name, nn.Sequential(*seq))
            setattr(self, spp, SPP())
            input_channels = output_channels

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        # ASU-SPP Network - backbone
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        spp2: Tensor = self.spp2(x)
        assert spp2.shape[1:] == torch.Size([464, 52, 52])

        x = self.stage3(x)
        spp3: Tensor = self.spp3(x)
        assert spp3.shape[1:] == torch.Size([928, 26, 26])

        x = self.stage4(x)
        spp4: Tensor = self.spp4(x)
        assert spp4.shape[1:] == torch.Size([1856, 13, 13])
        assert x.shape[1:] == torch.Size([464, 13, 13])

        return spp2, spp3, spp4


class Neck(nn.Module):
    def __init__(self):
        super(Neck, self).__init__()

        self.neck52_conv = nn.Conv2d(464, 96, kernel_size=1, stride=1)  # Bn und ReLU?
        self.neck26_conv = nn.Conv2d(928, 96, kernel_size=1, stride=1)
        self.neck13_conv = nn.Conv2d(1856, 96, kernel_size=1, stride=1)

    def forward(self, spp: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        # PANet-Tiny - neck
        # neck52 N x 96 x 52 x 52
        neck52 = self.neck52_conv(spp[0])
        # neck26 N x 96 x 26 x 26
        neck26 = self.neck26_conv(spp[1])
        # neck13 N x 96 x 13 x 13
        neck13 = self.neck13_conv(spp[2])

        # neck13_up N x 96 x 26 x 26
        neck13_up = functional.interpolate(neck13, scale_factor=2, mode='bilinear')
        assert neck13_up.shape[1:] == torch.Size([96, 26, 26])
        # neck26 N x 96 x 26 x 26
        neck26 = torch.add(neck26, neck13_up)
        assert neck26.shape[1:] == torch.Size([96, 26, 26])
        # neck26_up N x 96 x 52 x 52
        neck26_up = functional.interpolate(neck26, scale_factor=2, mode='bilinear')
        assert neck26_up.shape[1:] == torch.Size([96, 52, 52])
        # output1 N x 96 x 52 x 52
        output1 = torch.add(neck52, neck26_up)
        assert output1.shape[1:] == torch.Size([96, 52, 52])
        # output1_down N x 96 x 26 x 26
        output1_down = functional.interpolate(output1, scale_factor=0.5, mode='bilinear')
        assert output1_down.shape[1:] == torch.Size([96, 26, 26])
        # output2 N x 96 x 26 x 26
        output2 = torch.add(output1_down, neck26)
        assert output2.shape[1:] == torch.Size([96, 26, 26])
        # output2_down N x 96 x 13 x 13
        output2_down = functional.interpolate(output2, scale_factor=0.5, mode='bilinear')
        assert output2_down.shape[1:] == torch.Size([96, 13, 13])
        # output2_down N x 96 x 13 x 13
        output3 = torch.add(output2_down, neck13)
        assert output3.shape[1:] == torch.Size([96, 13, 13])

        return output1, output2, output3


class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=5, stride=1, padding=2, groups=96),
            nn.Conv2d(96, 96, kernel_size=1, stride=1),
            nn.Conv2d(96, 96, kernel_size=5, stride=1, padding=2, groups=96),
            nn.Conv2d(96, 96, kernel_size=1, stride=1),
            nn.Conv2d(96, 9, kernel_size=1, stride=1)  # 4 classes + 4 Bbox-Werte + 1 centerness
        )

    def forward(self, output: Tensor) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        # tiny-head
        logits = []
        bboxes = []
        centers = []
        # split up head into class, bboxes and centerness
        output1 = self.head(output[0])
        assert output1.shape[1:] == torch.Size([9, 52, 52])
        logits.append(output1[:, 0:4, :, :])
        bboxes.append(output1[:, 4:8, :, :])
        centers.append(output1[:, 8:, :, :])

        output2 = self.head(output[1])
        assert output2.shape[1:] == torch.Size([9, 26, 26])
        logits.append(output2[:, 0:4, :, :])
        bboxes.append(output2[:, 4:8, :, :])
        centers.append(output2[:, 8:, :, :])

        output3 = self.head(output[2])
        assert output3.shape[1:] == torch.Size([9, 13, 13])
        logits.append(output3[:, 0:4, :, :])
        bboxes.append(output3[:, 4:8, :, :])
        centers.append(output3[:, 8:, :, :])

        return logits, bboxes, centers


class EmbeddedYolo(nn.Module):

    def __init__(self, opt):
        super().__init__()

        self.backbone = Backbone()

        print("backbone params (all):" + str(sum(p.numel() for p in self.backbone.parameters())))
        self.neck = Neck()
        print("neck params (all):" + str(sum(p.numel() for p in self.neck.parameters())))
        self.head = Head()
        self.fpn_strides = [8, 16, 32]
        self.loss = FCOSLoss(
            self.fpn_strides
        )
        self.postprocessor = FCOSPostprocessor(
            opt.conf_threshold,
            opt.top_n,
            opt.nms_threshold,
            opt.post_top_n,
            opt.min_size,
            opt.n_class,
        )

    # def train(self, mode=True):
    #     super().train(mode)
    #
    #     def freeze_bn(module):
    #         if isinstance(module, nn.BatchNorm2d):
    #             module.eval()
    #
    #     self.apply(freeze_bn)

    def forward(self, input, image_sizes=None, targets=None, detection=False):
        features = self.backbone(input)
        features = self.neck(features)
        cls_pred, box_pred, center_pred = self.head(features)
        location = self.compute_location(features)

        if self.training:
            loss_cls, loss_box, loss_center = self.loss(
                location, cls_pred, box_pred, center_pred, targets
            )
            losses = {
                'loss_cls': loss_cls,
                'loss_box': loss_box,
                'loss_center': loss_center,
            }

            return None, losses

        else:
            boxes = self.postprocessor(
                location, cls_pred, box_pred, center_pred, image_sizes
            )
            if detection:
                return boxes, None

            loss_cls, loss_box, loss_center = self.loss(
                location, cls_pred, box_pred, center_pred, targets
            )
            losses = {
                'loss_cls': loss_cls,
                'loss_box': loss_box,
                'loss_center': loss_center,
            }

            return boxes, losses

    def compute_location(self, features):
        # Koordinatenberechnung der Mittelpunkte der Feature Map Grid Zellen (in Pixel bezogen auf Ursprungsbild)
        locations = []

        for i, feat in enumerate(features):  # größte bis kleinste feature map
            _, _, height, width = feat.shape
            location_per_level = self.compute_location_per_level(
                height, width, self.fpn_strides[i], feat.device
            )
            locations.append(location_per_level)

        return locations

    def compute_location_per_level(self, height, width, stride, device):
        shift_x = torch.arange(
            0, width * stride, step=stride, dtype=torch.float32, device=device
        )
        shift_y = torch.arange(
            0, height * stride, step=stride, dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        location = torch.stack((shift_x, shift_y), 1) + stride // 2  # // 2 damit Rückübersetzung im Zentrum liegt

        return location


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    opt.n_class = 5
    opt.conf_threshold = 0.05
    opt.top_n = 1000
    opt.nms_threshold = 0.6
    opt.post_top_n = 100
    opt.min_size = 0
    model = EmbeddedYolo(opt).to(device)
    # print(model)

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(sum(p.numel() for p in model.parameters()))
    # fake_pic = torch.rand(1, 3, 416, 416, device=device)
    # logits = model(fake_pic)
    # print(logits.shape)
