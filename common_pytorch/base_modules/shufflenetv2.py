import torch.nn as nn
from torchvision.models.shufflenetv2 import InvertedResidual

shufflenet_spec = {'shufflenetv2_x0.5': ([4, 8, 4], [24, 48, 96, 192, 1024]),
                   'shufflenetv2_x1.0': ([4, 8, 4], [24, 116, 232, 464, 1024])}
                   # 'shufflenetv2_x1.5': ([4, 8, 4], [24, 176, 352, 704, 1024]), # no-pretrained model, useless
                   # 'shufflenetv2_x2.0': ([4, 8, 4], [24, 244, 488, 976, 2048])}

class ShuffleNetV2_Backbone(nn.Module):
    def __init__(self, stages_repeats, stages_out_channels, inverted_residual=InvertedResidual):
        super(ShuffleNetV2_Backbone, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        # ----------- remove -----------
        # self.fc = nn.Linear(output_channels, num_classes)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        # ----------- remove -----------
        # x = x.mean([2, 3])  # globalpool
        # x = self.fc(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)