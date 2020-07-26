import torch.nn as nn

class PoseNet_1branch(nn.Module):
    def __init__(self, backbone, heatmap_head):
        super(PoseNet_1branch, self).__init__()
        self.backbone = backbone
        self.heatmap_head = heatmap_head

    def forward(self, x):
        x = self.backbone(x)
        x = self.heatmap_head(x)
        heatmap = x[:, :4]
        tagmap = x[:, 4:]

        return heatmap, tagmap

class PoseNet_2branch(nn.Module):
    def __init__(self, backbone, heatmap_head, tagmap_head):
        super(PoseNet_2branch, self).__init__()
        self.backbone = backbone
        self.heatmap_head = heatmap_head
        self.tagmap_head = tagmap_head

    def forward(self, x):
        x = self.backbone(x)
        heatmap = self.heatmap_head(x)
        tagmap = self.tagmap_head(x)

        return heatmap, tagmap