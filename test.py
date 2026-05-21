import torch
from torchvision.models.resnet import ResNet, BasicBlock
import torch.nn as nn

class ContinualResNet(ResNet):
    def __init__(self, num_classes=100):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        self.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.maxpool = nn.Identity()
    def forward(self, x, task_id=None, **kwargs):
        return super().forward(x)

model = ContinualResNet()
x = torch.randn(128, 3, 32, 32)
out = model(x)
print('MIN:', out.min().item(), 'MAX:', out.max().item())
