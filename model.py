import torch
import torch.nn as nn
import torchvision.models as models


class MnistCnn(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(MnistCnn, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=(5, 5), padding=2),
            nn.GroupNorm(4, 8),
            nn.ReLU()
        )
        self.pool1 = nn.AvgPool2d(2)
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(3, 3)),
            nn.GroupNorm(4, 16),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 3)),
            nn.GroupNorm(4, 32),
            nn.ReLU()
        )
        self.pool2 = nn.AvgPool2d(2)
        self.fc = nn.Linear(5 * 5 * 32, num_classes)

    def forward(self, x):
        out = self.pool1(self.layer1(x))
        out = self.pool2(self.layer3(self.layer2(out)))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class FashionMnistCnn(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(FashionMnistCnn, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(5, 5), padding=2),
            nn.GroupNorm(4, 16),
            nn.ReLU()
        )
        self.pool1 = nn.AvgPool2d(2)
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 3)),
            nn.GroupNorm(4, 32),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3)),
            nn.GroupNorm(4, 64),
            nn.ReLU()
        )
        self.pool2 = nn.AvgPool2d(2)
        self.fc = nn.Linear(5 * 5 * 64, num_classes)

    def forward(self, x):
        out = self.pool1(self.layer1(x))
        out = self.pool2(self.layer3(self.layer2(out)))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class Cifar10Cnn(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Cifar10Cnn, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(5, 5), padding=2),
            nn.GroupNorm(4, 16),
            nn.ReLU()
        )
        self.pool1 = nn.AvgPool2d(2)
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 3)),
            nn.GroupNorm(4, 32),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3)),
            nn.GroupNorm(4, 64),
            nn.ReLU()
        )
        self.pool2 = nn.AvgPool2d(2)
        self.fc = nn.Linear(6 * 6 * 64, num_classes)

    def forward(self, x):
        out = self.pool1(self.layer1(x))
        out = self.pool2(self.layer3(self.layer2(out)))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def conv_block(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(num_groups=8, num_channels=out_channels),  # 使用 GroupNorm
        nn.ReLU()
    ]
    if pool:
        layers.append(nn.AvgPool2d(2))
    return nn.Sequential(*layers)


class CifarResNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CifarResNet, self).__init__()
        base_channels =16

        self.conv1 = conv_block(in_channels, base_channels)  # 第一层使用普通卷积
        self.conv2 = conv_block(base_channels, base_channels * 2, pool=True)  # 减少通道数
        self.res1 = nn.Sequential(
            conv_block(base_channels * 2, base_channels * 2),
            conv_block(base_channels * 2, base_channels * 2)
        )

        self.conv3 = conv_block(base_channels * 2, base_channels * 4, pool=True)  # 减少通道数
        self.conv4 = conv_block(base_channels * 4, base_channels * 8, pool=True)  # 减少通道数

        self.pool = nn.AdaptiveAvgPool2d(1)  # 使用全局平均池化
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(base_channels * 8, num_classes)  # 减少全连接层维度

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)

        out = self.res1(out) + out

        out = self.conv3(out)
        out = self.conv4(out)

        out = self.pool(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out


# class LightweightCIFAR10Net(nn.Module):
#     def __init__(self, in_channels=3, num_classes=10, groups=16):
#         super(LightweightCIFAR10Net, self).__init__()
#
#         # 卷积层块1: 输入3通道, 输出32通道
#         self.conv_block1 = nn.Sequential(
#             nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
#             nn.GroupNorm(groups, 32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 32, kernel_size=3, padding=1),
#             nn.GroupNorm(groups, 32),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#
#         # 卷积层块2: 32通道 -> 64通道
#         self.conv_block2 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.GroupNorm(groups, 64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.GroupNorm(groups, 64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#
#         # 卷积层块3: 64通道 -> 128通道
#         self.conv_block3 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.GroupNorm(groups, 128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 128, kernel_size=3, padding=1),
#             nn.GroupNorm(groups, 128),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#
#         # 全局平均池化替代全连接层
#         self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
#
#         # 分类器
#         self.fc = nn.Linear(128, num_classes)
#
#     def forward(self, x):
#         x = self.conv_block1(x)  # 输出: [32, 16, 16]
#         x = self.conv_block2(x)  # 输出: [64, 8, 8]
#         x = self.conv_block3(x)  # 输出: [128, 4, 4]
#         x = self.global_avg_pool(x)  # 输出: [128, 1, 1]
#         x = torch.flatten(x, 1)  # 展平: [128]
#         x = self.fc(x)
#         return x


if __name__ == '__main__':
    model1 = CifarResNet(in_channels=3, num_classes=10)  # VGG11(), ResNet9
    print(model1)
    print('model1 total params: {}'.format(sum(p.numel() for p in model1.parameters())))

    model2 = FashionMnistCnn(in_channels=1, num_classes=10)
    print(model2)
    print('model2 total params: {}'.format(sum(p.numel() for p in model2.parameters())))

    model3 = MnistCnn(in_channels=1, num_classes=10)
    print(model3)
    print('model3 total params: {}'.format(sum(p.numel() for p in model3.parameters())))