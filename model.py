import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from transformers import DistilBertModel, RobertaModel
from peft import LoraConfig, get_peft_model, TaskType


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


# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, in_planes, planes, stride=1, num_groups=8):  # 默认分组数改为8
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.gn1 = nn.GroupNorm(num_groups, planes)
#
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.gn2 = nn.GroupNorm(num_groups, planes)
#
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion * planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
#                 nn.GroupNorm(num_groups, self.expansion * planes)
#             )
#
#     def forward(self, x):
#         out = F.relu(self.gn1(self.conv1(x)))
#         out = self.gn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out
#
#
# class CifarResNet(nn.Module):
#     # 1. 深度降低: 默认 blocks 从 [2,2,2,2] 减少到 [1,1,1,1]
#     # 2. 分组数降低: 适应较小的通道数，避免报错
#     def __init__(self, num_blocks=[1, 1, 1, 1], block=BasicBlock, in_channels=3, num_classes=10, num_groups=8):
#         super(CifarResNet, self).__init__()
#
#         # 3. 初始通道数从 64 降低到 16
#         self.in_planes = 16
#         self.num_groups = num_groups
#
#         # 初始卷积层
#         self.conv1 = nn.Conv2d(in_channels, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.gn1 = nn.GroupNorm(self.num_groups, self.in_planes)
#
#         # 残差层 (通道数整体变为原来的 1/4)
#         # 参数量由 ~11.1M 暴降到 ~0.3M，但对 CIFAR-10 性能依然很好
#         self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2)
#
#         # 分类头：最后输出的通道数为 128
#         self.fc = nn.Linear(128 * block.expansion, num_classes)
#
#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for s in strides:
#             layers.append(block(self.in_planes, planes, s, num_groups=self.num_groups))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         out = F.relu(self.gn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#
#         # CIFAR-10 经过 3 次 stride=2 后，尺寸从 32 降到 4。
#         # 这里池化核依然是 4，输出尺寸变为 1x1
#         out = F.avg_pool2d(out, 4)
#         out = out.view(out.size(0), -1)
#         out = self.fc(out)
#         return out


class WSConv2d(nn.Conv2d):
    """
    效果不错
    继承自标准 nn.Conv2d，在每次前向传播时对权重进行标准化 (Weight Standardization)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(WSConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                       padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight

        # 按照论文公式，在输入通道、高、宽维度 (dim=1, 2, 3) 上计算均值和标准差
        weight_mean = weight.mean(dim=(1, 2, 3), keepdim=True)
        # 使用 unbiased=False 遵循论文中除以 N 的设定
        weight_std = weight.std(dim=(1, 2, 3), unbiased=False, keepdim=True)

        # 权重标准化 (加上 1e-5 防止除零)
        standardized_weight = (weight - weight_mean) / (weight_std + 1e-5)

        # 使用标准化后的权重进行 2D 卷积运算
        return F.conv2d(x, standardized_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


def conv_block(in_channels, out_channels, pool=False):
    layers = [
        # nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        WSConv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(num_groups=16, num_channels=out_channels),  # 使用 GroupNorm
        nn.ReLU()
    ]
    if pool:
        layers.append(nn.AvgPool2d(2))
    return nn.Sequential(*layers)


class CifarResNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CifarResNet, self).__init__()
        base_channels = 16

        self.conv1 = conv_block(in_channels, base_channels)
        self.conv2 = conv_block(base_channels, base_channels * 2, pool=True)
        self.res1 = nn.Sequential(
            conv_block(base_channels * 2, base_channels * 2),
            conv_block(base_channels * 2, base_channels * 2)
        )

        self.conv3 = conv_block(base_channels * 2, base_channels * 4, pool=True)
        self.conv4 = conv_block(base_channels * 4, base_channels * 8, pool=True)

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


class WideResidualBlock(nn.Module):
    """
    Wide ResNet 基本残差块
    归一化层放在残差分支（F(x)路径）内部 [3]
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            num_groups: int = 32,
            dropout_rate: float = 0.0  # DP-SGD 通常设为 0 [4]
    ):
        super().__init__()

        # ---- 残差分支（GN 放在此处） [3] ----
        self.residual_branch = nn.Sequential(
            # 第一层：GN → Activation → WSConv
            nn.GroupNorm(num_groups=num_groups, num_channels=in_channels),
            nn.ReLU(inplace=True),
            WSConv2d(in_channels, out_channels,
                     kernel_size=3, stride=stride,
                     padding=1, bias=False),

            # Dropout（DP-SGD 中通常不使用）
            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity(),

            # 第二层：GN → Activation → WSConv
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True),
            WSConv2d(out_channels, out_channels,
                     kernel_size=3, stride=1,
                     padding=1, bias=False),
        )

        # ---- Skip Connection（恒等映射路径，保持干净）----
        self.skip_connection = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # 需要调整维度时使用 1x1 卷积
            self.skip_connection = WSConv2d(
                in_channels, out_channels,
                kernel_size=1, stride=stride,
                padding=0, bias=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F(x) + x
        return self.residual_branch(x) + self.skip_connection(x)


class WideResNet(nn.Module):
    """
    Wide ResNet with Group Normalization + Weight Standardization

    WRN-16-4 配置：
      - 深度 = 16，宽度因子 = 4
      - 每组 Block 数 N = (16 - 4) / 6 = 2
      - 通道：[16, 64, 128, 256]
      - 所有 BN 替换为 GN [3]

    在 ε=8, δ=1e-5, B=4096, K=16 下达到 78.7% 测试精度 [4]
    """

    def __init__(
            self,
            depth: int = 16,
            width_factor: int = 4,
            num_classes: int = 10,
            num_groups: int = 32,
            in_channels:int = 3,
            dropout_rate: float = 0.0
    ):
        super().__init__()

        assert (depth - 4) % 6 == 0, "深度必须满足 (depth-4) % 6 == 0"
        n_blocks = (depth - 4) // 6  # 每组的 Block 数，WRN-16-4 = 2

        # 各阶段通道数
        channels = [
            16,  # 初始卷积输出
            16 * width_factor,  # Stage 1: 64
            32 * width_factor,  # Stage 2: 128
            64 * width_factor,  # Stage 3: 256
        ]

        # ---- 初始卷积层 ----
        self.init_conv = WSConv2d(
            in_channels, channels[0],
            kernel_size=3, stride=1,
            padding=1, bias=False
        )

        # ---- 三个残差阶段 ----
        # Stage 1: 64 channels, stride=1（不降采样）
        self.stage1 = self._make_stage(
            channels[0], channels[1],
            n_blocks, stride=1,
            num_groups=num_groups,
            dropout_rate=dropout_rate
        )

        # Stage 2: 128 channels, stride=2（降采样）
        self.stage2 = self._make_stage(
            channels[1], channels[2],
            n_blocks, stride=2,
            num_groups=num_groups,
            dropout_rate=dropout_rate
        )

        # Stage 3: 256 channels, stride=2（降采样）
        self.stage3 = self._make_stage(
            channels[2], channels[3],
            n_blocks, stride=2,
            num_groups=num_groups,
            dropout_rate=dropout_rate
        )

        # ---- 输出层：GN → ReLU → 全局平均池化 → 分类头 ----
        self.output_layer = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=channels[3]),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Linear(channels[3], num_classes)

        # 权重初始化
        self._init_weights()

    def _make_stage(
            self,
            in_channels: int,
            out_channels: int,
            n_blocks: int,
            stride: int,
            num_groups: int,
            dropout_rate: float
    ) -> nn.Sequential:
        """构建一个残差阶段"""
        layers = []
        for i in range(n_blocks):
            layers.append(WideResidualBlock(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                stride=stride if i == 0 else 1,
                num_groups=num_groups,
                dropout_rate=dropout_rate
            ))
        return nn.Sequential(*layers)

    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, WSConv2d)):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.init_conv(x)  # [B, 16, 32, 32]
        out = self.stage1(out)  # [B, 64, 32, 32]
        out = self.stage2(out)  # [B, 128, 16, 16]
        out = self.stage3(out)  # [B, 256, 8, 8]
        out = self.output_layer(out)  # [B, 256, 1, 1]
        out = out.flatten(1)  # [B, 256]
        out = self.fc(out)  # [B, num_classes]
        return out


# ================================================================
# 工厂函数
# ================================================================
def wrn_16(in_channels: int = 3, num_classes: int = 10, num_groups: int = 32, width: int = 4) -> WideResNet:
    """
    WRN-16-X with GN + WS for DP-SGD
    在 ε=8, B=4096, K=16 下达到 78.7% 测试精度 [4]
    """
    return WideResNet(
        depth=16,
        width_factor=width,
        num_classes=num_classes,
        num_groups=num_groups,
        in_channels=in_channels,
        dropout_rate=0.0  # DP-SGD 中移除正则化 [4]
    )


class DistilBertClassifier(nn.Module):
    def __init__(self, num_classes):
        super(DistilBertClassifier, self).__init__()

        # 加载基础模型DistilBertModel
        self.bert = DistilBertModel.from_pretrained('./pretrained/DistilBertModel')

        hidden_size = self.bert.config.hidden_size

        lora_config = LoraConfig(
            inference_mode=False,
            r=8,  # LoRA 秩，通常为 4, 8, 16
            lora_alpha=16,  # LoRA 缩放因子
            lora_dropout=0,  # Dropout 概率
            target_modules=["q_lin", "v_lin"]  # 必须匹配 DistilBERT 的层名称
        )
        self.bert = get_peft_model(self.bert, lora_config)

        # self.pre_classifier = nn.Linear(hidden_size, hidden_size)
        # self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, data):
        input_ids, attention_mask = data['input_ids'], data['attention_mask']
        # outputs.last_hidden_state 维度: (batch_size, sequence_length, hidden_size)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # 提取 [CLS] token (序列的第 0 个 token) 作为整个句子的表示
        pooled_output = outputs[0][:, 0]
        # pooled_output = outputs.last_hidden_state[:, 0]

        # 传递给分类头
        # pooled_output = self.pre_classifier(pooled_output)
        # pooled_output = self.relu(pooled_output)
        logits = self.fc(pooled_output)

        return logits


class RobertaClassifier(nn.Module):
    def __init__(self, num_classes):
        super(RobertaClassifier, self).__init__()

        # 1. 加载基础模型
        # 使用 roberta-large，参数量约为 350M
        # self.roberta = RobertaModel.from_pretrained('roberta-large')
        self.bert = RobertaModel.from_pretrained('./pretrained/RobertaModel')

        hidden_size = self.bert.config.hidden_size

        # 2. 构建分类头
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, data):
        input_ids, attention_mask = data['input_ids'], data['attention_mask']

        # 3. 获取模型输出
        # outputs.last_hidden_state 维度: (batch_size, sequence_length, hidden_size)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # 4. 提取特征向量
        # RoBERTa 通常使用序列的最后一个 token 作为句子表示 (类似于 CLS)
        # 注意：DistilBERT 使用 [:, 0]，而 RoBERTa 通常使用 [:, -1]
        pooled_output = outputs.last_hidden_state[:, -1]

        # 5. 传递给分类头
        logits = self.fc(pooled_output)

        return logits


if __name__ == '__main__':
    # model = DistilBertClassifier(num_classes=10)
    # print('DistilBert total params: {}'.format(sum(p.numel() for p in model.parameters())))

    model = RobertaClassifier(num_classes=10)
    print('Roberta total params: {}'.format(sum(p.numel() for p in model.parameters())))

    # model0 = wrn_16(in_channels=3, num_classes=10, num_groups=8)
    # print('wrn_16 total params: {}'.format(sum(p.numel() for p in model0.parameters())))

    # model1 = CifarResNet(in_channels=3, num_classes=10)
    # print('model1 total params: {}'.format(sum(p.numel() for p in model1.parameters())))

    # model2 = FashionMnistCnn(in_channels=1, num_classes=10)
    # print(model2)
    # print('model2 total params: {}'.format(sum(p.numel() for p in model2.parameters())))

    # model3 = MnistCnn(in_channels=1, num_classes=10)
    # print(model3)
    # print('model3 total params: {}'.format(sum(p.numel() for p in model3.parameters())))
