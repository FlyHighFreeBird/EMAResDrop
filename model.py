import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.ops import DropBlock2d


# 定义EMA模块
class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super().__init__()
        self.groups = factor
        assert channels % self.groups == 0, "channels must be divisible by groups"
        self.inner_channels = channels // self.groups

        # 初始化基础网络层
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Softmax(dim=-1)

        # 空间压缩层
        self.height_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.width_pool = nn.AdaptiveAvgPool2d((1, None))

        # 特征处理层
        self.gn = nn.GroupNorm(self.inner_channels, self.inner_channels)
        self.spatial_fusion = nn.Conv2d(self.inner_channels, self.inner_channels, kernel_size=1)
        self.local_conv = nn.Conv2d(self.inner_channels, self.inner_channels, kernel_size=3, padding=1)

    def _spatial_mask(self, x):
        """生成空间注意力掩码"""
        # 高度和宽度压缩
        h = self.height_pool(x)  # [B*g, C, H, 1]
        w = self.width_pool(x).permute(0, 1, 3, 2)  # [B*g, C, W, 1]

        # 空间信息融合
        concat = torch.cat([h, w], dim=2)  # [B*g, C, H+W, 1]
        fused = self.spatial_fusion(concat)  # 融合空间信息

        # 拆分并调整维度
        h_mask, w_mask = torch.split(fused, [x.size(2), x.size(3)], dim=2)
        return h_mask.sigmoid(), w_mask.permute(0, 1, 3, 2).sigmoid()

    def _cross_attention(self, feat_a, feat_b):
        """交叉注意力计算"""
        # 全局特征压缩
        global_a = self.agp(feat_a)  # [B*g, C, 1, 1]
        global_a = global_a.view(feat_a.size(0), -1, 1)  # [B*g, C, 1]

        # 注意力权重计算
        attn_weights = self.softmax(global_a.permute(0, 2, 1))  # [B*g, 1, C]
        spatial_features = feat_b.view(feat_b.size(0), self.inner_channels, -1)  # [B*g, C, H*W]

        # 特征融合
        return torch.matmul(attn_weights, spatial_features)  # [B*g, 1, H*W]

    def forward(self, x):
        # 通道分组与空间注意力
        b, c, h, w = x.shape
        grouped_x = x.view(b * self.groups, self.inner_channels, h, w)

        # 空间注意力调制
        h_mask, w_mask = self._spatial_mask(grouped_x)
        modulated_x = self.gn(grouped_x * h_mask * w_mask)

        # 局部特征提取
        local_feat = self.local_conv(grouped_x)

        # 双路交叉注意力
        attn1 = self._cross_attention(modulated_x, local_feat)
        attn2 = self._cross_attention(local_feat, modulated_x)

        # 注意力融合
        combined_attn = (attn1 + attn2).view(b * self.groups, 1, h, w)
        return (grouped_x * combined_attn.sigmoid()).view(b, c, h, w)


class EMAResDrop(nn.Module):
    def __init__(self, block_size=7, keep_prob=0.9, num_classes=5, dropout_prob=0.5):
        super(ResNet50WithDropBlockAndEMA, self).__init__()

        self.block_size = block_size
        self.keep_prob = keep_prob

        # 加载ResNet50
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # 在layer4和fc之间插入EMA模块
        self.ema = EMA(channels=2048)  # layer4输出的通道数为2048

        # 替换fc层为num_classes类分类任务
        self.fc = nn.Sequential(
            nn.Linear(2048, num_classes)
        )

        # 冻结layer1, layer2和全连接层
        for param in self.resnet50.parameters():
            param.requires_grad = False

        # 只解冻layer4及其后的层
        for param in self.resnet50.layer4.parameters():
            param.requires_grad = True
        for param in self.resnet50.fc.parameters():
            param.requires_grad = True
        for param in self.resnet50.layer3.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)

        x = self.resnet50.layer1(x)
        x = self.resnet50.layer2(x)
        x = self.resnet50.layer3(x)
        x = DropBlock2d(block_size=self.block_size, p=1 - self.keep_prob)(x)
        x = F.relu(x)
        x = self.resnet50.layer4(x)
        features = x
        x = DropBlock2d(block_size=self.block_size, p=1 - self.keep_prob)(x)
        x = F.relu(x)

        # 将EMA模块应用于layer4的输出
        x = self.ema(x)

        # 全局平均池化
        x = self.resnet50.avgpool(x)
        x = torch.flatten(x, 1)

        # 经过全连接层（fc）
        x = self.fc(x)  # [batch_size, 2048, 1, 1]
        return x


if __name__ == "__main__":
    # 创建模型实例（五分类任务）
    model = EMAResDrop(num_classes=5)

    # 测试模型
    input_tensor = torch.randn(1, 3, 224, 224)
    output, _ = model(input_tensor)
    print(output.shape)
