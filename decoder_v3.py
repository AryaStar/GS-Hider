import torch.nn as nn
from torch.nn import functional as F

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super(ChannelAttention, self).__init__()

        # 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 全局最大池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        mid_channels = max(1, channels // reduction)

        # 共享MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(mid_channels, channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x : (B,C,H,W)

        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))

        out = avg_out + max_out
        attention = self.sigmoid(out)

        return x * attention

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # 1. 初始特征提取 (16 -> 64)
        self.conv1 = nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 2. 中间特征处理模块 A (包含残差与注意力)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.ca1 = ChannelAttention(channels=64, reduction=4)
        
        # 3. 中间特征处理模块 B (包含残差与注意力)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.ca2 = ChannelAttention(channels=64, reduction=4)
        
        # 4. 最终解码到 RGB (64 -> 3)
        self.conv5 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)  # (C,H,W) -> (1,C,H,W)
        # 初始提取
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.2)
        
        # 模块 A: 提取 -> 注意力加权 -> 残差连接
        residual = x
        x1 = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        x1 = self.ca1(x1)  # 在高维特征上做注意力
        # x = x1 + residual  # 引入残差
        x = x1
        
        # 模块 B: 提取 -> 提取 -> 注意力加权 -> 残差连接
        # residual = x
        x2 = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)
        x2 = F.leaky_relu(self.bn4(self.conv4(x2)), negative_slope=0.2)
        x2 = self.ca2(x2)
        # x = x2 + residual
        x = x2
        
        # 输出层
        x = self.conv5(x)

        if x.shape[0] == 1:
            x = x.squeeze(0)
        
        return x
    
class WatermarkCNN(nn.Module):
    def __init__(self):
        super(WatermarkCNN, self).__init__()

        # 1. 初始特征提取 (16 -> 64)
        self.conv1 = nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 2. 中间特征处理模块 A (包含残差与注意力)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.ca1 = ChannelAttention(channels=64, reduction=4)
        
        # 3. 中间特征处理模块 B (包含残差与注意力)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.ca2 = ChannelAttention(channels=64, reduction=4)
        
        # 4. 最终解码到 RGB (64 -> 3)
        self.conv5 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)  # (C,H,W) -> (1,C,H,W)
        # 初始提取
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.2)
        
        # 模块 A: 提取 -> 注意力加权 -> 残差连接
        residual = x
        x1 = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        x1 = self.ca1(x1)  # 在高维特征上做注意力
        # x = x1 + residual  # 引入残差
        x = x1
        
        # 模块 B: 提取 -> 提取 -> 注意力加权 -> 残差连接
        # residual = x
        x2 = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)
        x2 = F.leaky_relu(self.bn4(self.conv4(x2)), negative_slope=0.2)
        x2 = self.ca2(x2)
        # x = x2 + residual # 残差
        x = x2
        
        # 输出层
        x = self.conv5(x)

        if x.shape[0] == 1:
            x = x.squeeze(0)
        
        return x


def count_parameters(model):
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size_in_MB = param_count * 4 / (1024 ** 2) 
    return param_size_in_MB

if __name__ == "__main__":
    model = SimpleCNN()
    print(f"Total parameters: {count_parameters(model)}")