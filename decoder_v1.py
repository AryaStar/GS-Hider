import torch.nn as nn
from torch.nn import functional as F

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super(ChannelAttention, self).__init__()

        # 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 全局最大池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 共享MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
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
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1)

        # # 加入通道注意力
        self.ca = ChannelAttention(channels=64)
        # self.ca2 = ChannelAttention(channels=64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):

        x = F.relu(self.conv1(x))

        # 通道注意力
        x = self.ca(x)

        x = F.relu(self.conv2(x))

        x = F.relu(self.conv3(x))

        x = F.relu(self.conv4(x))

        # x = self.ca2(x)

        x = self.conv5(x)

        return x
    
class WatermarkCNN(nn.Module):
    def __init__(self):
        super(WatermarkCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1)

        # # 加入通道注意力
        self.ca = ChannelAttention(channels=64)
        # self.ca2 = ChannelAttention(channels=64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):

        x = F.relu(self.conv1(x))

        # # 通道注意力
        x = self.ca(x)
        
        x = F.relu(self.conv2(x))

        x = F.relu(self.conv3(x))

        x = F.relu(self.conv4(x))

        # x = self.ca2(x)

        x = self.conv5(x)

        return x


def count_parameters(model):
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size_in_MB = param_count * 4 / (1024 ** 2) 
    return param_size_in_MB

if __name__ == "__main__":
    model = SimpleCNN()
    print(f"Total parameters: {count_parameters(model)}")