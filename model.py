import torch, torch.nn as nn

class DarkNet53(nn.Module):
    def __init__(self, n_cls=1, in_c=3):
        super().__init__()
        self.n_cls = n_cls
        self.in_c = in_c
        
        self.conv1 = conv(self.in_c, out_c=32, kernel=3, stride=1, pad=1)
        self.conv2 = conv(in_c=32, out_c=64, kernel=3, stride=2, pad=1)
        self.residual_block1 = residual(in_c=64, iter=1)
        
        self.conv3 = conv(in_c=64, out_c=128, kernel=3, stride=2, pad=1)
        self.residual_block2 = residual(in_c=128, iter=2)
        
        self.conv4 = conv(in_c=128, out_c=256, kernel=3, stride=2, pad=1)
        self.residual_block3 = residual(in_c=256, iter=8)
        
        self.conv5 = conv(in_c=256, out_c=512, kernel=3, stride=2, pad=1)
        self.residual_block4 = residual(in_c=512, iter=8)
        
        self.conv6 = conv(in_c=512, out_c=1024, kernel=3, stride=2, pad=1)
        self.residual_block5 = residual(in_c=1024, iter=4)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, self.n_cls)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.residual_block1(x)
        
        x = self.conv3(x)
        x = self.residual_block2(x)
        
        x = self.conv4(x)
        x = self.residual_block3(x)
        
        x = self.conv5(x)
        x = self.residual_block4(x)
        
        x = self.conv6(x)
        x = self.residual_block5(x)
        
        x = self.avgpool(x)
        x = x.view(-1,1024)
        x = self.fc(x)
        x = self.sigmoid(x)
        
        return x
        
class ResidualBlock(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        mid_c = in_c // 2
        self.layer1 = conv(in_c=in_c, out_c=mid_c, kernel=1, stride=1, pad=0)
        self.layer2 = conv(in_c=mid_c, out_c=in_c, kernel=3, stride=1, pad=1)
        
    def forward(self, x):
        res = x
        x = self.layer1(x)
        x = self.layer2(x)
        x += res
        
        return x
        

def conv(in_c, out_c, kernel, stride, pad):
    conv = nn.Conv2d(in_c, out_c, kernel, stride, pad)
    bn = nn.BatchNorm2d(out_c)
    act = nn.LeakyReLU(0.1)
    
    return nn.Sequential(*[conv, bn, act])


def residual(in_c, iter):
    layers = []
    for _ in range(iter):
        layers.append(ResidualBlock(in_c))
    
    return nn.Sequential(*layers)
