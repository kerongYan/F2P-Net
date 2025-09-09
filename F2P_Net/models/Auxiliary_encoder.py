import torch.nn as nn


class Auxiliary_encoder_base_Block(nn.Module):
    exansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Auxiliary_encoder_base_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels*self.exansion)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out
    
class Auxiliary_encoder_bottleneck_Block(nn.Module):
    exansion = 4
    def __init__(self,in_channels, out_channels, stride=1, downsample=None):
        super(Auxiliary_encoder_bottleneck_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.exansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*self.exansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
   
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out
    
class Auxiliary_encoder(nn.Module):
    def __init__(self, block, layers):
        '''
        block:For the corresponding network architecture, ResNet18 and ResNet34 utilise the ResNet_base_Block, whilst ResNet50 and above employ the ResNet_bottleneck_Block.
        layers:Number of residual blocks contained in each layer, for example, ResNet18 and ResNet34 are [2,2,2,2], ResNet34 is [3,4,6,3], ResNet50 is [3,4,6,3], ResNet101 is [3,4,23,3], ResNet152 is [3,8,36,3]
        num_classes:Number of classes
        include_top:Whether to include the fully connected layer

        '''
        super(Auxiliary_encoder, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Create four layers for the residual network, each layer comprising multiple residual blocks.
        self.layer1 = self._make_layer(block, 64, layers[0]) 
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Initialize parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        '''
        For the corresponding network architecture, ResNet18 and ResNet34 utilise the ResNet_base_Block, whilst ResNet50 and above employ the ResNet_bottleneck_Block.
        out_channels:Number of output channels
        blocks:Number of residual blocks contained in each layer
        '''

        downsample = None
        if stride != 1 or self.in_channels != out_channels*block.exansion:
            '''
            For the construction of Layer 1, using ResNet18 and ResNet34 does not satisfy this condition, as the first block has 64 input channels, 64 output channels, and a stride of 1.
            For Layer 1 using ResNet50 and ResNet101, the first block has 64 input channels, 256 output channels, and a stride of 1, satisfying this condition. Downsampling is required to ensure consistent channel dimensions for the residual connection.

            '''
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels*block.exansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*block.exansion)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels*block.exansion 
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Through a series of residual connections across four layers: layer1, layer2, layer3, layer4
        x = self.layer1(x)
        f1 = x
        x = self.layer2(x)
        f2 = x
        x = self.layer3(x)
        f3 = x
        x = self.layer4(x)
        f4 = x

        return [f1, f2, f3, f4]