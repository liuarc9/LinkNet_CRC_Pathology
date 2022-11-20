import torch
import torch.nn as nn
from nets.resnet import Resnet

class conv_block_1(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_block_3(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_block_5(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_5, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_block_7(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_7, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_3_1(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_3_1, self).__init__()

        #self.conv_1 = conv_block_1(ch_in, ch_out)
        #self.conv_2 = conv_block_2(ch_in, ch_out)
        self.conv_3 = conv_block_3(ch_in, ch_out)
        #self.conv_5 = conv_block_5(ch_in, ch_out)
        self.conv_7 = conv_block_7(ch_in, ch_out)
        #self.conv_9 = conv_block_9(ch_in, ch_out)

        self.conv = nn.Conv2d(ch_out * 2, ch_out, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        #x1 = self.conv_1(x)
        #x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        #x5 = self.conv_5(x)
        x7 = self.conv_7(x)
        #x9 = self.conv_9(x)

        x = torch.cat((x3, x7), dim=1)
        x = self.conv(x)

        return x

class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

class Unet(nn.Module):
    def __init__(self, num_classes=2, in_channels=3, pretrained=False):
        super(Unet, self).__init__()
        self.Resnet = Resnet(pretrained=pretrained,in_channels=in_channels)
        in_filters = [192, 384, 768, 1024]
        out_filters = [64, 128, 256, 512]
        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        # final conv (without any concat)
        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        init_conv = nn.Sequential(Resnet.conv1, Resnet.bn1, Resnet.relu, Resnet.maxpool)

    def forward(self, inputs):
        feat1 = self.init_conv(inputs)
        feat2 = self.Resnet.layer1(feat1)
        feat3 = self.Resnet.layer2(feat2)
        feat4 = self.Resnet.layer3(feat3)
        feat5 = self.Resnet.layer4(feat4)

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        final = self.final(up1)
        
        return final

    def _initialize_weights(self, *stages):
        for modules in stages:
            for module in modules.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()

