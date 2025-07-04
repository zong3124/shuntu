import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.wtconv.wtconv2d import PyramidalWTConv  # 金字塔小波模块

from networks.res2net import res2net50
nonlinearity = nn.ReLU(inplace=True)
DEVICE =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
from networks.zong.Base_Weve.FAM import FAM #频率调节


nonlinearity = nn.ReLU(inplace=True)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.relu2(self.norm2(self.deconv2(x)))
        x = self.relu3(self.norm3(self.conv3(x)))
        return x


class PWFNet(nn.Module):
    def __init__(self, num_classes=1):
        super(PWFNet, self).__init__()
        filters = [256, 512, 1024, 2048]
        resnet = res2net50(pretrained=True)

        self.fbm2 = FAM(
            in_channels=filters[1], k_list=[2, 4, 8]
        )
        self.fbm3 = FAM(
            in_channels=filters[2], k_list=[2, 4, 8]
        )
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.PWC1 = PyramidalWTConv(filters[1], filters[1])
        self.PWC2 = PyramidalWTConv(filters[2], filters[2])
        self.PWC3 = PyramidalWTConv(filters[3], filters[3])



        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, stride=2, padding=1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        # 中层：PWC + FAM
        e2 = self.encoder2(e1)
        e2 = self.PWC1(e2)
        e2 = self.fam2(e2)


        # 高层：PWC + FAM
        e3 = self.encoder3(e2)
        e3 = self.PWC2(e3)
        e3 = self.fam3(e3)


        # 顶层：仅PWC
        e4 = self.encoder4(e3)
        e4 = self.PWC3(e4)

        # 解码器
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return torch.sigmoid(out)

if __name__ == "__main__":
    import torch
    model = PWFNet().to(DEVICE)
    input = torch.rand(4, 3, 512, 512).to(DEVICE)
    # test_tensor_A = torch.rand((1, 3, 512, 512)).to(DEVICE)
    y = model(input)
    # print(y.size())
    import thop
    flops, params = thop.profile(model, inputs=(input,))
    # 打印结果
    print('FLOPs: %.2f G' % (flops / (1000 ** 3)))  # FLOPs 转换为 GFLOPs
    print('Params: %.2f M' % (params / (1000) ** 2))
    print('Params: %.2f' % (params))
