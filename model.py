import torch
from torch import nn

class CBL(nn.Module):
    def __init__(self, conv_dim1, conv_dim2, kernel_size, stride=1):
        super(CBL, self).__init__()

        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(conv_dim1, conv_dim2, kernel_size=kernel_size,
                              stride=stride, padding=pad, bias=False)
        self.batch_norm = nn.BatchNorm2d(conv_dim2)
        self.leaky_relu = nn.LeakyReLU(inplace=False)

    def forward(self, x):
        x = self.leaky_relu(self.batch_norm(self.conv(x)))
        return x


class ResBlock(nn.Module):
    def __init__(self, conv_dim):
        super(ResBlock, self).__init__()

        self.cbl1 = CBL(conv_dim1=conv_dim, conv_dim2=conv_dim, kernel_size=1)
        self.cbl3 = CBL(conv_dim1=conv_dim, conv_dim2=conv_dim, kernel_size=3)
        self.norm = nn.BatchNorm2d(conv_dim)

    def forward(self, x):
        residual = x
        x = self.norm(self.cbl3(self.cbl1(x)))
        x += residual
        return x



class ChannelAttn(nn.Module):
    def __init__(self, conv_dim, k_size=3):
        super(ChannelAttn, self).__init__()
        """ 
        Instead of using the standard channel attention for CBAM, an ECA was 
        used from this paper:    
            https://arxiv.org/pdf/1910.03151.pdf

        This is then integrated into the CBAM module to create ECA-CBAM.
        ECA-CBAM outperformed the other five attention mechanisms such as 
        BAM, CBAM, ECA, CA, and SeNet and the cross combination of these models
        in this paper:
            https://dl.acm.org/doi/fullHtml/10.1145/3529466.3529468
        """

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        x = self.sigmoid(x)
        return x


class SpatialAttn(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttn, self).__init__()

        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = torch.mean(x, dim=1, keepdim=True)
        x2, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([x1, x2], dim=1)
        x = self.sigmoid(self.conv(x))
        return x


class CBAM(nn.Module):
    def __init__(self, conv_dim):
        super(CBAM, self).__init__()

        self.channel_attn = ChannelAttn(conv_dim)
        self.spatial_attn = SpatialAttn()

    def forward(self, x):
        ca = self.channel_attn(x)
        x *= ca
        sa = self.spatial_attn(x)
        x *= sa
        return x

# just the convolutional layers of vgg19
class VGG19_backbone(nn.Module):
    def __init__(self):
        super(VGG19_backbone, self).__init__()

        self.layers1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )

        self.layers2 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )

        self.layers3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )

    def forward(self, x):
        x1 = self.layers1(x)
        x2 = self.layers2(x1)
        x3 = self.layers3(x2)

        return [x1, x2, x3]


class PANet(nn.Module):
    def __init__(self, num_classes=20):
        super(PANet, self).__init__()

        self.num_classes = num_classes
        res_dim = 3

        res_layers = []
        for i in range(res_dim):
            res_layers.append(ResBlock(conv_dim=640))

        self.res_4 = nn.Sequential(*res_layers)

        self.upsample = nn.Upsample(scale_factor=2)

        self.layers1 = nn.Sequential(CBL(conv_dim1=512, conv_dim2=384, kernel_size=1),
                                     nn.Upsample(scale_factor=2))

        self.layers2 = nn.Sequential(*res_layers, CBAM(conv_dim=640))

        self.layers3 = nn.Sequential(CBL(conv_dim1=640, conv_dim2=456, kernel_size=1),
                                     nn.Upsample(scale_factor=2))
        res_layers = []
        for i in range(res_dim):
            res_layers.append(ResBlock(conv_dim=584))

        self.layers4 = nn.Sequential(*res_layers, CBAM(conv_dim=584))

        self.layers5 = nn.Sequential(CBL(conv_dim1=584, conv_dim2=512, kernel_size=3, stride=2))

        res_layers = []
        for i in range(res_dim):
            res_layers.append(ResBlock(conv_dim=1152))

        self.layers6 = nn.Sequential(*res_layers, CBAM(conv_dim=1152))

        self.layers7 = nn.Sequential(CBL(conv_dim1=1152, conv_dim2=1024, kernel_size=3, stride=2))

        res_layers = []
        for i in range(res_dim):
            res_layers.append(ResBlock(conv_dim=1536))

        self.layers8 = nn.Sequential(*res_layers, CBAM(conv_dim=1536))

        ## yolo heads
        self.yolo_head1 = nn.Sequential(CBL(conv_dim1=584, conv_dim2=416, kernel_size=3),
                                        nn.Conv2d(416, 75, 1))
        self.yolo_head2 = nn.Sequential(CBL(conv_dim1=1152, conv_dim2=896, kernel_size=3),
                                        nn.Conv2d(896, 75, 1))
        self.yolo_head3 = nn.Sequential(CBL(conv_dim1=1536, conv_dim2=1024, kernel_size=3),
                                        nn.Conv2d(1024, 75, 1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inp):
        x1, x2, x3 = inp
        x = self.layers1(x3)
        x = torch.cat([x, x2], dim=1)
        x_2 = self.layers2(x)

        x = self.layers3(x_2)
        x = torch.cat([x1, x], dim=1)
        x = self.layers4(x)
        x1_out = self.yolo_head1(x).reshape(x.shape[0], 3, self.num_classes + 5,
                                            x.shape[2], x.shape[3]).permute(0, 1,3, 4, 2)

        x = self.layers5(x)
        x = torch.cat([x, x_2], dim=1)
        x = self.layers6(x)
        x2_out = self.yolo_head2(x).reshape(x.shape[0], 3, self.num_classes + 5,
                                            x.shape[2], x.shape[3]).permute(0, 1, 3, 4, 2)

        x = self.layers7(x)
        x = torch.cat([x, x3], dim=1)
        x = self.layers8(x)
        x3_out = self.yolo_head3(x).reshape(x.shape[0], 3, self.num_classes + 5,
                                            x.shape[2], x.shape[3]).permute(0, 1,3, 4, 2)

        return [x1_out, x2_out, x3_out]


class AS_YOLO(nn.Module):
    def __init__(self, backbone=VGG19_backbone(), pannet=PANet(), pretrained=False, PATH="/kaggle/working/backbone.pt"):
        super(AS_YOLO, self).__init__()

        self.backbone = backbone
        self.pannet = pannet

        if pretrained:
            self.backbone.load_state_dict(torch.load(PATH))
            for param in self.backbone.parameters():
                param.grad = None

    def forward(self, x):
        out = self.backbone(x)
        out = self.pannet(out)

        return out


if __name__ == "__main__":
    pre_train = True
    model = AS_YOLO(pretrained=pre_train)

    # to make sure the backbone convolutional layers to not train
    if pre_train:
        for name, param in model.named_parameters():
            if name.split(".")[0] == 'backbone':
                assert param.requires_grad == False
            else:
                assert param.requires_grad == True

    # to make sure the outputs are correct
    img_dim = 416
    num_classes = 20
    batch_size = 4
    x = torch.randn((batch_size, 3, img_dim, img_dim))
    out = model(x)
    assert out[2].shape == (batch_size, 3, img_dim // 32, img_dim // 32, num_classes + 5), out[2].shape
    assert out[1].shape == (batch_size, 3, img_dim // 16, img_dim // 16, num_classes + 5), out[1].shape
    assert out[0].shape == (batch_size, 3, img_dim // 8, img_dim // 8, num_classes + 5), out[0].shape
    print("model works successfully!")
