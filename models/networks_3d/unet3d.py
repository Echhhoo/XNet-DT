import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
# from models.utils.networks_other import init_weights
# from models.utils.unet_utils import UnetConv3, UnetUp3, UnetUp3_CT
from thop import profile

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, init_features=64):
        """
        Implementations based on the Unet3D paper: https://arxiv.org/abs/1606.06650
        """

        super(UNet3D, self).__init__()

        features = init_features
        self.encoder1 = UNet3D._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = UNet3D._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = UNet3D._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = UNet3D._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = UNet3D._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose3d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet3D._block((features * 8) * 2 , features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet3D._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet3D._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet3D._block(features * 2, features, name="dec1")

        self.conv = nn.Conv3d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        outputs = self.conv(dec1)
        return outputs

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm3d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm3d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True))
                ]
            )
        )


class UNet3D_min(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, init_features=32):
        """
        Implementations based on the Unet3D paper: https://arxiv.org/abs/1606.06650
        """

        super(UNet3D_min, self).__init__()

        features = init_features
        self.encoder1 = UNet3D._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = UNet3D._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = UNet3D._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = UNet3D._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = UNet3D._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose3d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet3D._block((features * 8) * 2 , features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet3D._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet3D._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet3D._block(features * 2, features, name="dec1")

        self.conv = nn.Conv3d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        outputs = self.conv(dec1)
        return outputs

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm3d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm3d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
        
        

# class unet_3D_ssl(nn.Module):

#     def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=1, is_batchnorm=True):
#         super(unet_3D_ssl, self).__init__()
#         self.is_deconv = is_deconv
#         self.in_channels = in_channels
#         self.is_batchnorm = is_batchnorm
#         self.feature_scale = feature_scale

#         filters = [64, 128, 256, 512, 1024]
#         filters = [int(x / self.feature_scale) for x in filters]

#         # downsampling
#         self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm, kernel_size=(
#             3, 3, 3), padding_size=(1, 1, 1))
#         self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

#         self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm, kernel_size=(
#             3, 3, 3), padding_size=(1, 1, 1))
#         self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

#         self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm, kernel_size=(
#             3, 3, 3), padding_size=(1, 1, 1))
#         self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

#         self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm, kernel_size=(
#             3, 3, 3), padding_size=(1, 1, 1))
#         self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

#         self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm, kernel_size=(
#             3, 3, 3), padding_size=(1, 1, 1))

#         # upsampling
#         self.up_concat4 = UnetUp3_CT(filters[4], filters[3], is_batchnorm)
#         self.up_concat3 = UnetUp3_CT(filters[3], filters[2], is_batchnorm)
#         self.up_concat2 = UnetUp3_CT(filters[2], filters[1], is_batchnorm)
#         self.up_concat1 = UnetUp3_CT(filters[1], filters[0], is_batchnorm)

#         # final conv (without any concat)
#         self.final = nn.Conv3d(filters[0], n_classes, 1)

#         self.dropout1 = nn.Dropout(p=0.3)
#         self.dropout2 = nn.Dropout(p=0.3)

#         # initialise weights
#         for m in self.modules():
#             if isinstance(m, nn.Conv3d):
#                 init_weights(m, init_type='kaiming')
#             elif isinstance(m, nn.BatchNorm3d):
#                 init_weights(m, init_type='kaiming')

#     def forward(self, inputs):
#         conv1 = self.conv1(inputs)
#         maxpool1 = self.maxpool1(conv1)

#         conv2 = self.conv2(maxpool1)
#         maxpool2 = self.maxpool2(conv2)

#         conv3 = self.conv3(maxpool2)
#         maxpool3 = self.maxpool3(conv3)

#         conv4 = self.conv4(maxpool3)
#         maxpool4 = self.maxpool4(conv4)

#         center = self.center(maxpool4)
#         center = self.dropout1(center)
#         up4 = self.up_concat4(conv4, center)
#         up3 = self.up_concat3(conv3, up4)
#         up2 = self.up_concat2(conv2, up3)
#         up1 = self.up_concat1(conv1, up2)
#         up1 = self.dropout2(up1)

#         final = self.final(up1)

#         return final

#     @staticmethod
#     def apply_argmax_softmax(pred):
#         log_p = F.softmax(pred, dim=1)

#         return log_p

def unet3d(in_channels, num_classes):
    model = UNet3D(in_channels, num_classes)
    init_weights(model, 'kaiming')
    return model

def unet3d_min(in_channels, num_classes):
    model = UNet3D_min(in_channels, num_classes)
    init_weights(model, 'kaiming')
    return model

# def unet3d_ssl(in_channels, num_classes):
#     model = unet_3D_ssl(in_channels, num_classes)
#     # init_weights(model, 'kaiming')
#     return model

# if __name__ == '__main__':
#     model = unet3d_ssl(in_channels=1, n_classes=2)
#     # model.eval()
#     # input = torch.rand(2, 1, 128, 128, 128)
#     # output = model(input)
#     # output = output.data.cpu().numpy()
#     # print(output)
#     # print(output.shape)
#     summary(model,(2, 1, 128, 128, 128), device='cpu')
#     print(model)
# if __name__ == '__main__':
#     model = unet3d(1,2)
#     model.eval()
#     input1 = torch.rand(2, 1, 128, 128, 128)
#     macs, params = profile(model, inputs=(input1,))
#     print("MACs:", macs/1000000000," GFLOP")


if __name__ == '__main__':
    # compute FLOPS & PARAMETERS
    from thop import profile
    from thop import clever_format

    model = unet3d(in_channels=1, num_classes=2)
    summary(model, (2, 1, 128, 128, 128), device='cpu')
    print(model)
    input = torch.randn(2, 1, 128, 128, 128).cuda()
    macs, params = profile(model.cuda(), (input,))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs)
    print(params)