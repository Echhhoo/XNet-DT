import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn import init
from models.utils.dca import DCA


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



class XNetv3_3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, init_features=32):
        """
        Implementations based on the Unet3D paper: https://arxiv.org/abs/1606.06650
        """

        super(XNetv3_3D, self).__init__()

        features = init_features

        # main network
        self.M_encoder1 = XNetv3_3D._block(in_channels, features, name="enc1")
        self.M_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.M_encoder2 = XNetv3_3D._block(features, features * 2, name="enc2")
        self.M_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.M_encoder3 = XNetv3_3D._block(features * 2, features * 4, name="enc3")
        self.M_pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.M_encoder4 = XNetv3_3D._block(features * 4, features * 8, name="enc4")
        self.M_pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.M_bottleneck = XNetv3_3D._block(features * 8, features * 16, name="bottleneck")

        self.M_upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.M_decoder4 = XNetv3_3D._block((features * 8) * 2 , features * 8, name="dec4")
        self.M_upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.M_decoder3 = XNetv3_3D._block((features * 4) * 2, features * 4, name="dec3")
        self.M_upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.M_decoder2 = XNetv3_3D._block((features * 2) * 2, features * 2, name="dec2")
        self.M_upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.M_decoder1 = XNetv3_3D._block(features * 2, features, name="dec1")

        self.M_conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

        # L network
        self.L_encoder1 = XNetv3_3D._block(in_channels*2, features, name="enc1")
        self.L_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.L_encoder2 = XNetv3_3D._block(features, features * 2, name="enc2")
        self.L_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.L_encoder3 = XNetv3_3D._block(features * 2, features * 4, name="enc3")
        self.L_pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.L_encoder4 = XNetv3_3D._block(features * 4, features * 8, name="enc4")
        self.L_pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.L_bottleneck = XNetv3_3D._block(features * 8, features * 16, name="bottleneck")

        self.L_upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.L_decoder4 = XNetv3_3D._block((features * 8) * 2 , features * 8, name="dec4")
        self.L_upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.L_decoder3 = XNetv3_3D._block((features * 4) * 2, features * 4, name="dec3")
        self.L_upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.L_decoder2 = XNetv3_3D._block((features * 2)* 2, features * 2, name="dec2")
        self.L_upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.L_decoder1 = XNetv3_3D._block(features* 2, features, name="dec1")
        self.L_conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

        # H network
        self.H_encoder1 = XNetv3_3D._block(in_channels*2, features, name="enc1")
        self.H_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.H_encoder2 = XNetv3_3D._block(features, features * 2, name="enc2")
        self.H_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.H_encoder3 = XNetv3_3D._block(features * 2, features * 4, name="enc3")
        self.H_pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.H_encoder4 = XNetv3_3D._block(features * 4, features * 8, name="enc4")
        self.H_pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.H_bottleneck = XNetv3_3D._block(features * 8, features * 16, name="bottleneck")

        self.H_upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.H_decoder4 = XNetv3_3D._block((features * 8)* 2, features * 8, name="dec4")
        self.H_upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.H_decoder3 = XNetv3_3D._block((features * 4)* 2, features * 4, name="dec3")
        self.H_upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.H_decoder2 = XNetv3_3D._block((features * 2) * 2, features * 2, name="dec2")
        self.H_upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.H_decoder1 = XNetv3_3D._block(features * 2, features, name="dec1")
        self.H_conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

        # fusion
        self.M_H_conv1 = XNetv3_3D._block(features * 2, features, name="fusion1")
        self.M_H_conv2 = XNetv3_3D._block(features * 4, features * 2, name="fusion2")
        self.M_L_conv3 = XNetv3_3D._block(features * 8, features * 4, name="fusion3")
        self.M_L_conv4 = XNetv3_3D._block(features * 16, features * 8, name="fusion4")
        
        # # 假设这是初始化参数，实际应用中可能需要根据具体情况调整初始化方式
        # self.lh_exchange_weight_L = nn.Parameter(torch.randn(1))
        # self.lh_exchange_weight_H = nn.Parameter(torch.randn(1))
        # # Sigmoid层用于限制参数范围
        # self.sigmoid_L = nn.Sigmoid()
        # self.sigmoid_H = nn.Sigmoid()

    def forward(self, x_main, x_L, x_HA,x_HR):
        x_L_new=torch.cat((x_L,  x_HA), dim=1)
        x_H_new=torch.cat((x_L,  x_HR), dim=1)
        
        # lh_exchange_weight_L = self.sigmoid_L(self.lh_exchange_weight_L)
        # lh_exchange_weight_H = self.sigmoid_H(self.lh_exchange_weight_H)
        # x_L_new =(1-lh_exchange_weight_L)*x_L + lh_exchange_weight_L * x_HA
        # x_H_new =(1-lh_exchange_weight_H)*x_L + lh_exchange_weight_H * x_HR
        
        
        # x_L_new=0.5*x_L+0.5*x_HA
        # x_H_new=0.5*x_L+0.5*x_HR
        # Main encoder
        M_enc1 = self.M_encoder1(x_main)
        M_enc2 = self.M_encoder2(self.M_pool1(M_enc1))
        M_enc3 = self.M_encoder3(self.M_pool2(M_enc2))
        M_enc4 = self.M_encoder4(self.M_pool3(M_enc3))

        M_bottleneck = self.M_bottleneck(self.M_pool4(M_enc4))

        # L encoder
        L_enc1 = self.L_encoder1(x_L_new)
        L_enc2 = self.L_encoder2(self.L_pool1(L_enc1))
        L_enc3 = self.L_encoder3(self.L_pool2(L_enc2))
        L_enc4 = self.L_encoder4(self.L_pool3(L_enc3))

        L_bottleneck = self.L_bottleneck(self.L_pool4(L_enc4))

        # H encoder
        H_enc1 = self.H_encoder1(x_H_new)
        H_enc2 = self.H_encoder2(self.H_pool1(H_enc1))
        H_enc3 = self.H_encoder3(self.H_pool2(H_enc2))
        H_enc4 = self.H_encoder4(self.H_pool3(H_enc3))

        H_bottleneck = self.H_bottleneck(self.H_pool4(H_enc4))

        # fusion
        M_H_enc1 = torch.cat((M_enc1, H_enc1), dim=1)
        M_H_enc1 = self.M_H_conv1(M_H_enc1)
        M_H_enc2 = torch.cat((M_enc2, H_enc2), dim=1)
        M_H_enc2 = self.M_H_conv2(M_H_enc2)
        M_L_enc3 = torch.cat((M_enc3, L_enc3), dim=1)
        M_L_enc3 = self.M_L_conv3(M_L_enc3)
        M_L_enc4 = torch.cat((M_enc4, L_enc4), dim=1)
        M_L_enc4 = self.M_L_conv4(M_L_enc4)

        # Main decoder
        M_dec4 = self.M_upconv4(M_bottleneck)
        M_dec4 = torch.cat((M_dec4, M_L_enc4), dim=1)
        M_dec4 = self.M_decoder4(M_dec4)
        
        M_dec3 = self.M_upconv3(M_dec4)
        M_dec3 = torch.cat((M_dec3, M_L_enc3), dim=1)
        M_dec3 = self.M_decoder3(M_dec3)
        
        M_dec2 = self.M_upconv2(M_dec3)
        M_dec2 = torch.cat((M_dec2, M_H_enc2), dim=1)
        M_dec2 = self.M_decoder2(M_dec2)
        
        M_dec1 = self.M_upconv1(M_dec2)
        M_dec1 = torch.cat((M_dec1, M_H_enc1), dim=1)
        M_dec1 = self.M_decoder1(M_dec1)
        M_outputs = self.M_conv(M_dec1)

        # L decoder
        L_dec4 = self.L_upconv4(L_bottleneck)
        L_dec4 = torch.cat((L_dec4, M_L_enc4), dim=1)
        L_dec4 = self.L_decoder4(L_dec4)
        
        L_dec3 = self.L_upconv3(L_dec4)
        L_dec3 = torch.cat((L_dec3, M_L_enc3), dim=1)
        L_dec3 = self.L_decoder3(L_dec3)
        
        L_dec2 = self.L_upconv2(L_dec3)
        L_dec2 = torch.cat((L_dec2, L_enc2), dim=1)
        L_dec2 = self.L_decoder2(L_dec2)
        
        L_dec1 = self.L_upconv1(L_dec2)
        L_dec1 = torch.cat((L_dec1, L_enc1), dim=1)
        L_dec1 = self.L_decoder1(L_dec1)
        L_outputs = self.L_conv(L_dec1)

        # H decoder
        H_dec4 = self.H_upconv4(H_bottleneck)
        H_dec4 = torch.cat((H_dec4, H_enc4), dim=1)
        H_dec4 = self.H_decoder4(H_dec4)
        
        H_dec3 = self.H_upconv3(H_dec4)
        H_dec3 = torch.cat((H_dec3, H_enc3), dim=1)
        H_dec3 = self.H_decoder3(H_dec3)
        
        H_dec2 = self.H_upconv2(H_dec3)
        H_dec2 = torch.cat((H_dec2, M_H_enc2), dim=1)
        H_dec2 = self.H_decoder2(H_dec2)
        
        H_dec1 = self.H_upconv1(H_dec2)
        H_dec1 = torch.cat((H_dec1, M_H_enc1), dim=1)
        H_dec1 = self.H_decoder1(H_dec1)
        H_outputs = self.H_conv(H_dec1)

        return M_outputs, L_outputs, H_outputs

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



class XNetv3_3D_Add(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, init_features=32):
        """
        Implementations based on the Unet3D paper: https://arxiv.org/abs/1606.06650
        """

        super(XNetv3_3D_Add, self).__init__()

        features = init_features

        # main network
        self.M_encoder1 = XNetv3_3D_Add._block(in_channels, features, name="enc1")
        self.M_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.M_encoder2 = XNetv3_3D_Add._block(features, features * 2, name="enc2")
        self.M_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.M_encoder3 = XNetv3_3D_Add._block(features * 2, features * 4, name="enc3")
        self.M_pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.M_encoder4 = XNetv3_3D_Add._block(features * 4, features * 8, name="enc4")
        self.M_pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.M_bottleneck = XNetv3_3D_Add._block(features * 8, features * 16, name="bottleneck")

        self.M_upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.M_decoder4 = XNetv3_3D_Add._block((features * 8) * 2 , features * 8, name="dec4")
        self.M_upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.M_decoder3 = XNetv3_3D_Add._block((features * 4) * 2, features * 4, name="dec3")
        self.M_upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.M_decoder2 = XNetv3_3D_Add._block((features * 2) * 2, features * 2, name="dec2")
        self.M_upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.M_decoder1 = XNetv3_3D_Add._block(features * 2, features, name="dec1")

        self.M_conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

        # L network
        self.L_encoder1 = XNetv3_3D_Add._block(in_channels, features, name="enc1")
        self.L_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.L_encoder2 = XNetv3_3D_Add._block(features, features * 2, name="enc2")
        self.L_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.L_encoder3 = XNetv3_3D_Add._block(features * 2, features * 4, name="enc3")
        self.L_pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.L_encoder4 = XNetv3_3D_Add._block(features * 4, features * 8, name="enc4")
        self.L_pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.L_bottleneck = XNetv3_3D_Add._block(features * 8, features * 16, name="bottleneck")

        self.L_upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.L_decoder4 = XNetv3_3D_Add._block((features * 8) * 2 , features * 8, name="dec4")
        self.L_upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.L_decoder3 = XNetv3_3D_Add._block((features * 4) * 2, features * 4, name="dec3")
        self.L_upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.L_decoder2 = XNetv3_3D_Add._block((features * 2)* 2, features * 2, name="dec2")
        self.L_upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.L_decoder1 = XNetv3_3D_Add._block(features* 2, features, name="dec1")
        self.L_conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

        # H network
        self.H_encoder1 = XNetv3_3D_Add._block(in_channels, features, name="enc1")
        self.H_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.H_encoder2 = XNetv3_3D_Add._block(features, features * 2, name="enc2")
        self.H_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.H_encoder3 = XNetv3_3D_Add._block(features * 2, features * 4, name="enc3")
        self.H_pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.H_encoder4 = XNetv3_3D_Add._block(features * 4, features * 8, name="enc4")
        self.H_pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.H_bottleneck = XNetv3_3D_Add._block(features * 8, features * 16, name="bottleneck")

        self.H_upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.H_decoder4 = XNetv3_3D_Add._block((features * 8)* 2, features * 8, name="dec4")
        self.H_upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.H_decoder3 = XNetv3_3D_Add._block((features * 4)* 2, features * 4, name="dec3")
        self.H_upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.H_decoder2 = XNetv3_3D_Add._block((features * 2) * 2, features * 2, name="dec2")
        self.H_upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.H_decoder1 = XNetv3_3D_Add._block(features * 2, features, name="dec1")
        self.H_conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

        # fusion
        self.M_H_conv1 = XNetv3_3D_Add._block(features * 2, features, name="fusion1")
        self.M_H_conv2 = XNetv3_3D_Add._block(features * 4, features * 2, name="fusion2")
        self.M_L_conv3 = XNetv3_3D_Add._block(features * 8, features * 4, name="fusion3")
        self.M_L_conv4 = XNetv3_3D_Add._block(features * 16, features * 8, name="fusion4")
        
        # 假设这是初始化参数，实际应用中可能需要根据具体情况调整初始化方式
        self.lh_exchange_weight_L = nn.Parameter(torch.randn(1))
        self.lh_exchange_weight_H = nn.Parameter(torch.randn(1))
        # Sigmoid层用于限制参数范围
        self.sigmoid_L = nn.Sigmoid()
        self.sigmoid_H = nn.Sigmoid()

    def forward(self, x_main, x_L, x_HA,x_HR):
        # x_L_new=torch.cat((x_L,  x_HA), dim=1)
        # x_H_new=torch.cat((x_L,  x_HR), dim=1)
        
        lh_exchange_weight_L = self.sigmoid_L(self.lh_exchange_weight_L)
        lh_exchange_weight_H = self.sigmoid_H(self.lh_exchange_weight_H)
        x_L_new =(1-lh_exchange_weight_L)*x_L + lh_exchange_weight_L * x_HA
        x_H_new =(1-lh_exchange_weight_H)*x_L + lh_exchange_weight_H * x_HR
        
        
        # x_L_new=0.5*x_L+0.5*x_HA
        # x_H_new=0.5*x_L+0.5*x_HR
        # Main encoder
        M_enc1 = self.M_encoder1(x_main)
        M_enc2 = self.M_encoder2(self.M_pool1(M_enc1))
        M_enc3 = self.M_encoder3(self.M_pool2(M_enc2))
        M_enc4 = self.M_encoder4(self.M_pool3(M_enc3))

        M_bottleneck = self.M_bottleneck(self.M_pool4(M_enc4))

        # L encoder
        L_enc1 = self.L_encoder1(x_L_new)
        L_enc2 = self.L_encoder2(self.L_pool1(L_enc1))
        L_enc3 = self.L_encoder3(self.L_pool2(L_enc2))
        L_enc4 = self.L_encoder4(self.L_pool3(L_enc3))

        L_bottleneck = self.L_bottleneck(self.L_pool4(L_enc4))

        # H encoder
        H_enc1 = self.H_encoder1(x_H_new)
        H_enc2 = self.H_encoder2(self.H_pool1(H_enc1))
        H_enc3 = self.H_encoder3(self.H_pool2(H_enc2))
        H_enc4 = self.H_encoder4(self.H_pool3(H_enc3))

        H_bottleneck = self.H_bottleneck(self.H_pool4(H_enc4))

        # fusion
        M_H_enc1 = torch.cat((M_enc1, H_enc1), dim=1)
        M_H_enc1 = self.M_H_conv1(M_H_enc1)
        M_H_enc2 = torch.cat((M_enc2, H_enc2), dim=1)
        M_H_enc2 = self.M_H_conv2(M_H_enc2)
        M_L_enc3 = torch.cat((M_enc3, L_enc3), dim=1)
        M_L_enc3 = self.M_L_conv3(M_L_enc3)
        M_L_enc4 = torch.cat((M_enc4, L_enc4), dim=1)
        M_L_enc4 = self.M_L_conv4(M_L_enc4)

        # Main decoder
        M_dec4 = self.M_upconv4(M_bottleneck)
        M_dec4 = torch.cat((M_dec4, M_L_enc4), dim=1)
        M_dec4 = self.M_decoder4(M_dec4)
        
        M_dec3 = self.M_upconv3(M_dec4)
        M_dec3 = torch.cat((M_dec3, M_L_enc3), dim=1)
        M_dec3 = self.M_decoder3(M_dec3)
        
        M_dec2 = self.M_upconv2(M_dec3)
        M_dec2 = torch.cat((M_dec2, M_H_enc2), dim=1)
        M_dec2 = self.M_decoder2(M_dec2)
        
        M_dec1 = self.M_upconv1(M_dec2)
        M_dec1 = torch.cat((M_dec1, M_H_enc1), dim=1)
        M_dec1 = self.M_decoder1(M_dec1)
        M_outputs = self.M_conv(M_dec1)

        # L decoder
        L_dec4 = self.L_upconv4(L_bottleneck)
        L_dec4 = torch.cat((L_dec4, M_L_enc4), dim=1)
        L_dec4 = self.L_decoder4(L_dec4)
        
        L_dec3 = self.L_upconv3(L_dec4)
        L_dec3 = torch.cat((L_dec3, M_L_enc3), dim=1)
        L_dec3 = self.L_decoder3(L_dec3)
        
        L_dec2 = self.L_upconv2(L_dec3)
        L_dec2 = torch.cat((L_dec2, L_enc2), dim=1)
        L_dec2 = self.L_decoder2(L_dec2)
        
        L_dec1 = self.L_upconv1(L_dec2)
        L_dec1 = torch.cat((L_dec1, L_enc1), dim=1)
        L_dec1 = self.L_decoder1(L_dec1)
        L_outputs = self.L_conv(L_dec1)

        # H decoder
        H_dec4 = self.H_upconv4(H_bottleneck)
        H_dec4 = torch.cat((H_dec4, H_enc4), dim=1)
        H_dec4 = self.H_decoder4(H_dec4)
        
        H_dec3 = self.H_upconv3(H_dec4)
        H_dec3 = torch.cat((H_dec3, H_enc3), dim=1)
        H_dec3 = self.H_decoder3(H_dec3)
        
        H_dec2 = self.H_upconv2(H_dec3)
        H_dec2 = torch.cat((H_dec2, M_H_enc2), dim=1)
        H_dec2 = self.H_decoder2(H_dec2)
        
        H_dec1 = self.H_upconv1(H_dec2)
        H_dec1 = torch.cat((H_dec1, M_H_enc1), dim=1)
        H_dec1 = self.H_decoder1(H_dec1)
        H_outputs = self.H_conv(H_dec1)

        return M_outputs, L_outputs, H_outputs

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





class XNetv3_3D_min(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, init_features=32):
        """
        Implementations based on the Unet3D paper: https://arxiv.org/abs/1606.06650
        """

        super(XNetv3_3D_min, self).__init__()

        features = init_features

        # main network
        self.M_encoder1 = XNetv3_3D_min._block(in_channels, features, name="enc1")
        self.M_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.M_encoder2 = XNetv3_3D_min._block(features, features * 2, name="enc2")
        self.M_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.M_encoder3 = XNetv3_3D_min._block(features * 2, features * 4, name="enc3")
        self.M_pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.M_encoder4 = XNetv3_3D_min._block(features * 4, features * 8, name="enc4")
        self.M_pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.M_bottleneck = XNetv3_3D_min._block(features * 8, features * 16, name="bottleneck")

        self.M_upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.M_decoder4 = XNetv3_3D_min._block((features * 8) * 2 , features * 8, name="dec4")
        self.M_upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.M_decoder3 = XNetv3_3D_min._block((features * 4) * 2, features * 4, name="dec3")
        self.M_upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.M_decoder2 = XNetv3_3D_min._block((features * 2) * 2, features * 2, name="dec2")
        self.M_upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.M_decoder1 = XNetv3_3D_min._block(features * 2, features, name="dec1")

        self.M_conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

        # L network
        self.L_encoder1 = XNetv3_3D_min._block(in_channels, features, name="enc1")
        self.L_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.L_encoder2 = XNetv3_3D_min._block(features, features * 2, name="enc2")
        self.L_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.L_encoder3 = XNetv3_3D_min._block(features * 2, features * 4, name="enc3")
        self.L_pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.L_encoder4 = XNetv3_3D_min._block(features * 4, features * 8, name="enc4")
        self.L_pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.L_bottleneck = XNetv3_3D_min._block(features * 8, features * 16, name="bottleneck")

        self.L_upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.L_decoder4 = XNetv3_3D_min._block((features * 8) * 2 , features * 8, name="dec4")
        self.L_upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.L_decoder3 = XNetv3_3D_min._block((features * 4) * 2, features * 4, name="dec3")
        self.L_upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.L_decoder2 = XNetv3_3D_min._block((features * 2), features * 2, name="dec2")
        self.L_upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.L_decoder1 = XNetv3_3D_min._block(features, features, name="dec1")
        self.L_conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

        # H network
        self.H_encoder1 = XNetv3_3D_min._block(in_channels, features, name="enc1")
        self.H_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.H_encoder2 = XNetv3_3D_min._block(features, features * 2, name="enc2")
        self.H_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.H_encoder3 = XNetv3_3D_min._block(features * 2, features * 4, name="enc3")
        self.H_pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.H_encoder4 = XNetv3_3D_min._block(features * 4, features * 8, name="enc4")
        self.H_pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.H_bottleneck = XNetv3_3D_min._block(features * 8, features * 16, name="bottleneck")

        self.H_upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.H_decoder4 = XNetv3_3D_min._block((features * 8), features * 8, name="dec4")
        self.H_upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.H_decoder3 = XNetv3_3D_min._block((features * 4), features * 4, name="dec3")
        self.H_upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.H_decoder2 = XNetv3_3D_min._block((features * 2) * 2, features * 2, name="dec2")
        self.H_upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.H_decoder1 = XNetv3_3D_min._block(features * 2, features, name="dec1")
        self.H_conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

        # fusion
        self.M_H_conv1 = XNetv3_3D_min._block(features * 2, features, name="fusion1")
        self.M_H_conv2 = XNetv3_3D_min._block(features * 4, features * 2, name="fusion2")
        self.M_L_conv3 = XNetv3_3D_min._block(features * 8, features * 4, name="fusion3")
        self.M_L_conv4 = XNetv3_3D_min._block(features * 16, features * 8, name="fusion4")

    def forward(self, x_main, x_L, x_HA,x_HR):
        # x_L_new=torch.cat((x_L,  x_HA), dim=1)
        # x_H_new=torch.cat((x_L,  x_HR), dim=1)
        x_L_new=0.5*x_L+0.5*x_HA
        x_H_new=0.5*x_L+0.5*x_HR
        # Main encoder
        M_enc1 = self.M_encoder1(x_main)
        M_enc2 = self.M_encoder2(self.M_pool1(M_enc1))
        M_enc3 = self.M_encoder3(self.M_pool2(M_enc2))
        M_enc4 = self.M_encoder4(self.M_pool3(M_enc3))

        M_bottleneck = self.M_bottleneck(self.M_pool4(M_enc4))

        # L encoder
        L_enc1 = self.L_encoder1(x_L_new)
        L_enc2 = self.L_encoder2(self.L_pool1(L_enc1))
        L_enc3 = self.L_encoder3(self.L_pool2(L_enc2))
        L_enc4 = self.L_encoder4(self.L_pool3(L_enc3))

        L_bottleneck = self.L_bottleneck(self.L_pool4(L_enc4))

        # H encoder
        H_enc1 = self.H_encoder1(x_H_new)
        H_enc2 = self.H_encoder2(self.H_pool1(H_enc1))
        H_enc3 = self.H_encoder3(self.H_pool2(H_enc2))
        H_enc4 = self.H_encoder4(self.H_pool3(H_enc3))

        H_bottleneck = self.H_bottleneck(self.H_pool4(H_enc4))

        # fusion
        M_H_enc1 = torch.cat((M_enc1, H_enc1), dim=1)
        M_H_enc1 = self.M_H_conv1(M_H_enc1)
        M_H_enc2 = torch.cat((M_enc2, H_enc2), dim=1)
        M_H_enc2 = self.M_H_conv2(M_H_enc2)
        M_L_enc3 = torch.cat((M_enc3, L_enc3), dim=1)
        M_L_enc3 = self.M_L_conv3(M_L_enc3)
        M_L_enc4 = torch.cat((M_enc4, L_enc4), dim=1)
        M_L_enc4 = self.M_L_conv4(M_L_enc4)

        # Main decoder
        M_dec4 = self.M_upconv4(M_bottleneck)
        M_dec4 = torch.cat((M_dec4, M_L_enc4), dim=1)
        M_dec4 = self.M_decoder4(M_dec4)
        
        M_dec3 = self.M_upconv3(M_dec4)
        M_dec3 = torch.cat((M_dec3, M_L_enc3), dim=1)
        M_dec3 = self.M_decoder3(M_dec3)
        
        M_dec2 = self.M_upconv2(M_dec3)
        M_dec2 = torch.cat((M_dec2, M_H_enc2), dim=1)
        M_dec2 = self.M_decoder2(M_dec2)
        
        M_dec1 = self.M_upconv1(M_dec2)
        M_dec1 = torch.cat((M_dec1, M_H_enc1), dim=1)
        M_dec1 = self.M_decoder1(M_dec1)
        M_outputs = self.M_conv(M_dec1)

        # L decoder
        L_dec4 = self.L_upconv4(L_bottleneck)
        L_dec4 = torch.cat((L_dec4, M_L_enc4), dim=1)
        L_dec4 = self.L_decoder4(L_dec4)
        
        L_dec3 = self.L_upconv3(L_dec4)
        L_dec3 = torch.cat((L_dec3, M_L_enc3), dim=1)
        L_dec3 = self.L_decoder3(L_dec3)
        
        L_dec2 = self.L_upconv2(L_dec3)
        # L_dec2 = torch.cat((L_dec2, L_enc2), dim=1)
        L_dec2 = self.L_decoder2(L_dec2)
        
        L_dec1 = self.L_upconv1(L_dec2)
        # L_dec1 = torch.cat((L_dec1, L_enc1), dim=1)
        L_dec1 = self.L_decoder1(L_dec1)
        L_outputs = self.L_conv(L_dec1)

        # H decoder
        H_dec4 = self.H_upconv4(H_bottleneck)
        # H_dec4 = torch.cat((H_dec4, H_enc4), dim=1)
        H_dec4 = self.H_decoder4(H_dec4)
        
        H_dec3 = self.H_upconv3(H_dec4)
        # H_dec3 = torch.cat((H_dec3, H_enc3), dim=1)
        H_dec3 = self.H_decoder3(H_dec3)
        
        H_dec2 = self.H_upconv2(H_dec3)
        H_dec2 = torch.cat((H_dec2, M_H_enc2), dim=1)
        H_dec2 = self.H_decoder2(H_dec2)
        
        H_dec1 = self.H_upconv1(H_dec2)
        H_dec1 = torch.cat((H_dec1, M_H_enc1), dim=1)
        H_dec1 = self.H_decoder1(H_dec1)
        H_outputs = self.H_conv(H_dec1)

        return M_outputs, L_outputs, H_outputs

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

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        # 为了获取空间注意力图，先通过卷积将通道数降为1
        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=kernel_size // 2)
        # 实例化一个批归一化层，有助于加速训练和提高模型的稳定性
        self.bn = nn.BatchNorm3d(1)
        # 使用ReLU激活函数引入非线性，增强模型的表达能力
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        参数:
        x: 输入的三维特征数据，形状为 (batch_size, channels, depth, height, width)，
           这里的channels为2，是将L和M分支特征在通道维度上拼接后的结果

        返回:
        经过空间注意力加权后的特征数据，形状同样为 (batch_size, channels, depth, height, width)
        """
        # 获取输入特征的batch_size、通道数、深度、高度和宽度信息
        batch_size, channels, depth, height, width = x.size()

        # 计算空间注意力图
        attention_map = self.conv1(x)
        attention_map = self.bn(attention_map)
        attention_map = self.relu(attention_map)

        # 对注意力图进行归一化，确保其值在0到1之间，便于作为权重进行加权融合
        attention_map = torch.sigmoid(attention_map)

        # 将注意力图与输入特征在通道维度上对应元素相乘，实现空间注意力加权
        return x * attention_map
import torch.fft
import torch.nn as nn


class FrequencyAttention(nn.Module):
    def __init__(self, in_channels):
        super(FrequencyAttention, self).__init__()

        # 全局平均池化层，用于将空间维度信息压缩，得到通道维度的特征描述符
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)

        # 两个全连接层，用于学习通道间的频域特征相关性
        self.fc1 = nn.Linear(in_channels, in_channels // 2)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // 2, in_channels)

        # 用于将学习到的频域注意力权重调整到合适的范围
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        参数:
        x: 输入的三维特征数据，形状为 (batch_size, channels, depth, height, width)

        返回:
        经过频域注意力加权后的特征数据，形状同样为 (batch_size, channels, depth, height, width)
        """
        batch_size, channels, _, _, _ = x.size()

        # 对输入特征进行离散傅里叶变换（DFT），得到频域表示
        freq_domain = torch.fft.fftn(x, dim=(2, 3, 4))
        # 取频域表示的幅度谱（绝对值），作为后续分析频域特征的基础
        magnitude_spectrum = torch.abs(freq_domain)

        # 通过全局平均池化将空间维度信息压缩到通道维度
        avg_pooled = self.global_avg_pool(magnitude_spectrum).view(batch_size, -1)

        # 通过全连接层学习频域特征之间的相关性，得到通道维度的注意力权重
        attention_weights = self.fc1(avg_pooled)
        attention_weights = self.relu(attention_weights)
        attention_weights = self.fc2(attention_weights)

        # 使用sigmoid函数将注意力权重归一化到0到1之间
        attention_weights = self.sigmoid(attention_weights).view(batch_size, channels, 1, 1, 1)

        # 将频域注意力权重与原始输入特征在通道维度上相乘，实现频域注意力加权
        return x * attention_weights

class XNetv3_3D_right(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, init_features=32):
        """
        Implementations based on the Unet3D paper: https://arxiv.org/abs/1606.06650
        """

        super(XNetv3_3D_right, self).__init__()

        features = init_features
        
        # # 可学习参数用于交换L和H信息
        # self.lh_exchange_weight_L = nn.Parameter(torch.clamp(torch.randn(1), min=0.2, max=1))  # 这里根据实际情况调整参数维度
        # self.lh_exchange_weight_H = nn.Parameter(torch.clamp(torch.randn(1), min=0.2, max=1))  # 这里根据实际情况调整参数维度

        # main network
        self.M_encoder1 = XNetv3_3D_right._block(in_channels, features, name="enc1")
        self.M_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.M_encoder2 = XNetv3_3D_right._block(features, features * 2, name="enc2")
        self.M_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.M_encoder3 = XNetv3_3D_right._block(features * 2, features * 4, name="enc3")
        self.M_pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.M_encoder4 = XNetv3_3D_right._block(features * 4, features * 8, name="enc4")
        self.M_pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.M_bottleneck = XNetv3_3D_right._block(features * 8, features * 16, name="bottleneck")

        self.M_upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.M_decoder4 = XNetv3_3D_right._block((features * 8)  , features * 8, name="dec4")
        self.M_upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.M_decoder3 = XNetv3_3D_right._block((features * 4) , features * 4, name="dec3")
        self.M_upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.M_decoder2 = XNetv3_3D_right._block((features * 2) * 3, features * 2, name="dec2")
        self.M_upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.M_decoder1 = XNetv3_3D_right._block(features * 3, features, name="dec1")

        self.M_conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

        # L network
        self.L_encoder1 = XNetv3_3D_right._block(in_channels, features, name="enc1")
        self.L_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.L_encoder2 = XNetv3_3D_right._block(features, features * 2, name="enc2")
        self.L_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.L_encoder3 = XNetv3_3D_right._block(features * 2, features * 4, name="enc3")
        self.L_pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.L_encoder4 = XNetv3_3D_right._block(features * 4, features * 8, name="enc4")
        self.L_pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.L_bottleneck = XNetv3_3D_right._block(features * 8, features * 16, name="bottleneck")

        self.L_upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.L_decoder4 = XNetv3_3D_right._block((features * 8)  , features * 8, name="dec4")
        self.L_upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.L_decoder3 = XNetv3_3D_right._block((features * 4) , features * 4, name="dec3")
        self.L_upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.L_decoder2 = XNetv3_3D_right._block((features * 2) *2, features * 2, name="dec2")
        self.L_upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.L_decoder1 = XNetv3_3D_right._block(features*2, features, name="dec1")
        self.L_conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

        # H network
        self.H_encoder1 = XNetv3_3D_right._block(in_channels, features, name="enc1")
        self.H_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.H_encoder2 = XNetv3_3D_right._block(features, features * 2, name="enc2")
        self.H_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.H_encoder3 = XNetv3_3D_right._block(features * 2, features * 4, name="enc3")
        self.H_pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.H_encoder4 = XNetv3_3D_right._block(features * 4, features * 8, name="enc4")
        self.H_pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.H_bottleneck = XNetv3_3D_right._block(features * 8, features * 16, name="bottleneck")

        self.H_upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.H_decoder4 = XNetv3_3D_right._block((features * 8)  , features * 8, name="dec4")
        self.H_upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.H_decoder3 = XNetv3_3D_right._block((features * 4) , features * 4, name="dec3")
        self.H_upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.H_decoder2 = XNetv3_3D_right._block((features * 2) *2, features * 2, name="dec2")
        self.H_upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.H_decoder1 = XNetv3_3D_right._block(features*2, features, name="dec1")
        self.H_conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)


        # fusion
        self.M_H_conv1 = XNetv3_3D_right._block(features * 2, features, name="fusion1")
        self.M_H_conv2 = XNetv3_3D_right._block(features * 4, features * 2, name="fusion2")
        self.M_H_conv3 = XNetv3_3D_right._block(features * 8, features*4, name="fusion3")
        self.M_H_conv4 = XNetv3_3D_right._block(features * 16, features * 8, name="fusion4")
        
        
        self.M_L_conv1 = XNetv3_3D_right._block(features * 2, features, name="fusion1")
        self.M_L_conv2 = XNetv3_3D_right._block(features * 4, features * 2, name="fusion2")
        self.M_L_conv3 = XNetv3_3D_right._block(features * 8, features * 4, name="fusion3")
        self.M_L_conv4 = XNetv3_3D_right._block(features * 16, features * 8, name="fusion4")


    def forward(self, x_main, x_L, x_HA,x_HR):
        
        # x_L_new = x_L + self.lh_exchange_weight_L * x_HA
        # x_H_new = x_L + self.lh_exchange_weight_H * x_HR
        x_L_new = 0.5*x_L + 0.5* x_HA
        x_H_new =  0.5*x_L +  0.5*x_HR
        # x_L_new = (x_L_new -torch.mean(x_L_new)) / torch.std(x_L_new)
        # x_H_new = (x_H_new -torch.mean(x_H_new)) / torch.std(x_H_new)
        
        # Main encoder
        M_enc1 = self.M_encoder1(x_main)
        M_enc2 = self.M_encoder2(self.M_pool1(M_enc1))
        M_enc3 = self.M_encoder3(self.M_pool2(M_enc2))
        M_enc4 = self.M_encoder4(self.M_pool3(M_enc3))

        M_bottleneck = self.M_bottleneck(self.M_pool4(M_enc4))

        # L encoder
        L_enc1 = self.L_encoder1(x_L_new)
        L_enc2 = self.L_encoder2(self.L_pool1(L_enc1))
        L_enc3 = self.L_encoder3(self.L_pool2(L_enc2))
        L_enc4 = self.L_encoder4(self.L_pool3(L_enc3))

        L_bottleneck = self.L_bottleneck(self.L_pool4(L_enc4))

        # H encoder
        H_enc1 = self.H_encoder1(x_H_new)
        H_enc2 = self.H_encoder2(self.H_pool1(H_enc1))
        H_enc3 = self.H_encoder3(self.H_pool2(H_enc2))
        H_enc4 = self.H_encoder4(self.H_pool3(H_enc3))

        H_bottleneck = self.H_bottleneck(self.H_pool4(H_enc4))

        # fusion
        M_H_enc1 = torch.cat((M_enc1, H_enc1), dim=1)
        M_H_enc1 = self.M_H_conv1(M_H_enc1)
        M_H_enc2 = torch.cat((M_enc2, H_enc2), dim=1)
        M_H_enc2 = self.M_H_conv2(M_H_enc2)
        
        # M_L_enc3 = torch.cat((M_enc3, L_enc3), dim=1)
        # M_L_enc3 = self.M_L_conv3(M_L_enc3)
        # M_L_enc4 = torch.cat((M_enc4, L_enc4), dim=1)
        # M_L_enc4 = self.M_L_conv4(M_L_enc4)
        
        M_L_enc1 = torch.cat((M_enc1, L_enc1), dim=1)
        M_L_enc1 = self.M_L_conv1(M_L_enc1)
        M_L_enc2 = torch.cat((M_enc2, L_enc2), dim=1)
        M_L_enc2 = self.M_L_conv2(M_L_enc2)
        
        # L_H_enc3 = torch.cat((M_enc1, H_enc1), dim=1)
        # L_H_enc3 = self.M_H_conv1(M_H_enc1)
        # L_H_enc4 = torch.cat((M_enc2, H_enc2), dim=1)
        # L_H_enc4 = self.M_H_conv2(M_H_enc2)

        # Main decoder
        M_dec4 = self.M_upconv4(M_bottleneck)
        # M_dec4 = torch.cat((M_dec4, M_L_enc4), dim=1)
        M_dec4 = self.M_decoder4(M_dec4)
        M_dec3 = self.M_upconv3(M_dec4)
        # M_dec3 = torch.cat((M_dec3, M_L_enc3), dim=1)
        M_dec3 = self.M_decoder3(M_dec3)
        M_dec2 = self.M_upconv2(M_dec3)
        M_dec2 = torch.cat((M_dec2, M_H_enc2, M_L_enc2), dim=1)
        M_dec2 = self.M_decoder2(M_dec2)
        M_dec1 = self.M_upconv1(M_dec2)
        M_dec1 = torch.cat((M_dec1, M_H_enc1, M_L_enc1), dim=1)
        M_dec1 = self.M_decoder1(M_dec1)
        M_outputs = self.M_conv(M_dec1)

        # L decoder
        L_dec4 = self.L_upconv4(L_bottleneck)
        # L_dec4 = torch.cat((L_dec4, M_L_enc4), dim=1)
        L_dec4 = self.L_decoder4(L_dec4)
        L_dec3 = self.L_upconv3(L_dec4)
        # L_dec3 = torch.cat((L_dec3, M_L_enc3), dim=1)
        L_dec3 = self.L_decoder3(L_dec3)
        L_dec2 = self.L_upconv2(L_dec3)
        L_dec2 = torch.cat((L_dec2, M_L_enc2), dim=1)
        L_dec2 = self.L_decoder2(L_dec2)
        
        L_dec1 = self.L_upconv1(L_dec2)
        L_dec1 = torch.cat((L_dec1, M_L_enc1), dim=1)
        L_dec1 = self.L_decoder1(L_dec1)
        L_outputs = self.L_conv(L_dec1)

        # H decoder
        H_dec4 = self.H_upconv4(H_bottleneck)
        # H_dec4 = torch.cat((H_dec4, H_enc4), dim=1)
        H_dec4 = self.H_decoder4(H_dec4)
        
        H_dec3 = self.H_upconv3(H_dec4)
        # H_dec3 = torch.cat((H_dec3, H_enc3), dim=1)
        H_dec3 = self.H_decoder3(H_dec3)
        
        H_dec2 = self.H_upconv2(H_dec3)
        H_dec2 = torch.cat((H_dec2, M_H_enc2), dim=1)
        H_dec2 = self.H_decoder2(H_dec2)
        
        H_dec1 = self.H_upconv1(H_dec2)
        H_dec1 = torch.cat((H_dec1, M_H_enc1), dim=1)
        H_dec1 = self.H_decoder1(H_dec1)
        
        H_outputs = self.H_conv(H_dec1)

        return M_outputs, L_outputs, H_outputs

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
   

class XNetv3_3D_new(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, init_features=32):
        """
        Implementations based on the Unet3D paper: https://arxiv.org/abs/1606.06650
        """

        super(XNetv3_3D_new, self).__init__()

        features = init_features
        
        # # 可学习参数用于交换L和H信息
        # self.lh_exchange_weight_L = nn.Parameter(torch.clamp(torch.randn(1), min=0.2, max=1))  # 这里根据实际情况调整参数维度
        # self.lh_exchange_weight_H = nn.Parameter(torch.clamp(torch.randn(1), min=0.2, max=1))  # 这里根据实际情况调整参数维度

        # main network
        self.M_encoder1 = XNetv3_3D_new._block(in_channels, features, name="enc1")
        self.M_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.M_encoder2 = XNetv3_3D_new._block(features, features * 2, name="enc2")
        self.M_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.M_encoder3 = XNetv3_3D_new._block(features * 2, features * 4, name="enc3")
        self.M_pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.M_encoder4 = XNetv3_3D_new._block(features * 4, features * 8, name="enc4")
        self.M_pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.M_bottleneck = XNetv3_3D_new._block(features * 8, features * 16, name="bottleneck")

        self.M_upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.M_decoder4 = XNetv3_3D_new._block((features * 8)  , features * 8, name="dec4")
        self.M_upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.M_decoder3 = XNetv3_3D_new._block((features * 4) * 2, features * 4, name="dec3")
        self.M_upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.M_decoder2 = XNetv3_3D_new._block((features * 2) * 2, features * 2, name="dec2")
        self.M_upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.M_decoder1 = XNetv3_3D_new._block(features*2, features, name="dec1")

        self.M_conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

        # L network
        self.L_encoder1 = XNetv3_3D_new._block(in_channels, features, name="enc1")
        self.L_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.L_encoder2 = XNetv3_3D_new._block(features, features * 2, name="enc2")
        self.L_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.L_encoder3 = XNetv3_3D_new._block(features * 2, features * 4, name="enc3")
        self.L_pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.L_encoder4 = XNetv3_3D_new._block(features * 4, features * 8, name="enc4")
        self.L_pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.L_bottleneck = XNetv3_3D_new._block(features * 8, features * 16, name="bottleneck")

        self.L_upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.L_decoder4 = XNetv3_3D_new._block((features * 8) , features * 8, name="dec4")
        self.L_upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.L_decoder3 = XNetv3_3D_new._block((features * 4) * 2, features * 4, name="dec3")
        self.L_upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.L_decoder2 = XNetv3_3D_new._block((features * 2) * 2, features * 2, name="dec2")
        self.L_upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.L_decoder1 = XNetv3_3D_new._block(features*2, features, name="dec1")
        self.L_conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

        # H network
        self.H_encoder1 = XNetv3_3D_new._block(in_channels, features, name="enc1")
        self.H_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.H_encoder2 = XNetv3_3D_new._block(features, features * 2, name="enc2")
        self.H_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.H_encoder3 = XNetv3_3D_new._block(features * 2, features * 4, name="enc3")
        self.H_pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.H_encoder4 = XNetv3_3D_new._block(features * 4, features * 8, name="enc4")
        self.H_pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.H_bottleneck = XNetv3_3D_new._block(features * 8, features * 16, name="bottleneck")

        self.H_upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.H_decoder4 = XNetv3_3D_new._block((features * 8), features * 8, name="dec4")
        self.H_upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.H_decoder3 = XNetv3_3D_new._block((features * 4) * 2, features * 4, name="dec3")
        self.H_upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.H_decoder2 = XNetv3_3D_new._block((features * 2) * 2, features * 2, name="dec2")
        self.H_upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.H_decoder1 = XNetv3_3D_new._block(features*2, features, name="dec1")
        self.H_conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

        # fusion
        # self.M_H_conv1 = XNetv3_3D_new._block(features * 2, features, name="fusion1")
        # self.M_H_conv2 = XNetv3_3D_new._block(features * 4, features * 2, name="fusion2")
        # self.M_L_conv1 = XNetv3_3D_new._block(features * 2, features, name="fusion1")
        # self.M_L_conv2 = XNetv3_3D_new._block(features * 4, features * 2, name="fusion2")
        # self.M_L_conv3 = XNetv3_3D_new._block(features * 8, features * 4, name="fusion3")
        # self.M_L_conv4 = XNetv3_3D_new._block(features * 16, features * 8, name="fusion4")
        # self.M_L_H_conv1 = XNetv3_3D_new._block(features * 3, features , name="fusion1")
        self.M_L_H_conv2 = XNetv3_3D_new._block(features * 6, features * 2, name="fusion2")
        self.M_L_H_conv3 = XNetv3_3D_new._block(features * 12, features * 4, name="fusion3")

    def forward(self, x_main, x_L, x_HA,x_HR):
        
        # x_L_new = x_L + self.lh_exchange_weight_L * x_HA
        # x_H_new = x_L + self.lh_exchange_weight_H * x_HR
        x_L_new = 0.5*x_L + 0.5*x_HA
        x_H_new = 0.5*x_L + 0.5*x_HR
        # x_L_new = (x_L_new -torch.mean(x_L_new)) / torch.std(x_L_new)
        # x_H_new = (x_H_new -torch.mean(x_H_new)) / torch.std(x_H_new)
        
        # Main encoder
        M_enc1 = self.M_encoder1(x_main)
        M_enc2 = self.M_encoder2(self.M_pool1(M_enc1))
        M_enc3 = self.M_encoder3(self.M_pool2(M_enc2))
        M_enc4 = self.M_encoder4(self.M_pool3(M_enc3))

        M_bottleneck = self.M_bottleneck(self.M_pool4(M_enc4))

        # L encoder
        L_enc1 = self.L_encoder1(x_L_new)
        L_enc2 = self.L_encoder2(self.L_pool1(L_enc1))
        L_enc3 = self.L_encoder3(self.L_pool2(L_enc2))
        L_enc4 = self.L_encoder4(self.L_pool3(L_enc3))

        L_bottleneck = self.L_bottleneck(self.L_pool4(L_enc4))

        # H encoder
        H_enc1 = self.H_encoder1(x_H_new)
        H_enc2 = self.H_encoder2(self.H_pool1(H_enc1))
        H_enc3 = self.H_encoder3(self.H_pool2(H_enc2))
        H_enc4 = self.H_encoder4(self.H_pool3(H_enc3))

        H_bottleneck = self.H_bottleneck(self.H_pool4(H_enc4))

        # fusion
        # M_H_enc1 = torch.cat((M_enc1, H_enc1), dim=1)
        # M_H_enc1 = self.M_H_conv1(M_H_enc1)
        # M_H_enc2 = torch.cat((M_enc2, H_enc2), dim=1)
        # M_H_enc2 = self.M_H_conv2(M_H_enc2)
        
        # M_L_enc3 = torch.cat((M_enc3, L_enc3), dim=1)
        # M_L_enc3 = self.M_L_conv3(M_L_enc3)
        # M_L_enc4 = torch.cat((M_enc4, L_enc4), dim=1)
        # M_L_enc4 = self.M_L_conv4(M_L_enc4)
        # M_L_H_enc1 = torch.cat((M_enc1, L_enc1 , H_enc1), dim=1)
        # M_L_H_enc1 = self.M_L_H_conv1(M_L_H_enc1)
        M_L_H_enc2 = torch.cat((M_enc2, L_enc2 , H_enc2), dim=1)
        M_L_H_enc2 = self.M_L_H_conv2(M_L_H_enc2)
        M_L_H_enc3 = torch.cat((M_enc3, L_enc3 , H_enc3), dim=1)
        M_L_H_enc3 = self.M_L_H_conv3(M_L_H_enc3)
        
       
        # L_H_enc3 = torch.cat((M_enc1, H_enc1), dim=1)
        # L_H_enc3 = self.M_H_conv1(M_H_enc1)
        # L_H_enc4 = torch.cat((M_enc2, H_enc2), dim=1)
        # L_H_enc4 = self.M_H_conv2(M_H_enc2)

        # Main decoder
        M_dec4 = self.M_upconv4(M_bottleneck)
        # M_dec4 = torch.cat((M_dec4, M_L_enc4), dim=1)
        M_dec4 = self.M_decoder4(M_dec4)
        M_dec3 = self.M_upconv3(M_dec4)
        M_dec3 = torch.cat((M_dec3, M_L_H_enc3), dim=1)
        M_dec3 = self.M_decoder3(M_dec3)
        M_dec2 = self.M_upconv2(M_dec3)
        M_dec2 = torch.cat((M_dec2, M_L_H_enc2), dim=1)
        M_dec2 = self.M_decoder2(M_dec2)
        M_dec1 = self.M_upconv1(M_dec2)
        # M_dec1 = torch.cat((M_dec1, M_L_H_enc1), dim=1)
        M_dec1 = self.M_decoder1(M_dec1)
        M_outputs = self.M_conv(M_dec1)

        # L decoder
        L_dec4 = self.L_upconv4(L_bottleneck)
        # L_dec4 = torch.cat((L_dec4, M_L_H_enc4), dim=1)
        L_dec4 = self.L_decoder4(L_dec4)
        L_dec3 = self.L_upconv3(L_dec4)
        L_dec3 = torch.cat((L_dec3, M_L_H_enc3), dim=1)
        L_dec3 = self.L_decoder3(L_dec3)
        L_dec2 = self.L_upconv2(L_dec3)
        L_dec2 = torch.cat((L_dec2, M_L_H_enc2), dim=1)
        L_dec2 = self.L_decoder2(L_dec2)
        L_dec1 = self.L_upconv1(L_dec2)
        # L_dec1 = torch.cat((L_dec1, M_L_H_enc1), dim=1)
        L_dec1 = self.L_decoder1(L_dec1)
        L_outputs = self.L_conv(L_dec1)

        # H decoder
        H_dec4 = self.H_upconv4(H_bottleneck)
        # H_dec4 = torch.cat((H_dec4, H_enc4), dim=1)
        H_dec4 = self.H_decoder4(H_dec4)
        H_dec3 = self.H_upconv3(H_dec4)
        H_dec3 = torch.cat((H_dec3, M_L_H_enc3), dim=1)
        H_dec3 = self.H_decoder3(H_dec3)
        H_dec2 = self.H_upconv2(H_dec3)
        H_dec2 = torch.cat((H_dec2, M_L_H_enc2), dim=1)
        H_dec2 = self.H_decoder2(H_dec2)
        H_dec1 = self.H_upconv1(H_dec2)
        # H_dec1 = torch.cat((H_dec1, M_L_H_enc1), dim=1)
        H_dec1 = self.H_decoder1(H_dec1)
        H_outputs = self.H_conv(H_dec1)

        return M_outputs, L_outputs, H_outputs

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

class XNetv3_3D_xh(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, init_features=32):
        """
        Implementations based on the Unet3D paper: https://arxiv.org/abs/1606.06650
        """

        super(XNetv3_3D_xh, self).__init__()

        features = init_features 
        
        # # 可学习参数用于交换L和H信息
        # self.lh_exchange_weight_L = nn.Parameter(torch.clamp(torch.randn(1), min=0.2, max=1))  # 这里根据实际情况调整参数维度
        # self.lh_exchange_weight_H = nn.Parameter(torch.clamp(torch.randn(1), min=0.2, max=1))  # 这里根据实际情况调整参数维度

        # main network
        self.M_encoder1 = XNetv3_3D_new._block(in_channels, features, name="enc1")
        self.M_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.M_encoder2 = XNetv3_3D_new._block(features, features * 2, name="enc2")
        self.M_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.M_encoder3 = XNetv3_3D_new._block(features * 2, features * 4, name="enc3")
        self.M_pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.M_encoder4 = XNetv3_3D_new._block(features * 4, features * 8, name="enc4")
        self.M_pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.M_bottleneck = XNetv3_3D_new._block(features * 8, features * 16, name="bottleneck")

        self.M_upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.M_decoder4 = XNetv3_3D_new._block((features * 8)  , features * 8, name="dec4")
        self.M_upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.M_decoder3 = XNetv3_3D_new._block((features * 4) * 2, features * 4, name="dec3")
        self.M_upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.M_decoder2 = XNetv3_3D_new._block((features * 2) * 2, features * 2, name="dec2")
        self.M_upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.M_decoder1 = XNetv3_3D_new._block(features, features, name="dec1")

        self.M_conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

        # L network
        self.L_encoder1 = XNetv3_3D_new._block(in_channels, features, name="enc1")
        self.L_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.L_encoder2 = XNetv3_3D_new._block(features, features * 2, name="enc2")
        self.L_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.L_encoder3 = XNetv3_3D_new._block(features * 2, features * 4, name="enc3")
        self.L_pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.L_encoder4 = XNetv3_3D_new._block(features * 4, features * 8, name="enc4")
        self.L_pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.L_bottleneck = XNetv3_3D_new._block(features * 8, features * 16, name="bottleneck")

        self.L_upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.L_decoder4 = XNetv3_3D_new._block((features * 8) , features * 8, name="dec4")
        self.L_upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.L_decoder3 = XNetv3_3D_new._block((features * 4) * 2, features * 4, name="dec3")
        self.L_upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.L_decoder2 = XNetv3_3D_new._block((features * 2) * 2, features * 2, name="dec2")
        self.L_upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.L_decoder1 = XNetv3_3D_new._block(features, features, name="dec1")
        self.L_conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

        # H network
        self.H_encoder1 = XNetv3_3D_new._block(in_channels, features, name="enc1")
        self.H_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.H_encoder2 = XNetv3_3D_new._block(features, features * 2, name="enc2")
        self.H_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.H_encoder3 = XNetv3_3D_new._block(features * 2, features * 4, name="enc3")
        self.H_pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.H_encoder4 = XNetv3_3D_new._block(features * 4, features * 8, name="enc4")
        self.H_pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.H_bottleneck = XNetv3_3D_new._block(features * 8, features * 16, name="bottleneck")

        self.H_upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.H_decoder4 = XNetv3_3D_new._block((features * 8), features * 8, name="dec4")
        self.H_upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.H_decoder3 = XNetv3_3D_new._block((features * 4) * 2, features * 4, name="dec3")
        self.H_upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.H_decoder2 = XNetv3_3D_new._block((features * 2) * 2, features * 2, name="dec2")
        self.H_upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.H_decoder1 = XNetv3_3D_new._block(features , features, name="dec1")
        self.H_conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

        # fusion
        # self.M_H_conv1 = XNetv3_3D_new._block(features * 2, features, name="fusion1")
        # self.M_H_conv2 = XNetv3_3D_new._block(features * 4, features * 2, name="fusion2")
        # self.M_L_conv1 = XNetv3_3D_new._block(features * 2, features, name="fusion1")
        # self.M_L_conv2 = XNetv3_3D_new._block(features * 4, features * 2, name="fusion2")
        # self.M_L_conv3 = XNetv3_3D_new._block(features * 8, features * 4, name="fusion3")
        # self.M_L_conv4 = XNetv3_3D_new._block(features * 16, features * 8, name="fusion4")
        
        self.M_L_conv2 = XNetv3_3D_new._block(features * 4, features * 2, name="fusion2")
        self.M_L_conv3 = XNetv3_3D_new._block(features * 8, features * 4, name="fusion3")
        
        self.H_M_conv2 = XNetv3_3D_new._block(features * 4, features * 2, name="fusion2")
        self.H_M_conv3 = XNetv3_3D_new._block(features * 8, features * 4, name="fusion3")
        
        self.L_H_conv2 = XNetv3_3D_new._block(features * 4, features * 2, name="fusion2")
        self.L_H_conv3 = XNetv3_3D_new._block(features * 8, features * 4, name="fusion3")

    def forward(self, x_main, x_L, x_HA,x_HR):
        
        # x_L_new = x_L + self.lh_exchange_weight_L * x_HA
        # x_H_new = x_L + self.lh_exchange_weight_H * x_HR
        x_L_new = 0.5*x_L + 0.5*x_HA
        x_H_new = 0.5*x_L + 0.5*x_HR
        # x_L_new = (x_L_new -torch.mean(x_L_new)) / torch.std(x_L_new)
        # x_H_new = (x_H_new -torch.mean(x_H_new)) / torch.std(x_H_new)
        
        # Main encoder
        M_enc1 = self.M_encoder1(x_main)
        M_enc2 = self.M_encoder2(self.M_pool1(M_enc1))
        M_enc3 = self.M_encoder3(self.M_pool2(M_enc2))
        M_enc4 = self.M_encoder4(self.M_pool3(M_enc3))

        M_bottleneck = self.M_bottleneck(self.M_pool4(M_enc4))

        # L encoder
        L_enc1 = self.L_encoder1(x_L_new)
        L_enc2 = self.L_encoder2(self.L_pool1(L_enc1))
        L_enc3 = self.L_encoder3(self.L_pool2(L_enc2))
        L_enc4 = self.L_encoder4(self.L_pool3(L_enc3))

        L_bottleneck = self.L_bottleneck(self.L_pool4(L_enc4))

        # H encoder
        H_enc1 = self.H_encoder1(x_H_new)
        H_enc2 = self.H_encoder2(self.H_pool1(H_enc1))
        H_enc3 = self.H_encoder3(self.H_pool2(H_enc2))
        H_enc4 = self.H_encoder4(self.H_pool3(H_enc3))

        H_bottleneck = self.H_bottleneck(self.H_pool4(H_enc4))

        # fusion
        # M_H_enc1 = torch.cat((M_enc1, H_enc1), dim=1)
        # M_H_enc1 = self.M_H_conv1(M_H_enc1)
        # M_H_enc2 = torch.cat((M_enc2, H_enc2), dim=1)
        # M_H_enc2 = self.M_H_conv2(M_H_enc2)
        
        # M_L_enc3 = torch.cat((M_enc3, L_enc3), dim=1)
        # M_L_enc3 = self.M_L_conv3(M_L_enc3)
        # M_L_enc4 = torch.cat((M_enc4, L_enc4), dim=1)
        # M_L_enc4 = self.M_L_conv4(M_L_enc4)
        M_L_enc2 = torch.cat((M_enc2, L_enc2), dim=1)
        M_L_enc2 = self.M_L_conv2(M_L_enc2)
        M_L_enc3 = torch.cat((M_enc3, L_enc3), dim=1)
        M_L_enc3 = self.M_L_conv3(M_L_enc3)
        
        H_M_enc2 = torch.cat((H_enc2, M_enc2), dim=1)
        H_M_enc2 = self.H_M_conv2(H_M_enc2)
        H_M_enc3 = torch.cat((H_enc3, M_enc3), dim=1)
        H_M_enc3 = self.H_M_conv3(H_M_enc3)
        
        L_H_enc2 = torch.cat((L_enc2, H_enc2), dim=1)
        L_H_enc2 = self.L_H_conv2(L_H_enc2)
        L_H_enc3 = torch.cat((L_enc3, H_enc3), dim=1)
        L_H_enc3 = self.L_H_conv3(L_H_enc3)


       
        # L_H_enc3 = torch.cat((M_enc1, H_enc1), dim=1)
        # L_H_enc3 = self.M_H_conv1(M_H_enc1)
        # L_H_enc4 = torch.cat((M_enc2, H_enc2), dim=1)
        # L_H_enc4 = self.M_H_conv2(M_H_enc2)

        # Main decoder
        M_dec4 = self.M_upconv4(M_bottleneck)
        # M_dec4 = torch.cat((M_dec4, M_L_enc4), dim=1)
        M_dec4 = self.M_decoder4(M_dec4)
        M_dec3 = self.M_upconv3(M_dec4)
        M_dec3 = torch.cat((M_dec3, M_L_enc3), dim=1)
        M_dec3 = self.M_decoder3(M_dec3)
        M_dec2 = self.M_upconv2(M_dec3)
        M_dec2 = torch.cat((M_dec2, M_L_enc2), dim=1)
        M_dec2 = self.M_decoder2(M_dec2)
        M_dec1 = self.M_upconv1(M_dec2)
        # M_dec1 = torch.cat((M_dec1, M_H_enc1), dim=1)
        M_dec1 = self.M_decoder1(M_dec1)
        M_outputs = self.M_conv(M_dec1)

        # L decoder
        L_dec4 = self.L_upconv4(L_bottleneck)
        # L_dec4 = torch.cat((L_dec4, M_L_H_enc4), dim=1)
        L_dec4 = self.L_decoder4(L_dec4)
        L_dec3 = self.L_upconv3(L_dec4)
        L_dec3 = torch.cat((L_dec3, L_H_enc3), dim=1)
        L_dec3 = self.L_decoder3(L_dec3)
        L_dec2 = self.L_upconv2(L_dec3)
        L_dec2 = torch.cat((L_dec2, L_H_enc2), dim=1)
        L_dec2 = self.L_decoder2(L_dec2)
        L_dec1 = self.L_upconv1(L_dec2)
        # L_dec1 = torch.cat((L_dec1, L_enc1), dim=1)
        L_dec1 = self.L_decoder1(L_dec1)
        L_outputs = self.L_conv(L_dec1)

        # H decoder
        H_dec4 = self.H_upconv4(H_bottleneck)
        # H_dec4 = torch.cat((H_dec4, H_enc4), dim=1)
        H_dec4 = self.H_decoder4(H_dec4)
        H_dec3 = self.H_upconv3(H_dec4)
        H_dec3 = torch.cat((H_dec3, H_M_enc3), dim=1)
        H_dec3 = self.H_decoder3(H_dec3)
        H_dec2 = self.H_upconv2(H_dec3)
        H_dec2 = torch.cat((H_dec2, H_M_enc2), dim=1)
        H_dec2 = self.H_decoder2(H_dec2)
        H_dec1 = self.H_upconv1(H_dec2)
        # H_dec1 = torch.cat((H_dec1, M_H_enc1), dim=1)
        H_dec1 = self.H_decoder1(H_dec1)
        H_outputs = self.H_conv(H_dec1)

        return M_outputs, L_outputs, H_outputs

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

class XNetv3_3D_jh(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, init_features=32):
        """
        Implementations based on the Unet3D paper: https://arxiv.org/abs/1606.06650
        """

        super(XNetv3_3D_jh, self).__init__()

        features = init_features
        
        # # 可学习参数用于交换L和H信息
        # self.lh_exchange_weight_L = nn.Parameter(torch.clamp(torch.randn(1), min=0.2, max=1))  # 这里根据实际情况调整参数维度
        # self.lh_exchange_weight_H = nn.Parameter(torch.clamp(torch.randn(1), min=0.2, max=1))  # 这里根据实际情况调整参数维度

        # main network
        self.M_encoder1 = XNetv3_3D_new._block(in_channels, features, name="enc1")
        self.M_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.M_encoder2 = XNetv3_3D_new._block(features, features * 2, name="enc2")
        self.M_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.M_encoder3 = XNetv3_3D_new._block(features * 2, features * 4, name="enc3")
        self.M_pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.M_encoder4 = XNetv3_3D_new._block(features * 4, features * 8, name="enc4")
        self.M_pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.M_bottleneck = XNetv3_3D_new._block(features * 8, features * 16, name="bottleneck")

        self.M_upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.M_decoder4 = XNetv3_3D_new._block((features * 8)  , features * 8, name="dec4")
        self.M_upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.M_decoder3 = XNetv3_3D_new._block((features * 4) * 2, features * 4, name="dec3")
        self.M_upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.M_decoder2 = XNetv3_3D_new._block((features * 2) * 2, features * 2, name="dec2")
        self.M_upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.M_decoder1 = XNetv3_3D_new._block(features, features, name="dec1")

        self.M_conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

        # L network
        self.L_encoder1 = XNetv3_3D_new._block(in_channels, features, name="enc1")
        self.L_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.L_encoder2 = XNetv3_3D_new._block(features, features * 2, name="enc2")
        self.L_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.L_encoder3 = XNetv3_3D_new._block(features * 2, features * 4, name="enc3")
        self.L_pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.L_encoder4 = XNetv3_3D_new._block(features * 4, features * 8, name="enc4")
        self.L_pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.L_bottleneck = XNetv3_3D_new._block(features * 8, features * 16, name="bottleneck")

        self.L_upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.L_decoder4 = XNetv3_3D_new._block((features * 8) , features * 8, name="dec4")
        self.L_upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.L_decoder3 = XNetv3_3D_new._block((features * 4) * 2, features * 4, name="dec3")
        self.L_upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.L_decoder2 = XNetv3_3D_new._block((features * 2) * 2, features * 2, name="dec2")
        self.L_upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.L_decoder1 = XNetv3_3D_new._block(features, features, name="dec1")
        self.L_conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

        # H network
        self.H_encoder1 = XNetv3_3D_new._block(in_channels, features, name="enc1")
        self.H_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.H_encoder2 = XNetv3_3D_new._block(features, features * 2, name="enc2")
        self.H_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.H_encoder3 = XNetv3_3D_new._block(features * 2, features * 4, name="enc3")
        self.H_pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.H_encoder4 = XNetv3_3D_new._block(features * 4, features * 8, name="enc4")
        self.H_pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.H_bottleneck = XNetv3_3D_new._block(features * 8, features * 16, name="bottleneck")

        self.H_upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.H_decoder4 = XNetv3_3D_new._block((features * 8), features * 8, name="dec4")
        self.H_upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.H_decoder3 = XNetv3_3D_new._block((features * 4) * 2, features * 4, name="dec3")
        self.H_upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.H_decoder2 = XNetv3_3D_new._block((features * 2) * 2, features * 2, name="dec2")
        self.H_upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.H_decoder1 = XNetv3_3D_new._block(features , features, name="dec1")
        self.H_conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

        # fusion
        # self.M_H_conv1 = XNetv3_3D_new._block(features * 2, features, name="fusion1")
        # self.M_H_conv2 = XNetv3_3D_new._block(features * 4, features * 2, name="fusion2")
        # self.M_L_conv1 = XNetv3_3D_new._block(features * 2, features, name="fusion1")
        # self.M_L_conv2 = XNetv3_3D_new._block(features * 4, features * 2, name="fusion2")
        # self.M_L_conv3 = XNetv3_3D_new._block(features * 8, features * 4, name="fusion3")
        # self.M_L_conv4 = XNetv3_3D_new._block(features * 16, features * 8, name="fusion4")
        
        self.M_L_conv2 = XNetv3_3D_new._block(features * 4, features * 2, name="fusion2")
        self.M_L_conv3 = XNetv3_3D_new._block(features * 8, features * 4, name="fusion3")
        
        self.H_M_conv2 = XNetv3_3D_new._block(features * 4, features * 2, name="fusion2")
        self.H_M_conv3 = XNetv3_3D_new._block(features * 8, features * 4, name="fusion3")
        
        self.L_H_conv2 = XNetv3_3D_new._block(features * 4, features * 2, name="fusion2")
        self.L_H_conv3 = XNetv3_3D_new._block(features * 8, features * 4, name="fusion3")

    def forward(self, x_main, x_L, x_HA,x_HR):
        
        # x_L_new = x_L + self.lh_exchange_weight_L * x_HA
        # x_H_new = x_L + self.lh_exchange_weight_H * x_HR
        x_L_new = 0.5 * x_L + 0.5 * x_HA
        x_H_new = 0.5 * x_L + 0.5 * x_HR
    
        # x_L_new = (x_L_new -torch.mean(x_L_new)) / torch.std(x_L_new)
        # x_H_new = (x_H_new -torch.mean(x_H_new)) / torch.std(x_H_new)
        
        # Main encoder
        M_enc1 = self.M_encoder1(x_main)
        M_enc2 = self.M_encoder2(self.M_pool1(M_enc1))
        M_enc3 = self.M_encoder3(self.M_pool2(M_enc2))
        M_enc4 = self.M_encoder4(self.M_pool3(M_enc3))

        M_bottleneck = self.M_bottleneck(self.M_pool4(M_enc4))

        # L encoder
        L_enc1 = self.L_encoder1(x_L_new)
        L_enc2 = self.L_encoder2(self.L_pool1(L_enc1))
        L_enc3 = self.L_encoder3(self.L_pool2(L_enc2))
        L_enc4 = self.L_encoder4(self.L_pool3(L_enc3))

        L_bottleneck = self.L_bottleneck(self.L_pool4(L_enc4))

        # H encoder
        H_enc1 = self.H_encoder1(x_H_new)
        H_enc2 = self.H_encoder2(self.H_pool1(H_enc1))
        H_enc3 = self.H_encoder3(self.H_pool2(H_enc2))
        H_enc4 = self.H_encoder4(self.H_pool3(H_enc3))

        H_bottleneck = self.H_bottleneck(self.H_pool4(H_enc4))

        # fusion
        # M_H_enc1 = torch.cat((M_enc1, H_enc1), dim=1)
        # M_H_enc1 = self.M_H_conv1(M_H_enc1)
        # M_H_enc2 = torch.cat((M_enc2, H_enc2), dim=1)
        # M_H_enc2 = self.M_H_conv2(M_H_enc2)
        
        # M_L_enc3 = torch.cat((M_enc3, L_enc3), dim=1)
        # M_L_enc3 = self.M_L_conv3(M_L_enc3)
        # M_L_enc4 = torch.cat((M_enc4, L_enc4), dim=1)
        # M_L_enc4 = self.M_L_conv4(M_L_enc4)
        M_L_enc2 = torch.cat((M_enc2, L_enc2), dim=1)
        M_L_enc2 = self.M_L_conv2(M_L_enc2)
        M_L_enc3 = torch.cat((M_enc3, L_enc3), dim=1)
        M_L_enc3 = self.M_L_conv3(M_L_enc3)
        
        H_M_enc2 = torch.cat((H_enc2, M_enc2), dim=1)
        H_M_enc2 = self.H_M_conv2(H_M_enc2)
        H_M_enc3 = torch.cat((H_enc3, M_enc3), dim=1)
        H_M_enc3 = self.H_M_conv3(H_M_enc3)
        
        L_H_enc2 = torch.cat((L_enc2, H_enc2), dim=1)
        L_H_enc2 = self.L_H_conv2(L_H_enc2)
        L_H_enc3 = torch.cat((L_enc3, H_enc3), dim=1)
        L_H_enc3 = self.L_H_conv3(L_H_enc3)


       
        # L_H_enc3 = torch.cat((M_enc1, H_enc1), dim=1)
        # L_H_enc3 = self.M_H_conv1(M_H_enc1)
        # L_H_enc4 = torch.cat((M_enc2, H_enc2), dim=1)
        # L_H_enc4 = self.M_H_conv2(M_H_enc2)

        # Main decoder
        M_dec4 = self.M_upconv4(M_bottleneck)
        # M_dec4 = torch.cat((M_dec4, M_L_enc4), dim=1)
        M_dec4 = self.M_decoder4(M_dec4)
        M_dec3 = self.M_upconv3(M_dec4)
        M_dec3 = torch.cat((M_dec3, L_H_enc3), dim=1)
        M_dec3 = self.M_decoder3(M_dec3)
        M_dec2 = self.M_upconv2(M_dec3)
        M_dec2 = torch.cat((M_dec2, L_H_enc2), dim=1)
        M_dec2 = self.M_decoder2(M_dec2)
        M_dec1 = self.M_upconv1(M_dec2)
        # M_dec1 = torch.cat((M_dec1, M_H_enc1), dim=1)
        M_dec1 = self.M_decoder1(M_dec1)
        M_outputs = self.M_conv(M_dec1)

        # L decoder
        L_dec4 = self.L_upconv4(L_bottleneck)
        # L_dec4 = torch.cat((L_dec4, M_L_H_enc4), dim=1)
        L_dec4 = self.L_decoder4(L_dec4)
        L_dec3 = self.L_upconv3(L_dec4)
        L_dec3 = torch.cat((L_dec3, H_M_enc3), dim=1)
        L_dec3 = self.L_decoder3(L_dec3)
        L_dec2 = self.L_upconv2(L_dec3)
        L_dec2 = torch.cat((L_dec2, H_M_enc2), dim=1)
        L_dec2 = self.L_decoder2(L_dec2)
        L_dec1 = self.L_upconv1(L_dec2)
        # L_dec1 = torch.cat((L_dec1, L_enc1), dim=1)
        L_dec1 = self.L_decoder1(L_dec1)
        L_outputs = self.L_conv(L_dec1)

        # H decoder
        H_dec4 = self.H_upconv4(H_bottleneck)
        # H_dec4 = torch.cat((H_dec4, H_enc4), dim=1)
        H_dec4 = self.H_decoder4(H_dec4)
        H_dec3 = self.H_upconv3(H_dec4)
        H_dec3 = torch.cat((H_dec3, M_L_enc3), dim=1)
        H_dec3 = self.H_decoder3(H_dec3)
        H_dec2 = self.H_upconv2(H_dec3)
        H_dec2 = torch.cat((H_dec2, M_L_enc2), dim=1)
        H_dec2 = self.H_decoder2(H_dec2)
        H_dec1 = self.H_upconv1(H_dec2)
        # H_dec1 = torch.cat((H_dec1, M_H_enc1), dim=1)
        H_dec1 = self.H_decoder1(H_dec1)
        H_outputs = self.H_conv(H_dec1)

        return M_outputs, L_outputs, H_outputs

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
        
    
class XNetv3_3D_re(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, init_features=32):
        """
        Implementations based on the Unet3D paper: https://arxiv.org/abs/1606.06650
        """

        super(XNetv3_3D_re, self).__init__()

        features = init_features
        
        # # 可学习参数用于交换L和H信息
        # self.lh_exchange_weight_L = nn.Parameter(torch.clamp(torch.randn(1), min=0.2, max=1))  # 这里根据实际情况调整参数维度
        # self.lh_exchange_weight_H = nn.Parameter(torch.clamp(torch.randn(1), min=0.2, max=1))  # 这里根据实际情况调整参数维度

        # main network
        self.M_encoder1 = XNetv3_3D_re._block(in_channels, features, name="enc1")
        self.M_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.M_encoder2 = XNetv3_3D_re._block(features, features * 2, name="enc2")
        self.M_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.M_encoder3 = XNetv3_3D_re._block(features * 2, features * 4, name="enc3")
        self.M_pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.M_encoder4 = XNetv3_3D_re._block(features * 4, features * 8, name="enc4")
        self.M_pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.M_bottleneck = XNetv3_3D_re._block(features * 8, features * 16, name="bottleneck")

        self.M_upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.M_decoder4 = XNetv3_3D_re._block((features * 8)  , features * 8, name="dec4")
        self.M_upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.M_decoder3 = XNetv3_3D_re._block((features * 4) * 2, features * 4, name="dec3")
        self.M_upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.M_decoder2 = XNetv3_3D_re._block((features * 2) * 2, features * 2, name="dec2")
        self.M_upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.M_decoder1 = XNetv3_3D_re._block(features, features, name="dec1")

        self.M_conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

        # L network
        self.L_encoder1 = XNetv3_3D_re._block(in_channels *2, features, name="enc1")
        self.L_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.L_encoder2 = XNetv3_3D_re._block(features, features * 2, name="enc2")
        self.L_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.L_encoder3 = XNetv3_3D_re._block(features * 2, features * 4, name="enc3")
        self.L_pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.L_encoder4 = XNetv3_3D_re._block(features * 4, features * 8, name="enc4")
        self.L_pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.L_bottleneck = XNetv3_3D_re._block(features * 8, features * 16, name="bottleneck")

        self.L_upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.L_decoder4 = XNetv3_3D_re._block((features * 8) , features * 8, name="dec4")
        self.L_upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.L_decoder3 = XNetv3_3D_re._block((features * 4) * 2, features * 4, name="dec3")
        self.L_upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.L_decoder2 = XNetv3_3D_re._block((features * 2) * 2, features * 2, name="dec2")
        self.L_upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.L_decoder1 = XNetv3_3D_re._block(features, features, name="dec1")
        self.L_conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

        # H network
        self.H_encoder1 = XNetv3_3D_re._block(in_channels *2, features, name="enc1")
        self.H_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.H_encoder2 = XNetv3_3D_re._block(features, features * 2, name="enc2")
        self.H_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.H_encoder3 = XNetv3_3D_re._block(features * 2, features * 4, name="enc3")
        self.H_pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.H_encoder4 = XNetv3_3D_re._block(features * 4, features * 8, name="enc4")
        self.H_pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.H_bottleneck = XNetv3_3D_re._block(features * 8, features * 16, name="bottleneck")

        self.H_upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.H_decoder4 = XNetv3_3D_re._block((features * 8), features * 8, name="dec4")
        self.H_upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.H_decoder3 = XNetv3_3D_re._block((features * 4) * 2, features * 4, name="dec3")
        self.H_upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.H_decoder2 = XNetv3_3D_re._block((features * 2) * 2, features * 2, name="dec2")
        self.H_upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.H_decoder1 = XNetv3_3D_re._block(features , features, name="dec1")
        self.H_conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

        # fusion
        # self.M_H_conv1 = XNetv3_3D_re._block(features * 2, features, name="fusion1")
        # self.M_H_conv2 = XNetv3_3D_re._block(features * 4, features * 2, name="fusion2")
        # self.M_L_conv1 = XNetv3_3D_re._block(features * 2, features, name="fusion1")
        # self.M_L_conv2 = XNetv3_3D_re._block(features * 4, features * 2, name="fusion2")
        # self.M_L_conv3 = XNetv3_3D_re._block(features * 8, features * 4, name="fusion3")
        # self.M_L_conv4 = XNetv3_3D_re._block(features * 16, features * 8, name="fusion4")
        
        self.M_L_conv2 = XNetv3_3D_re._block(features * 4, features * 2, name="fusion2")
        self.M_L_conv3 = XNetv3_3D_re._block(features * 8, features * 4, name="fusion3")
        
        self.H_M_conv2 = XNetv3_3D_re._block(features * 4, features * 2, name="fusion2")
        self.H_M_conv3 = XNetv3_3D_re._block(features * 8, features * 4, name="fusion3")
        
        self.L_H_conv2 = XNetv3_3D_re._block(features * 4, features * 2, name="fusion2")
        self.L_H_conv3 = XNetv3_3D_re._block(features * 8, features * 4, name="fusion3")

    def forward(self, x_main, x_L, x_HA,x_HR):
        
        # x_L_re = x_L + self.lh_exchange_weight_L * x_HA
        # x_H_re = x_L + self.lh_exchange_weight_H * x_HR
        # x_L_re = 0.6 * x_L + 0.4 * x_HA
        # x_H_re = 0.6 * x_L + 0.4 * x_HR
        x_L_re=torch.cat((x_L,  x_HA), dim=1)
        x_H_re=torch.cat((x_L,  x_HR), dim=1)
    
        # x_L_re = (x_L_re -torch.mean(x_L_re)) / torch.std(x_L_re)
        # x_H_re = (x_H_re -torch.mean(x_H_re)) / torch.std(x_H_re)
        
        # Main encoder
        M_enc1 = self.M_encoder1(x_main)
        M_enc2 = self.M_encoder2(self.M_pool1(M_enc1))
        M_enc3 = self.M_encoder3(self.M_pool2(M_enc2))
        M_enc4 = self.M_encoder4(self.M_pool3(M_enc3))

        M_bottleneck = self.M_bottleneck(self.M_pool4(M_enc4))

        # L encoder
        L_enc1 = self.L_encoder1(x_L_re)
        L_enc2 = self.L_encoder2(self.L_pool1(L_enc1))
        L_enc3 = self.L_encoder3(self.L_pool2(L_enc2))
        L_enc4 = self.L_encoder4(self.L_pool3(L_enc3))

        L_bottleneck = self.L_bottleneck(self.L_pool4(L_enc4))

        # H encoder
        H_enc1 = self.H_encoder1(x_H_re)
        H_enc2 = self.H_encoder2(self.H_pool1(H_enc1))
        H_enc3 = self.H_encoder3(self.H_pool2(H_enc2))
        H_enc4 = self.H_encoder4(self.H_pool3(H_enc3))

        H_bottleneck = self.H_bottleneck(self.H_pool4(H_enc4))

        # fusion
        # M_H_enc1 = torch.cat((M_enc1, H_enc1), dim=1)
        # M_H_enc1 = self.M_H_conv1(M_H_enc1)
        # M_H_enc2 = torch.cat((M_enc2, H_enc2), dim=1)
        # M_H_enc2 = self.M_H_conv2(M_H_enc2)
        
        # M_L_enc3 = torch.cat((M_enc3, L_enc3), dim=1)
        # M_L_enc3 = self.M_L_conv3(M_L_enc3)
        # M_L_enc4 = torch.cat((M_enc4, L_enc4), dim=1)
        # M_L_enc4 = self.M_L_conv4(M_L_enc4)
        M_L_enc2 = torch.cat((M_enc2, L_enc2), dim=1)
        M_L_enc2 = self.M_L_conv2(M_L_enc2)
        M_L_enc3 = torch.cat((M_enc3, L_enc3), dim=1)
        M_L_enc3 = self.M_L_conv3(M_L_enc3)
        
        H_M_enc2 = torch.cat((H_enc2, M_enc2), dim=1)
        H_M_enc2 = self.H_M_conv2(H_M_enc2)
        H_M_enc3 = torch.cat((H_enc3, M_enc3), dim=1)
        H_M_enc3 = self.H_M_conv3(H_M_enc3)
        
        L_H_enc2 = torch.cat((L_enc2, H_enc2), dim=1)
        L_H_enc2 = self.L_H_conv2(L_H_enc2)
        L_H_enc3 = torch.cat((L_enc3, H_enc3), dim=1)
        L_H_enc3 = self.L_H_conv3(L_H_enc3)


       
        # L_H_enc3 = torch.cat((M_enc1, H_enc1), dim=1)
        # L_H_enc3 = self.M_H_conv1(M_H_enc1)
        # L_H_enc4 = torch.cat((M_enc2, H_enc2), dim=1)
        # L_H_enc4 = self.M_H_conv2(M_H_enc2)

        # Main decoder
        M_dec4 = self.M_upconv4(M_bottleneck)
        # M_dec4 = torch.cat((M_dec4, M_L_enc4), dim=1)
        M_dec4 = self.M_decoder4(M_dec4)
        M_dec3 = self.M_upconv3(M_dec4)
        M_dec3 = torch.cat((M_dec3, L_H_enc3), dim=1)
        M_dec3 = self.M_decoder3(M_dec3)
        M_dec2 = self.M_upconv2(M_dec3)
        M_dec2 = torch.cat((M_dec2, L_H_enc2), dim=1)
        M_dec2 = self.M_decoder2(M_dec2)
        M_dec1 = self.M_upconv1(M_dec2)
        # M_dec1 = torch.cat((M_dec1, M_H_enc1), dim=1)
        M_dec1 = self.M_decoder1(M_dec1)
        M_outputs = self.M_conv(M_dec1)

        # L decoder
        L_dec4 = self.L_upconv4(L_bottleneck)
        # L_dec4 = torch.cat((L_dec4, M_L_H_enc4), dim=1)
        L_dec4 = self.L_decoder4(L_dec4)
        L_dec3 = self.L_upconv3(L_dec4)
        L_dec3 = torch.cat((L_dec3, H_M_enc3), dim=1)
        L_dec3 = self.L_decoder3(L_dec3)
        L_dec2 = self.L_upconv2(L_dec3)
        L_dec2 = torch.cat((L_dec2, H_M_enc2), dim=1)
        L_dec2 = self.L_decoder2(L_dec2)
        L_dec1 = self.L_upconv1(L_dec2)
        # L_dec1 = torch.cat((L_dec1, L_enc1), dim=1)
        L_dec1 = self.L_decoder1(L_dec1)
        L_outputs = self.L_conv(L_dec1)

        # H decoder
        H_dec4 = self.H_upconv4(H_bottleneck)
        # H_dec4 = torch.cat((H_dec4, H_enc4), dim=1)
        H_dec4 = self.H_decoder4(H_dec4)
        H_dec3 = self.H_upconv3(H_dec4)
        H_dec3 = torch.cat((H_dec3, M_L_enc3), dim=1)
        H_dec3 = self.H_decoder3(H_dec3)
        H_dec2 = self.H_upconv2(H_dec3)
        H_dec2 = torch.cat((H_dec2, M_L_enc2), dim=1)
        H_dec2 = self.H_decoder2(H_dec2)
        H_dec1 = self.H_upconv1(H_dec2)
        # H_dec1 = torch.cat((H_dec1, M_H_enc1), dim=1)
        H_dec1 = self.H_decoder1(H_dec1)
        H_outputs = self.H_conv(H_dec1)

        return M_outputs, L_outputs, H_outputs

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
        


class xnetv3_3d_cat1(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, init_features=32):
        """
        Implementations based on the Unet3D paper: https://arxiv.org/abs/1606.06650
        """

        super(xnetv3_3d_cat1, self).__init__()

        features = init_features

        # main network
        self.M_encoder1 = xnetv3_3d_cat1._block(in_channels, features, name="enc1")
        self.M_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.M_encoder2 = xnetv3_3d_cat1._block(features, features * 2, name="enc2")
        self.M_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.M_encoder3 = xnetv3_3d_cat1._block(features * 2, features * 4, name="enc3")
        self.M_pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.M_encoder4 = xnetv3_3d_cat1._block(features * 4, features * 8, name="enc4")
        self.M_pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.M_bottleneck = xnetv3_3d_cat1._block(features * 8, features * 16, name="bottleneck")

        self.M_upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.M_decoder4 = xnetv3_3d_cat1._block((features * 8) * 2 , features * 8, name="dec4")
        self.M_upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.M_decoder3 = xnetv3_3d_cat1._block((features * 4) * 2, features * 4, name="dec3")
        self.M_upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.M_decoder2 = xnetv3_3d_cat1._block((features * 2) * 2, features * 2, name="dec2")
        self.M_upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.M_decoder1 = xnetv3_3d_cat1._block(features * 2, features, name="dec1")

        self.M_conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

        # L network
        self.L_encoder1 = xnetv3_3d_cat1._block(in_channels, features, name="enc1")
        self.L_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.L_encoder2 = xnetv3_3d_cat1._block(features, features * 2, name="enc2")
        self.L_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.L_encoder3 = xnetv3_3d_cat1._block(features * 2, features * 4, name="enc3")
        self.L_pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.L_encoder4 = xnetv3_3d_cat1._block(features * 4, features * 8, name="enc4")
        self.L_pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.L_bottleneck = xnetv3_3d_cat1._block(features * 8, features * 16, name="bottleneck")

        self.L_upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.L_decoder4 = xnetv3_3d_cat1._block((features * 8) * 2 , features * 8, name="dec4")
        self.L_upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.L_decoder3 = xnetv3_3d_cat1._block((features * 4) * 2, features * 4, name="dec3")
        self.L_upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.L_decoder2 = xnetv3_3d_cat1._block((features * 2), features * 2, name="dec2")
        self.L_upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.L_decoder1 = xnetv3_3d_cat1._block(features, features, name="dec1")
        self.L_conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

        # H network
        self.H_encoder1 = xnetv3_3d_cat1._block(in_channels, features, name="enc1")
        self.H_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.H_encoder2 = xnetv3_3d_cat1._block(features, features * 2, name="enc2")
        self.H_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.H_encoder3 = xnetv3_3d_cat1._block(features * 2, features * 4, name="enc3")
        self.H_pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.H_encoder4 = xnetv3_3d_cat1._block(features * 4, features * 8, name="enc4")
        self.H_pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.H_bottleneck = xnetv3_3d_cat1._block(features * 8, features * 16, name="bottleneck")

        self.H_upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.H_decoder4 = xnetv3_3d_cat1._block((features * 8), features * 8, name="dec4")
        self.H_upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.H_decoder3 = xnetv3_3d_cat1._block((features * 4), features * 4, name="dec3")
        self.H_upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.H_decoder2 = xnetv3_3d_cat1._block((features * 2) * 2, features * 2, name="dec2")
        self.H_upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.H_decoder1 = xnetv3_3d_cat1._block(features * 2, features, name="dec1")
        self.H_conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

        # fusion
        self.M_H_conv1 = xnetv3_3d_cat1._block(features * 2, features, name="fusion1")
        self.M_H_conv2 = xnetv3_3d_cat1._block(features * 4, features * 2, name="fusion2")
        self.M_L_conv3 = xnetv3_3d_cat1._block(features * 8, features * 4, name="fusion3")
        self.M_L_conv4 = xnetv3_3d_cat1._block(features * 16, features * 8, name="fusion4")
        
        # 假设这是初始化参数，实际应用中可能需要根据具体情况调整初始化方式
        self.lh_exchange_weight_L = nn.Parameter(torch.randn(1))
        self.lh_exchange_weight_H = nn.Parameter(torch.randn(1))
        # Sigmoid层用于限制参数范围
        self.sigmoid_L = nn.Sigmoid()
        self.sigmoid_H = nn.Sigmoid()

    def forward(self, x_main, x_L, x_HA,x_HR):
        # x_L_new=torch.cat((x_L,  x_HA), dim=1)
        # x_H_new=torch.cat((x_L,  x_HR), dim=1)
        
        lh_exchange_weight_L = self.sigmoid_L(self.lh_exchange_weight_L)
        lh_exchange_weight_H = self.sigmoid_H(self.lh_exchange_weight_H)
        x_L_new =(1-lh_exchange_weight_L)*x_L + lh_exchange_weight_L * x_HA
        x_H_new =(1-lh_exchange_weight_H)*x_L + lh_exchange_weight_H * x_HR
        
        
        # x_L_new=0.5*x_L+0.5*x_HA
        # x_H_new=0.5*x_L+0.5*x_HR
        # Main encoder
        M_enc1 = self.M_encoder1(x_main)
        M_enc2 = self.M_encoder2(self.M_pool1(M_enc1))
        M_enc3 = self.M_encoder3(self.M_pool2(M_enc2))
        M_enc4 = self.M_encoder4(self.M_pool3(M_enc3))

        M_bottleneck = self.M_bottleneck(self.M_pool4(M_enc4))

        # L encoder
        L_enc1 = self.L_encoder1(x_L_new)
        L_enc2 = self.L_encoder2(self.L_pool1(L_enc1))
        L_enc3 = self.L_encoder3(self.L_pool2(L_enc2))
        L_enc4 = self.L_encoder4(self.L_pool3(L_enc3))

        L_bottleneck = self.L_bottleneck(self.L_pool4(L_enc4))

        # H encoder
        H_enc1 = self.H_encoder1(x_H_new)
        H_enc2 = self.H_encoder2(self.H_pool1(H_enc1))
        H_enc3 = self.H_encoder3(self.H_pool2(H_enc2))
        H_enc4 = self.H_encoder4(self.H_pool3(H_enc3))

        H_bottleneck = self.H_bottleneck(self.H_pool4(H_enc4))

        # fusion
        M_H_enc1 = torch.cat((M_enc1, H_enc1), dim=1)
        M_H_enc1 = self.M_H_conv1(M_H_enc1)
        M_H_enc2 = torch.cat((M_enc2, H_enc2), dim=1)
        M_H_enc2 = self.M_H_conv2(M_H_enc2)
        M_L_enc3 = torch.cat((M_enc3, L_enc3), dim=1)
        M_L_enc3 = self.M_L_conv3(M_L_enc3)
        M_L_enc4 = torch.cat((M_enc4, L_enc4), dim=1)
        M_L_enc4 = self.M_L_conv4(M_L_enc4)

        # Main decoder
        M_dec4 = self.M_upconv4(M_bottleneck)
        M_dec4 = torch.cat((M_dec4, M_L_enc4), dim=1)
        M_dec4 = self.M_decoder4(M_dec4)
        
        M_dec3 = self.M_upconv3(M_dec4)
        M_dec3 = torch.cat((M_dec3, M_L_enc3), dim=1)
        M_dec3 = self.M_decoder3(M_dec3)
        
        M_dec2 = self.M_upconv2(M_dec3)
        M_dec2 = torch.cat((M_dec2, M_H_enc2), dim=1)
        M_dec2 = self.M_decoder2(M_dec2)
        
        M_dec1 = self.M_upconv1(M_dec2)
        M_dec1 = torch.cat((M_dec1, M_H_enc1), dim=1)
        M_dec1 = self.M_decoder1(M_dec1)
        M_outputs = self.M_conv(M_dec1)

        # L decoder
        L_dec4 = self.L_upconv4(L_bottleneck)
        L_dec4 = torch.cat((L_dec4, M_L_enc4), dim=1)
        L_dec4 = self.L_decoder4(L_dec4)
        
        L_dec3 = self.L_upconv3(L_dec4)
        L_dec3 = torch.cat((L_dec3, M_L_enc3), dim=1)
        L_dec3 = self.L_decoder3(L_dec3)
        
        L_dec2 = self.L_upconv2(L_dec3)
        # L_dec2 = torch.cat((L_dec2, L_enc2), dim=1)
        L_dec2 = self.L_decoder2(L_dec2)
        
        L_dec1 = self.L_upconv1(L_dec2)
        # L_dec1 = torch.cat((L_dec1, L_enc1), dim=1)
        L_dec1 = self.L_decoder1(L_dec1)
        L_outputs = self.L_conv(L_dec1)

        # H decoder
        H_dec4 = self.H_upconv4(H_bottleneck)
        # H_dec4 = torch.cat((H_dec4, H_enc4), dim=1)
        H_dec4 = self.H_decoder4(H_dec4)
        
        H_dec3 = self.H_upconv3(H_dec4)
        # H_dec3 = torch.cat((H_dec3, H_enc3), dim=1)
        H_dec3 = self.H_decoder3(H_dec3)
        
        H_dec2 = self.H_upconv2(H_dec3)
        H_dec2 = torch.cat((H_dec2, M_H_enc2), dim=1)
        H_dec2 = self.H_decoder2(H_dec2)
        
        H_dec1 = self.H_upconv1(H_dec2)
        H_dec1 = torch.cat((H_dec1, M_H_enc1), dim=1)
        H_dec1 = self.H_decoder1(H_dec1)
        H_outputs = self.H_conv(H_dec1)

        return M_outputs, L_outputs, H_outputs

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


class xnetv3_3d_add1(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, init_features=32):
        """
        Implementations based on the Unet3D paper: https://arxiv.org/abs/1606.06650
        """

        super(xnetv3_3d_add1, self).__init__()

        features = init_features

        # main network
        self.M_encoder1 = xnetv3_3d_add1._block(in_channels, features, name="enc1")
        self.M_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.M_encoder2 = xnetv3_3d_add1._block(features, features * 2, name="enc2")
        self.M_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.M_encoder3 = xnetv3_3d_add1._block(features * 2, features * 4, name="enc3")
        self.M_pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.M_encoder4 = xnetv3_3d_add1._block(features * 4, features * 8, name="enc4")
        self.M_pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.M_bottleneck = xnetv3_3d_add1._block(features * 8, features * 16, name="bottleneck")

        self.M_upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.M_decoder4 = xnetv3_3d_add1._block((features * 8) * 2 , features * 8, name="dec4")
        self.M_upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.M_decoder3 = xnetv3_3d_add1._block((features * 4) * 2, features * 4, name="dec3")
        self.M_upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.M_decoder2 = xnetv3_3d_add1._block((features * 2) * 2, features * 2, name="dec2")
        self.M_upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.M_decoder1 = xnetv3_3d_add1._block(features * 2, features, name="dec1")

        self.M_conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

        # L network
        self.L_encoder1 = xnetv3_3d_add1._block(in_channels, features, name="enc1")
        self.L_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.L_encoder2 = xnetv3_3d_add1._block(features, features * 2, name="enc2")
        self.L_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.L_encoder3 = xnetv3_3d_add1._block(features * 2, features * 4, name="enc3")
        self.L_pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.L_encoder4 = xnetv3_3d_add1._block(features * 4, features * 8, name="enc4")
        self.L_pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.L_bottleneck = xnetv3_3d_add1._block(features * 8, features * 16, name="bottleneck")

        self.L_upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.L_decoder4 = xnetv3_3d_add1._block((features * 8), features * 8, name="dec4")
        self.L_upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.L_decoder3 = xnetv3_3d_add1._block((features * 4) , features * 4, name="dec3")
        self.L_upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.L_decoder2 = xnetv3_3d_add1._block((features * 2)* 2, features * 2, name="dec2")
        self.L_upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.L_decoder1 = xnetv3_3d_add1._block(features* 2, features, name="dec1")
        self.L_conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

        # H network
        self.H_encoder1 = xnetv3_3d_add1._block(in_channels, features, name="enc1")
        self.H_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.H_encoder2 = xnetv3_3d_add1._block(features, features * 2, name="enc2")
        self.H_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.H_encoder3 = xnetv3_3d_add1._block(features * 2, features * 4, name="enc3")
        self.H_pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.H_encoder4 = xnetv3_3d_add1._block(features * 4, features * 8, name="enc4")
        self.H_pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.H_bottleneck = xnetv3_3d_add1._block(features * 8, features * 16, name="bottleneck")

        self.H_upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.H_decoder4 = xnetv3_3d_add1._block((features * 8)* 2, features * 8, name="dec4")
        self.H_upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.H_decoder3 = xnetv3_3d_add1._block((features * 4)* 2, features * 4, name="dec3")
        self.H_upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.H_decoder2 = xnetv3_3d_add1._block((features * 2), features * 2, name="dec2")
        self.H_upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.H_decoder1 = xnetv3_3d_add1._block(features, features, name="dec1")
        self.H_conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

        # fusion
        self.M_H_conv1 = xnetv3_3d_add1._block(features * 2, features, name="fusion1")
        self.M_H_conv2 = xnetv3_3d_add1._block(features * 4, features * 2, name="fusion2")
        self.M_L_conv3 = xnetv3_3d_add1._block(features * 8, features * 4, name="fusion3")
        self.M_L_conv4 = xnetv3_3d_add1._block(features * 16, features * 8, name="fusion4")
        
        # 假设这是初始化参数，实际应用中可能需要根据具体情况调整初始化方式
        self.lh_exchange_weight_L = nn.Parameter(torch.randn(1))
        self.lh_exchange_weight_H = nn.Parameter(torch.randn(1))
        # Sigmoid层用于限制参数范围
        self.sigmoid_L = nn.Sigmoid()
        self.sigmoid_H = nn.Sigmoid()

    def forward(self, x_main, x_L, x_HA,x_HR):
        # x_L_new=torch.cat((x_L,  x_HA), dim=1)
        # x_H_new=torch.cat((x_L,  x_HR), dim=1)
        
        lh_exchange_weight_L = self.sigmoid_L(self.lh_exchange_weight_L)
        lh_exchange_weight_H = self.sigmoid_H(self.lh_exchange_weight_H)
        x_L_new =(1-lh_exchange_weight_L)*x_L + lh_exchange_weight_L * x_HA
        x_H_new =(1-lh_exchange_weight_H)*x_L + lh_exchange_weight_H * x_HR
        
        
        # x_L_new=0.5*x_L+0.5*x_HA
        # x_H_new=0.5*x_L+0.5*x_HR
        # Main encoder
        M_enc1 = self.M_encoder1(x_main)
        M_enc2 = self.M_encoder2(self.M_pool1(M_enc1))
        M_enc3 = self.M_encoder3(self.M_pool2(M_enc2))
        M_enc4 = self.M_encoder4(self.M_pool3(M_enc3))

        M_bottleneck = self.M_bottleneck(self.M_pool4(M_enc4))

        # L encoder
        L_enc1 = self.L_encoder1(x_L_new)
        L_enc2 = self.L_encoder2(self.L_pool1(L_enc1))
        L_enc3 = self.L_encoder3(self.L_pool2(L_enc2))
        L_enc4 = self.L_encoder4(self.L_pool3(L_enc3))

        L_bottleneck = self.L_bottleneck(self.L_pool4(L_enc4))

        # H encoder
        H_enc1 = self.H_encoder1(x_H_new)
        H_enc2 = self.H_encoder2(self.H_pool1(H_enc1))
        H_enc3 = self.H_encoder3(self.H_pool2(H_enc2))
        H_enc4 = self.H_encoder4(self.H_pool3(H_enc3))

        H_bottleneck = self.H_bottleneck(self.H_pool4(H_enc4))

        # fusion
        M_H_enc1 = torch.cat((M_enc1, H_enc1), dim=1)
        M_H_enc1 = self.M_H_conv1(M_H_enc1)
        M_H_enc2 = torch.cat((M_enc2, H_enc2), dim=1)
        M_H_enc2 = self.M_H_conv2(M_H_enc2)
        M_L_enc3 = torch.cat((M_enc3, L_enc3), dim=1)
        M_L_enc3 = self.M_L_conv3(M_L_enc3)
        M_L_enc4 = torch.cat((M_enc4, L_enc4), dim=1)
        M_L_enc4 = self.M_L_conv4(M_L_enc4)

        # Main decoder
        M_dec4 = self.M_upconv4(M_bottleneck)
        M_dec4 = torch.cat((M_dec4, M_L_enc4), dim=1)
        M_dec4 = self.M_decoder4(M_dec4)
        
        M_dec3 = self.M_upconv3(M_dec4)
        M_dec3 = torch.cat((M_dec3, M_L_enc3), dim=1)
        M_dec3 = self.M_decoder3(M_dec3)
        
        M_dec2 = self.M_upconv2(M_dec3)
        M_dec2 = torch.cat((M_dec2, M_H_enc2), dim=1)
        M_dec2 = self.M_decoder2(M_dec2)
        
        M_dec1 = self.M_upconv1(M_dec2)
        M_dec1 = torch.cat((M_dec1, M_H_enc1), dim=1)
        M_dec1 = self.M_decoder1(M_dec1)
        M_outputs = self.M_conv(M_dec1)

        # L decoder
        L_dec4 = self.L_upconv4(L_bottleneck)
        L_dec4 = torch.add((L_dec4, M_L_enc4), dim=1)
        L_dec4 = self.L_decoder4(L_dec4)
        
        L_dec3 = self.L_upconv3(L_dec4)
        L_dec3 = torch.add((L_dec3, M_L_enc3), dim=1)
        L_dec3 = self.L_decoder3(L_dec3)
        
        L_dec2 = self.L_upconv2(L_dec3)
        L_dec2 = torch.cat((L_dec2, L_enc2), dim=1)
        L_dec2 = self.L_decoder2(L_dec2)
        
        L_dec1 = self.L_upconv1(L_dec2)
        L_dec1 = torch.cat((L_dec1, L_enc1), dim=1)
        L_dec1 = self.L_decoder1(L_dec1)
        L_outputs = self.L_conv(L_dec1)

        # H decoder
        H_dec4 = self.H_upconv4(H_bottleneck)
        H_dec4 = torch.cat((H_dec4, H_enc4), dim=1)
        H_dec4 = self.H_decoder4(H_dec4)
        
        H_dec3 = self.H_upconv3(H_dec4)
        H_dec3 = torch.cat((H_dec3, H_enc3), dim=1)
        H_dec3 = self.H_decoder3(H_dec3)
        
        H_dec2 = self.H_upconv2(H_dec3)
        H_dec2 = torch.add((H_dec2, M_H_enc2), dim=1)
        H_dec2 = self.H_decoder2(H_dec2)
        
        H_dec1 = self.H_upconv1(H_dec2)
        H_dec1 = torch.add((H_dec1, M_H_enc1), dim=1)
        H_dec1 = self.H_decoder1(H_dec1)
        H_outputs = self.H_conv(H_dec1)

        return M_outputs, L_outputs, H_outputs

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



class xnetv3_3d_add0(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, init_features=32):
        """
        Implementations based on the Unet3D paper: https://arxiv.org/abs/1606.06650
        """

        super(xnetv3_3d_add0, self).__init__()

        features = init_features

        # main network
        self.M_encoder1 = xnetv3_3d_add0._block(in_channels, features, name="enc1")
        self.M_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.M_encoder2 = xnetv3_3d_add0._block(features, features * 2, name="enc2")
        self.M_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.M_encoder3 = xnetv3_3d_add0._block(features * 2, features * 4, name="enc3")
        self.M_pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.M_encoder4 = xnetv3_3d_add0._block(features * 4, features * 8, name="enc4")
        self.M_pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.M_bottleneck = xnetv3_3d_add0._block(features * 8, features * 16, name="bottleneck")

        self.M_upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.M_decoder4 = xnetv3_3d_add0._block((features * 8) * 2 , features * 8, name="dec4")
        self.M_upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.M_decoder3 = xnetv3_3d_add0._block((features * 4) * 2, features * 4, name="dec3")
        self.M_upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.M_decoder2 = xnetv3_3d_add0._block((features * 2) * 2, features * 2, name="dec2")
        self.M_upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.M_decoder1 = xnetv3_3d_add0._block(features * 2, features, name="dec1")

        self.M_conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

        # L network
        self.L_encoder1 = xnetv3_3d_add0._block(in_channels, features, name="enc1")
        self.L_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.L_encoder2 = xnetv3_3d_add0._block(features, features * 2, name="enc2")
        self.L_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.L_encoder3 = xnetv3_3d_add0._block(features * 2, features * 4, name="enc3")
        self.L_pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.L_encoder4 = xnetv3_3d_add0._block(features * 4, features * 8, name="enc4")
        self.L_pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.L_bottleneck = xnetv3_3d_add0._block(features * 8, features * 16, name="bottleneck")

        self.L_upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.L_decoder4 = xnetv3_3d_add0._block((features * 8)  , features * 8, name="dec4")
        self.L_upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.L_decoder3 = xnetv3_3d_add0._block((features * 4), features * 4, name="dec3")
        self.L_upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.L_decoder2 = xnetv3_3d_add0._block((features * 2), features * 2, name="dec2")
        self.L_upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.L_decoder1 = xnetv3_3d_add0._block(features, features, name="dec1")
        self.L_conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

        # H network
        self.H_encoder1 = xnetv3_3d_add0._block(in_channels, features, name="enc1")
        self.H_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.H_encoder2 = xnetv3_3d_add0._block(features, features * 2, name="enc2")
        self.H_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.H_encoder3 = xnetv3_3d_add0._block(features * 2, features * 4, name="enc3")
        self.H_pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.H_encoder4 = xnetv3_3d_add0._block(features * 4, features * 8, name="enc4")
        self.H_pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.H_bottleneck = xnetv3_3d_add0._block(features * 8, features * 16, name="bottleneck")

        self.H_upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.H_decoder4 = xnetv3_3d_add0._block((features * 8), features * 8, name="dec4")
        self.H_upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.H_decoder3 = xnetv3_3d_add0._block((features * 4), features * 4, name="dec3")
        self.H_upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.H_decoder2 = xnetv3_3d_add0._block((features * 2), features * 2, name="dec2")
        self.H_upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.H_decoder1 = xnetv3_3d_add0._block(features , features, name="dec1")
        self.H_conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

        # fusion
        self.M_H_conv1 = xnetv3_3d_add0._block(features * 2, features, name="fusion1")
        self.M_H_conv2 = xnetv3_3d_add0._block(features * 4, features * 2, name="fusion2")
        self.M_L_conv3 = xnetv3_3d_add0._block(features * 8, features * 4, name="fusion3")
        self.M_L_conv4 = xnetv3_3d_add0._block(features * 16, features * 8, name="fusion4")
        
        # 假设这是初始化参数，实际应用中可能需要根据具体情况调整初始化方式
        self.lh_exchange_weight_L = nn.Parameter(torch.randn(1))
        self.lh_exchange_weight_H = nn.Parameter(torch.randn(1))
        # Sigmoid层用于限制参数范围
        self.sigmoid_L = nn.Sigmoid()
        self.sigmoid_H = nn.Sigmoid()

    def forward(self, x_main, x_L, x_HA,x_HR):
        # x_L_new=torch.cat((x_L,  x_HA), dim=1)
        # x_H_new=torch.cat((x_L,  x_HR), dim=1)
        
        lh_exchange_weight_L = self.sigmoid_L(self.lh_exchange_weight_L)
        lh_exchange_weight_H = self.sigmoid_H(self.lh_exchange_weight_H)
        x_L_new =(1-lh_exchange_weight_L)*x_L + lh_exchange_weight_L * x_HA
        x_H_new =(1-lh_exchange_weight_H)*x_L + lh_exchange_weight_H * x_HR
        
        
        # x_L_new=0.5*x_L+0.5*x_HA
        # x_H_new=0.5*x_L+0.5*x_HR
        # Main encoder
        M_enc1 = self.M_encoder1(x_main)
        M_enc2 = self.M_encoder2(self.M_pool1(M_enc1))
        M_enc3 = self.M_encoder3(self.M_pool2(M_enc2))
        M_enc4 = self.M_encoder4(self.M_pool3(M_enc3))

        M_bottleneck = self.M_bottleneck(self.M_pool4(M_enc4))

        # L encoder
        L_enc1 = self.L_encoder1(x_L_new)
        L_enc2 = self.L_encoder2(self.L_pool1(L_enc1))
        L_enc3 = self.L_encoder3(self.L_pool2(L_enc2))
        L_enc4 = self.L_encoder4(self.L_pool3(L_enc3))

        L_bottleneck = self.L_bottleneck(self.L_pool4(L_enc4))

        # H encoder
        H_enc1 = self.H_encoder1(x_H_new)
        H_enc2 = self.H_encoder2(self.H_pool1(H_enc1))
        H_enc3 = self.H_encoder3(self.H_pool2(H_enc2))
        H_enc4 = self.H_encoder4(self.H_pool3(H_enc3))

        H_bottleneck = self.H_bottleneck(self.H_pool4(H_enc4))

        # fusion
        M_H_enc1 = torch.cat((M_enc1, H_enc1), dim=1)
        M_H_enc1 = self.M_H_conv1(M_H_enc1)
        M_H_enc2 = torch.cat((M_enc2, H_enc2), dim=1)
        M_H_enc2 = self.M_H_conv2(M_H_enc2)
        M_L_enc3 = torch.cat((M_enc3, L_enc3), dim=1)
        M_L_enc3 = self.M_L_conv3(M_L_enc3)
        M_L_enc4 = torch.cat((M_enc4, L_enc4), dim=1)
        M_L_enc4 = self.M_L_conv4(M_L_enc4)

        # Main decoder
        M_dec4 = self.M_upconv4(M_bottleneck)
        M_dec4 = torch.cat((M_dec4, M_L_enc4), dim=1)
        M_dec4 = self.M_decoder4(M_dec4)
        
        M_dec3 = self.M_upconv3(M_dec4)
        M_dec3 = torch.cat((M_dec3, M_L_enc3), dim=1)
        M_dec3 = self.M_decoder3(M_dec3)
        
        M_dec2 = self.M_upconv2(M_dec3)
        M_dec2 = torch.cat((M_dec2, M_H_enc2), dim=1)
        M_dec2 = self.M_decoder2(M_dec2)
        
        M_dec1 = self.M_upconv1(M_dec2)
        M_dec1 = torch.cat((M_dec1, M_H_enc1), dim=1)
        M_dec1 = self.M_decoder1(M_dec1)
        M_outputs = self.M_conv(M_dec1)

        # L decoder
        L_dec4 = self.L_upconv4(L_bottleneck)
        L_dec4 = torch.add((L_dec4, M_L_enc4), dim=1)
        L_dec4 = self.L_decoder4(L_dec4)
        
        L_dec3 = self.L_upconv3(L_dec4)
        L_dec3 = torch.add((L_dec3, M_L_enc3), dim=1)
        L_dec3 = self.L_decoder3(L_dec3)
        
        L_dec2 = self.L_upconv2(L_dec3)
        # L_dec2 = torch.cat((L_dec2, L_enc2), dim=1)
        L_dec2 = self.L_decoder2(L_dec2)
        
        L_dec1 = self.L_upconv1(L_dec2)
        # L_dec1 = torch.cat((L_dec1, L_enc1), dim=1)
        L_dec1 = self.L_decoder1(L_dec1)
        L_outputs = self.L_conv(L_dec1)

        # H decoder
        H_dec4 = self.H_upconv4(H_bottleneck)
        # H_dec4 = torch.cat((H_dec4, H_enc4), dim=1)
        H_dec4 = self.H_decoder4(H_dec4)
        
        H_dec3 = self.H_upconv3(H_dec4)
        # H_dec3 = torch.cat((H_dec3, H_enc3), dim=1)
        H_dec3 = self.H_decoder3(H_dec3)
        
        H_dec2 = self.H_upconv2(H_dec3)
        H_dec2 = torch.add((H_dec2, M_H_enc2), dim=1)
        H_dec2 = self.H_decoder2(H_dec2)
        
        H_dec1 = self.H_upconv1(H_dec2)
        H_dec1 = torch.add((H_dec1, M_H_enc1), dim=1)
        H_dec1 = self.H_decoder1(H_dec1)
        H_outputs = self.H_conv(H_dec1)

        return M_outputs, L_outputs, H_outputs

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


def xnetv3_3d(in_channels, num_classes):
    model = XNetv3_3D(in_channels, num_classes)
    init_weights(model, 'kaiming')
    return model

def xnetv3_3d_min(in_channels, num_classes):
    model = XNetv3_3D_min(in_channels, num_classes)
    init_weights(model, 'kaiming')
    return model

def xnetv3_3d_right(in_channels, num_classes):
    model = XNetv3_3D_right(in_channels, num_classes)
    init_weights(model, 'kaiming')
    return model

def xnetv3_3d_new(in_channels, num_classes):
    model = XNetv3_3D_new(in_channels, num_classes)
    init_weights(model, 'kaiming')
    return model


def xnetv3_3d_xh(in_channels, num_classes):
    model = XNetv3_3D_xh(in_channels, num_classes)
    init_weights(model, 'kaiming')
    return model

def xnetv3_3d_jh(in_channels, num_classes):
    model = XNetv3_3D_jh(in_channels, num_classes)
    init_weights(model, 'kaiming')
    return model


def xnetv3_3d_re(in_channels, num_classes):
    model = XNetv3_3D_re(in_channels, num_classes)
    init_weights(model, 'kaiming')
    return model



def xnetv3_3d_Add(in_channels, num_classes):
    model = XNetv3_3D_Add(in_channels, num_classes)
    init_weights(model, 'kaiming')
    return model
def xnetv3_3d_add1(in_channels, num_classes):
    model = xnetv3_3d_add1(in_channels, num_classes)
    init_weights(model, 'kaiming')
    return model
def xnetv3_3d_add0(in_channels, num_classes):
    model = xnetv3_3d_add0(in_channels, num_classes)
    init_weights(model, 'kaiming')
    return model

def xnetv3_3d_cat1(in_channels, num_classes):
    model = xnetv3_3d_cat1(in_channels, num_classes)
    init_weights(model, 'kaiming')
    return model


# if __name__ == '__main__':
#     model = xnetv2_3d_min(1,10)
#     model.eval()
#     input1 = torch.rand(2, 1, 128, 128, 128)
#     input2 = torch.rand(2, 1, 128, 128, 128)
#     input3 = torch.rand(2, 1, 128, 128, 128)
#     output1, output2, output3 = model(input1, input2, input3)
#     output1 = output1.data.cpu().numpy()
#     # print(output)
#     print(output1.shape)
