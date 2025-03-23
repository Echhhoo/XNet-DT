import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn import init
from thop import profile
from torchinfo import summary


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
        # x_L_new=x_L
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

class XNetv3_3D_1cat(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, init_features=32):
        """
        Implementations based on the Unet3D paper: https://arxiv.org/abs/1606.06650
        """

        super(XNetv3_3D_1cat, self).__init__()

        features = init_features

        # main network
        self.M_encoder1 =XNetv3_3D_1cat._block(in_channels, features, name="enc1")
        self.M_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.M_encoder2 =XNetv3_3D_1cat._block(features, features * 2, name="enc2")
        self.M_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.M_encoder3 =XNetv3_3D_1cat._block(features * 2, features * 4, name="enc3")
        self.M_pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.M_encoder4 =XNetv3_3D_1cat._block(features * 4, features * 8, name="enc4")
        self.M_pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.M_bottleneck =XNetv3_3D_1cat._block(features * 8, features * 16, name="bottleneck")

        self.M_upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.M_decoder4 =XNetv3_3D_1cat._block((features * 8) * 2 , features * 8, name="dec4")
        self.M_upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.M_decoder3 =XNetv3_3D_1cat._block((features * 4) * 2, features * 4, name="dec3")
        self.M_upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.M_decoder2 =XNetv3_3D_1cat._block((features * 2) * 2, features * 2, name="dec2")
        self.M_upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.M_decoder1 =XNetv3_3D_1cat._block(features * 2, features, name="dec1")

        self.M_conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

        # L network
        self.L_encoder1 =XNetv3_3D_1cat._block(in_channels*2, features, name="enc1")
        self.L_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.L_encoder2 =XNetv3_3D_1cat._block(features, features * 2, name="enc2")
        self.L_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.L_encoder3 =XNetv3_3D_1cat._block(features * 2, features * 4, name="enc3")
        self.L_pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.L_encoder4 =XNetv3_3D_1cat._block(features * 4, features * 8, name="enc4")
        self.L_pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.L_bottleneck =XNetv3_3D_1cat._block(features * 8, features * 16, name="bottleneck")

        self.L_upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.L_decoder4 =XNetv3_3D_1cat._block((features * 8) * 2 , features * 8, name="dec4")
        self.L_upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.L_decoder3 =XNetv3_3D_1cat._block((features * 4) * 2, features * 4, name="dec3")
        self.L_upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.L_decoder2 =XNetv3_3D_1cat._block((features * 2)* 2, features * 2, name="dec2")
        self.L_upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.L_decoder1 =XNetv3_3D_1cat._block(features* 2, features, name="dec1")
        self.L_conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

        # H network
        self.H_encoder1 =XNetv3_3D_1cat._block(in_channels*2, features, name="enc1")
        self.H_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.H_encoder2 =XNetv3_3D_1cat._block(features, features * 2, name="enc2")
        self.H_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.H_encoder3 =XNetv3_3D_1cat._block(features * 2, features * 4, name="enc3")
        self.H_pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.H_encoder4 =XNetv3_3D_1cat._block(features * 4, features * 8, name="enc4")
        self.H_pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.H_bottleneck =XNetv3_3D_1cat._block(features * 8, features * 16, name="bottleneck")

        self.H_upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.H_decoder4 =XNetv3_3D_1cat._block((features * 8)* 2, features * 8, name="dec4")
        self.H_upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.H_decoder3 =XNetv3_3D_1cat._block((features * 4)* 2, features * 4, name="dec3")
        self.H_upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.H_decoder2 =XNetv3_3D_1cat._block((features * 2) * 2, features * 2, name="dec2")
        self.H_upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.H_decoder1 =XNetv3_3D_1cat._block(features * 2, features, name="dec1")
        self.H_conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

        # fusion
        self.M_H_conv1 =XNetv3_3D_1cat._block(features * 2, features, name="fusion1")
        self.M_H_conv2 =XNetv3_3D_1cat._block(features * 4, features * 2, name="fusion2")
        self.M_L_conv3 =XNetv3_3D_1cat._block(features * 8, features * 4, name="fusion3")
        self.M_L_conv4 =XNetv3_3D_1cat._block(features * 16, features * 8, name="fusion4")
        
        # 假设这是初始化参数，实际应用中可能需要根据具体情况调整初始化方式
        # self.lh_exchange_weight_L = nn.Parameter(torch.randn(1))
        # self.lh_exchange_weight_H = nn.Parameter(torch.randn(1))
        # Sigmoid层用于限制参数范围
        # self.sigmoid_L = nn.Sigmoid()
        # self.sigmoid_H = nn.Sigmoid()

    def forward(self, x_main, x_L, x_HA,x_HR):
        # x_L_new=torch.cat((x_L,  x_HA), dim=1)
        # x_H_new=torch.cat((x_L,  x_HR), dim=1)
        
        # lh_exchange_weight_L = self.sigmoid_L(self.lh_exchange_weight_L)
        # lh_exchange_weight_H = self.sigmoid_H(self.lh_exchange_weight_H)
        # x_L_new=x_L
        # x_L_new =(1-lh_exchange_weight_L)*x_L + lh_exchange_weight_L * x_HA
        # x_H_new =(1-lh_exchange_weight_H)*x_L + lh_exchange_weight_H * x_HR
        x_L_new =torch.cat((x_L , x_HA), dim=1)
        x_H_new =torch.cat((x_L , x_HR), dim=1)
        
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


class XNetv3_3D_0cat(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, init_features=32):
        """
        Implementations based on the Unet3D paper: https://arxiv.org/abs/1606.06650
        """

        super(XNetv3_3D_0cat, self).__init__()

        features = init_features

        # main network
        self.M_encoder1 =XNetv3_3D_0cat._block(in_channels, features, name="enc1")
        self.M_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.M_encoder2 =XNetv3_3D_0cat._block(features, features * 2, name="enc2")
        self.M_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.M_encoder3 =XNetv3_3D_0cat._block(features * 2, features * 4, name="enc3")
        self.M_pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.M_encoder4 =XNetv3_3D_0cat._block(features * 4, features * 8, name="enc4")
        self.M_pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.M_bottleneck =XNetv3_3D_0cat._block(features * 8, features * 16, name="bottleneck")

        self.M_upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.M_decoder4 =XNetv3_3D_0cat._block((features * 8) * 2 , features * 8, name="dec4")
        self.M_upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.M_decoder3 =XNetv3_3D_0cat._block((features * 4) * 2, features * 4, name="dec3")
        self.M_upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.M_decoder2 =XNetv3_3D_0cat._block((features * 2) * 2, features * 2, name="dec2")
        self.M_upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.M_decoder1 =XNetv3_3D_0cat._block(features * 2, features, name="dec1")

        self.M_conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

        # L network
        self.L_encoder1 =XNetv3_3D_0cat._block(in_channels*2, features, name="enc1")
        self.L_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.L_encoder2 =XNetv3_3D_0cat._block(features, features * 2, name="enc2")
        self.L_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.L_encoder3 =XNetv3_3D_0cat._block(features * 2, features * 4, name="enc3")
        self.L_pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.L_encoder4 =XNetv3_3D_0cat._block(features * 4, features * 8, name="enc4")
        self.L_pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.L_bottleneck =XNetv3_3D_0cat._block(features * 8, features * 16, name="bottleneck")

        self.L_upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.L_decoder4 =XNetv3_3D_0cat._block((features * 8) * 2 , features * 8, name="dec4")
        self.L_upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.L_decoder3 =XNetv3_3D_0cat._block((features * 4) * 2, features * 4, name="dec3")
        self.L_upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.L_decoder2 =XNetv3_3D_0cat._block((features * 2), features * 2, name="dec2")
        self.L_upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.L_decoder1 =XNetv3_3D_0cat._block(features, features, name="dec1")
        self.L_conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

        # H network
        self.H_encoder1 =XNetv3_3D_0cat._block(in_channels*2, features, name="enc1")
        self.H_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.H_encoder2 =XNetv3_3D_0cat._block(features, features * 2, name="enc2")
        self.H_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.H_encoder3 =XNetv3_3D_0cat._block(features * 2, features * 4, name="enc3")
        self.H_pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.H_encoder4 =XNetv3_3D_0cat._block(features * 4, features * 8, name="enc4")
        self.H_pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.H_bottleneck =XNetv3_3D_0cat._block(features * 8, features * 16, name="bottleneck")

        self.H_upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.H_decoder4 =XNetv3_3D_0cat._block((features * 8), features * 8, name="dec4")
        self.H_upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.H_decoder3 =XNetv3_3D_0cat._block((features * 4), features * 4, name="dec3")
        self.H_upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.H_decoder2 =XNetv3_3D_0cat._block((features * 2) * 2, features * 2, name="dec2")
        self.H_upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.H_decoder1 =XNetv3_3D_0cat._block(features * 2, features, name="dec1")
        self.H_conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

        # fusion
        self.M_H_conv1 =XNetv3_3D_0cat._block(features * 2, features, name="fusion1")
        self.M_H_conv2 =XNetv3_3D_0cat._block(features * 4, features * 2, name="fusion2")
        self.M_L_conv3 =XNetv3_3D_0cat._block(features * 8, features * 4, name="fusion3")
        self.M_L_conv4 =XNetv3_3D_0cat._block(features * 16, features * 8, name="fusion4")
        
        # 假设这是初始化参数，实际应用中可能需要根据具体情况调整初始化方式
        # self.lh_exchange_weight_L = nn.Parameter(torch.randn(1))
        # self.lh_exchange_weight_H = nn.Parameter(torch.randn(1))
        # # Sigmoid层用于限制参数范围
        # self.sigmoid_L = nn.Sigmoid()
        # self.sigmoid_H = nn.Sigmoid()

    def forward(self, x_main, x_L, x_HA,x_HR):
        # x_L_new=torch.cat((x_L,  x_HA), dim=1)
        # x_H_new=torch.cat((x_L,  x_HR), dim=1)
        
        # lh_exchange_weight_L = self.sigmoid_L(self.lh_exchange_weight_L)
        # lh_exchange_weight_H = self.sigmoid_H(self.lh_exchange_weight_H)
        # x_L_new =(1-lh_exchange_weight_L)*x_L + lh_exchange_weight_L * x_HA
        # x_H_new =(1-lh_exchange_weight_H)*x_L + lh_exchange_weight_H * x_HR
        x_L_new =torch.cat((x_L , x_HA), dim=1)
        x_H_new =torch.cat((x_L , x_HR), dim=1)
        
        
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
        L_dec4 = torch.add(L_dec4, M_L_enc4)
        L_dec4 = self.L_decoder4(L_dec4)
        
        L_dec3 = self.L_upconv3(L_dec4)
        L_dec3 = torch.add(L_dec3, M_L_enc3)
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
        H_dec2 = torch.add(H_dec2, M_H_enc2)
        H_dec2 = self.H_decoder2(H_dec2)
        
        H_dec1 = self.H_upconv1(H_dec2)
        H_dec1 = torch.add(H_dec1, M_H_enc1)
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
        L_dec4 = torch.add(L_dec4, M_L_enc4)
        L_dec4 = self.L_decoder4(L_dec4)
        
        L_dec3 = self.L_upconv3(L_dec4)
        L_dec3 = torch.add(L_dec3, M_L_enc3)
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
        H_dec2 = torch.add(H_dec2, M_H_enc2)
        H_dec2 = self.H_decoder2(H_dec2)
        
        H_dec1 = self.H_upconv1(H_dec2)
        H_dec1 = torch.add(H_dec1, M_H_enc1)
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


def xnetv3_3d_min(in_channels, num_classes):
    model = XNetv3_3D_min(in_channels, num_classes)
    init_weights(model, 'kaiming')
    return model


def xnetv3_3d_Add(in_channels, num_classes):
    model = XNetv3_3D_Add(in_channels, num_classes)
    init_weights(model, 'kaiming')
    return model
def xnetv3_3D_add1(in_channels, num_classes):
    model = xnetv3_3d_add1(in_channels, num_classes)
    init_weights(model, 'kaiming')
    return model
def xnetv3_3D_add0(in_channels, num_classes):
    model = xnetv3_3d_add0(in_channels, num_classes)
    init_weights(model, 'kaiming')
    return model
def xnetv3_3D_cat1(in_channels, num_classes):
    model = xnetv3_3d_cat1(in_channels, num_classes)
    init_weights(model, 'kaiming')
    return model
def xnetv3_3d_0cat(in_channels, num_classes):
    model = XNetv3_3D_0cat(in_channels, num_classes)
    init_weights(model, 'kaiming')
    return model
def xnetv3_3d_1cat(in_channels, num_classes):
    model = XNetv3_3D_1cat(in_channels, num_classes)
    init_weights(model, 'kaiming')
    return model

# if __name__ == '__main__':
#     model = xnetv3_3d_Add(1,2)
#     model.eval()
#     input1 = torch.rand(2, 1, 128, 128, 128)
#     input2 = torch.rand(2, 1, 128, 128, 128)
#     input3 = torch.rand(2, 1, 128, 128, 128)
#     input4 = torch.rand(2, 1, 128, 128, 128)
#     macs, params = profile(model, inputs=(input1,input2,input3,input4))
#     print("MACs:", macs/1000000000," GFLOP")
#     # output1, output2, output3 = model(input1)
#     # output1 = output1.data.cpu().numpy()
#     # print(output)
#     # print(output1.shape)

if __name__ == '__main__':
    # compute FLOPS & PARAMETERS
    from thop import profile
    from thop import clever_format

    model = xnetv3_3d_Add(in_channels=1, num_classes=2)
    summary(model, [(2, 1, 128, 128, 128), (2, 1, 128, 128, 128),(2, 1, 128, 128, 128), (2, 1, 128, 128, 128)], device='cpu')
    print(model)
    input = torch.randn(2, 1, 128, 128, 128).cuda()
    macs, params = profile(model.cuda(), (input,input,input,input,))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs)
    print(params)