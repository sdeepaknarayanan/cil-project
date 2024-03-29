"""
A part of the code was borrowed from:
https://github.com/nikhilroxtomar/Semantic-Segmentation-Architecture/blob/main/PyTorch/resunet.py
"""
import torch
import torch.nn as nn

class batchnorm_relu(nn.Module):
    def __init__(self, in_c):
        super().__init__()

        self.bn = nn.BatchNorm2d(in_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.bn(inputs)
        x = self.relu(x)
        return x

class residual_block(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()

        """ Convolutional layer """
        self.b1 = batchnorm_relu(in_c)
        self.c1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=stride)
        self.b2 = batchnorm_relu(out_c)
        self.c2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, stride=1)

        """ Shortcut Connection (Identity Mapping) """
        self.s = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0, stride=stride)

    def forward(self, inputs):
        x = self.b1(inputs)
        x = self.c1(x)
        x = self.b2(x)
        x = self.c2(x)
        s = self.s(inputs)

        skip = x + s
        return skip

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(in_c, out_c,2,2)

        # Transposed Convs
        self.r = residual_block(2*out_c, out_c)

    def forward(self, inputs, skip):
        x = self.upsample(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.r(x)
        return x



class build_resunet_symmetric(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoder 1 """
        self.c11 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.br1 = batchnorm_relu(32)
        self.c12 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.c13 = nn.Conv2d(3, 32, kernel_size=1, padding=0)

        """ Encoder 2 and 3 """
        self.r2 = residual_block(32, 64, stride=2) # 200
        self.r3 = residual_block(64, 128, stride=2) # 100
        self.r4 = residual_block(128, 256, stride=2) # 50 

        """ Bridge """
        self.bridge = residual_block(256, 512, stride=2) #25

        """ Decoder """
        self.d1 = decoder_block(512, 256)
        self.d2 = decoder_block(256, 128)
        self.d3 = decoder_block(128, 64)
        self.d4 = decoder_block(64, 32)
        

        """ Output """
        self.output = nn.Conv2d(32, 1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        """ Encoder 1 """
        x = self.c11(inputs)
        x = self.br1(x)
        x = self.c12(x)
        s = self.c13(inputs)
        skip1 = x + s
        # (1, 32, 400, 400)

        """ Encoder 2 and 3 """
        skip2 = self.r2(skip1)
        skip3 = self.r3(skip2)
        skip4 = self.r4(skip3)


        """ Bridge """
        b = self.bridge(skip4)

        """ Decoder """
        d1 = self.d1(b, skip4)
        d2 = self.d2(d1, skip3)
        d3 = self.d3(d2, skip2)
        d4 = self.d4(d3, skip1)
        """ output """
        output = self.output(d4)

        return output
