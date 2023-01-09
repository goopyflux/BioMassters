"""Basic building blocks for ResNet architecture."""

import torch
import torch.nn as nn


class TemporalSentinelModel(nn.Module):
    """Class that implements the full end-to-end model."""
    def __init__(self, n_tsamples, input_nc, output_nc, ngf=64, use_dropout=False, n_blocks=6, padding_type='zero', opt=None):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(TemporalSentinelModel, self).__init__()

        assert n_tsamples >= 3
        self.input_nc = input_nc

        model_initial = StackedResnet2D(opt=opt, input_nc=self.input_nc)
        self.n_channels = 32 if not opt else opt.resnet_F

        model_final = [nn.Conv3d(self.input_nc, self.n_channels, kernel_size=(3, 3, 3), padding=1, bias=True), nn.ReLU(True)]
        for i in range(4):
            model_final += [ResnetBlock3D(self.n_channels, padding_type=padding_type, norm_layer='none', use_bias=True, res_scale=0.1)]

        # model_final += [ReflectionPad3D(0, 1)]
        model_final += [nn.Conv3d(self.n_channels, output_nc, kernel_size=(n_tsamples, 3, 3), padding=(0, 1, 1))]

        model_final += [nn.Identity()]

        self.model_initial = model_initial
        self.model_final = nn.Sequential(*model_final)
        self.n_tsamples = n_tsamples

    def forward(self, input):
        """Standard forward"""
        initial_output = []
        for each in input:
            initial_output.append(self.model_initial(each))
        x = torch.stack(initial_output, dim=2)
        output = self.model_final(x)
        return output.transpose(0, 1).squeeze(0)

class ReflectionPad3D(nn.Module):
    def __init__(self, pad_D, pad_HW):
        super(ReflectionPad3D, self).__init__()
        self.padder_HW = nn.ReflectionPad2d(pad_HW)
        self.padder_D = nn.ReplicationPad3d((0, 0, 0, 0, pad_D, pad_D))

    def forward(self, x):
        assert (x.size(0) == 1), f"x: {x.size()} is not 1 x C x D x H x W"  # 1 x C x D x H x W
        y = x.squeeze(0).transpose(0, 1)  # D x C x H x W
        y = self.padder_HW(y)
        y = y.transpose(0, 1).unsqueeze(0)  # 1 x C x D x H x W
        return self.padder_D(y)

class BatchNorm3D(nn.Module):
    def __init__(self, num_features):
        super(BatchNorm3D, self).__init__()
        self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        assert (x.size(0) == 1), f"x: {x.size()} is not 1 x C x D x H x W"  # 1 x C x D x H x W
        y = x.squeeze(0).transpose(0, 1).contiguous()  # D x C x H x W
        y = self.bn(y)
        y = y.transpose(0, 1).unsqueeze(0)
        return y

class ResnetBlock3D(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_bias, res_scale = 1.0, late_relu=True):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock3D, self).__init__()
        self.res_scale = res_scale
        self.late_relu = late_relu
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, possibly normalisation layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [ReflectionPad3D(1, 1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        if norm_layer == 'BatchNorm3D':
            conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias), BatchNorm3D(dim), nn.ReLU(True)]
        else:
            conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias), nn.ReLU(True)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [ReflectionPad3D(1, 1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        if norm_layer == 'BatchNorm3D':
            conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias), BatchNorm3D(dim)]
        else:
            conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]

        if self.late_relu:
            self.block_output_relu = nn.ReLU(True)
        else:
            conv_block += [nn.ReLU(True)]


        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.res_scale * self.conv_block(x)  # add skip connections
        if self.late_relu: out = self.block_output_relu(out)
        return out


class StackedResnet2D(nn.Module):
    def __init__(self, input_nc, opt=None):
        super(StackedResnet2D, self).__init__()
        
        # architecture parameters
        self.F           = 32 if not opt else opt.resnet_F
        self.B           = 8 if not opt else opt.resnet_B
        self.kernel_size = 3
        self.padding_size= 1
        self.scale_res   = 0.1
        self.dropout     = True
        self.use_64C     = False # rather removing these layers in networks_branched.py
        self.use_SAR     = False if not opt else opt.include_S1
        self.use_long	 = False

        model = [nn.Conv2d(self.use_SAR*2+input_nc, self.F, kernel_size=self.kernel_size, padding=self.padding_size, bias=True), nn.ReLU(True)]
        # generate a given number of blocks
        for i in range(self.B):
            model += [ResnetBlock2D(self.F, use_dropout=self.dropout, use_bias=True,
                                  res_scale=self.scale_res, padding_size=self.padding_size)]

        # adding in intermediate mapping layer from self.F to 64 channels for STGAN pre-training
        if self.use_64C:
            model += [nn.Conv2d(self.F, 64, kernel_size=self.kernel_size, padding=self.padding_size, bias=True)]
        model += [nn.ReLU(True)]
        if self.dropout: model += [nn.Dropout(0.2)]


        if self.use_64C:
            model += [nn.Conv2d(64, 13, kernel_size=self.kernel_size, padding=self.padding_size, bias=True)]
        else:
            model += [nn.Conv2d(self.F, input_nc, kernel_size=self.kernel_size, padding=self.padding_size, bias=True)]
        
        self.model = nn.Sequential(*model)

    def forward(self, input):
        # long-skip connection: add cloudy MS input (excluding the trailing two SAR channels) and model output
        return self.model(input) # + self.use_long*input[:, :(-2*self.use_SAR), ...]


class ResnetBlock2D(nn.Module):
    def __init__(self, dim, use_dropout, use_bias, res_scale=0.1, padding_size=1):
        super(ResnetBlock2D, self).__init__()
        self.res_scale = res_scale
        self.padding_size = padding_size
        self.conv_block = self.build_conv_block(dim, use_dropout, use_bias)

        # conv_block:
        #   CONV (pad, conv, norm),
        #   RELU (relu, dropout),
        #   CONV (pad, conv, norm)
    def build_conv_block(self, dim, use_dropout, use_bias):
        conv_block = []

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=self.padding_size, bias=use_bias)]
        conv_block += [nn.ReLU(True)]

        if use_dropout:
            conv_block += [nn.Dropout(0.2)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=self.padding_size, bias=use_bias)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        # add residual mapping
        out = x + self.res_scale * self.conv_block(x)
        return out
