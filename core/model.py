import copy
import math
from munch import Munch
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class PIN(nn.Module):
    def __init__(self, hw, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=num_features, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=num_features, out_channels=num_features * 2, kernel_size=1, padding=0)
        if hw == 128:
            self.conv3 = nn.ConvTranspose2d(in_channels=num_features * 2, out_channels=num_features * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        if hw == 256:
            self.conv3 = nn.ConvTranspose2d(in_channels=num_features * 2, out_channels=num_features * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.conv4 = nn.ConvTranspose2d(in_channels=num_features * 2, out_channels=num_features * 2, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x, s):
        h = s.reshape(s.size(0), 64, 64)
        h = h.unsqueeze(1)
        h = self.conv2(self.conv1(h))
        if x.size(-1) == 128:
            h = self.conv3(h)
        if x.size(-1) == 256:
            h = self.conv4(self.conv3(h))

        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class PINResnetBlock(nn.Module):
    def __init__(self, dim, hw, padding_type, norm_layer, use_dropout, use_bias):
        super(PINResnetBlock, self).__init__()
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(hw=hw, num_features=dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(hw=hw, num_features=dim)]

        self.conv_block = nn.ModuleList(conv_block)

    def forward(self, x, s):
        """Forward function (with skip connections)"""
        x_ori = x
        for blk in self.conv_block:
            if isinstance(blk, PIN):
                x = blk(x, s)
            else:
                x = blk(x)

        out = x_ori + x  # add skip connections
        return out


class Generator(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, img_size=256, ngf=64, use_dropout=False, n_blocks=8, padding_type='reflect'):
        ResnetBlock_norm = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        PINResnetBlock_norm = PIN

        assert (n_blocks >= 0)
        super(Generator, self).__init__()

        # Encoder
        self.encoder = [nn.ReflectionPad2d(3),
                        nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
                        ResnetBlock_norm(ngf),
                        nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            self.encoder += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=True),
                             ResnetBlock_norm(ngf * mult * 2),
                             nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(int(n_blocks / 2)):
            self.encoder += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=ResnetBlock_norm,
                                         use_dropout=use_dropout, use_bias=True)]

        # Style Fusion Modules
        self.style_fusion_modules, self.decoder = [], []
        for i in range(int(n_blocks - n_blocks / 2)):
            self.style_fusion_modules += [PINResnetBlock(ngf * mult, hw=int(img_size / mult), padding_type=padding_type,
                                                         norm_layer=PINResnetBlock_norm,
                                                         use_dropout=use_dropout, use_bias=True)]

        # Decoder with PIN
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            self.decoder += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                                kernel_size=3, stride=2,
                                                padding=1, output_padding=1,
                                                bias=True),
                             PINResnetBlock_norm(hw=int(img_size / mult * 2),
                                                 num_features=int(ngf * mult / 2)),
                             nn.ReLU(True)]

        self.decoder += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]

        self.encoder = nn.Sequential(*self.encoder)
        self.style_fusion_modules = nn.ModuleList(self.style_fusion_modules)
        self.decoder = nn.ModuleList(self.decoder)

    def forward(self, x, s):
        x_ = self.encoder(x)
        i = 0
        for sf_module in self.style_fusion_modules:
            x_ = sf_module(x_, s[:, i, :])
            i += 1
        for blk in self.decoder:
            if isinstance(blk, PIN):
                x_ = blk(x_, s[:, i, :])
                i += 1
            else:
                x_ = blk(x_)
        return x_


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2), normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


class Discriminator(nn.Module):
    def __init__(self, input_channel=1, num_domains=4):
        super().__init__()
        self.input_channel = input_channel
        self.dim = 64
        self.num_domains = num_domains

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.cnns = nn.ModuleList()
        for _ in range(4):
            self.cnns.append(self._make_net())

    def _make_net(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [nn.Conv2d(self.input_channel, dim, 3, 1, 1)]
        for _ in range(4):
            dim_out = min(dim * 2, 512)
            cnn_x += [ResBlk(dim, dim_out, downsample=True)]
            dim = dim_out

        cnn_x += [nn.LeakyReLU(0.2)]
        cnn_x += [nn.Conv2d(dim_out, dim_out, 1, 1, 0)]
        cnn_x += [nn.LeakyReLU(0.2)]
        cnn_x += [nn.Conv2d(dim_out, self.num_domains, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)

        return cnn_x

    def forward(self, x, y):
        outputs = []
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        for model in self.cnns:
            out = model(x)  # (batch, num_domains, x, x)
            out = out[idx, y]  # (batch, x, x)
            outputs.append(out.unsqueeze(1))  # (batch, 1, x, x)
            x = self.downsample(x)

        return outputs

    def calc_dis_loss(self, input_fake, input_real, fake_label, real_label):
        # calculate the loss to train D
        outs0 = self.forward(input_fake, fake_label)
        outs1 = self.forward(input_real, real_label)
        loss = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            loss += F.binary_cross_entropy_with_logits(out0, torch.full_like(out0, fill_value=0)) \
                    + F.binary_cross_entropy_with_logits(out1, torch.full_like(out1, fill_value=1))

        return loss

    def calc_gen_loss(self, input_fake, fake_label):
        # calculate the loss to train G
        outs0 = self.forward(input_fake, fake_label)
        loss = 0

        for it, (out0) in enumerate(outs0):
            loss += F.binary_cross_entropy_with_logits(out0, torch.full_like(out0, fill_value=1))

        return loss


def build_model(args):
    generator = nn.DataParallel(
        Generator(input_nc=args.input_channel, output_nc=args.input_channel, img_size=args.img_size, ngf=64,
                  use_dropout=False, n_blocks=args.n_blocks, padding_type='reflect'))
    discriminator = nn.DataParallel(Discriminator(args.input_channel, args.num_domains))
    style_vectors = nn.Parameter(torch.randn(args.num_domains, int(args.n_blocks / 2) + 2, args.style_dim))
    style_vectors_ema = copy.deepcopy(style_vectors)
    generator_ema = copy.deepcopy(generator)

    nets = Munch(generator=generator,
                 discriminator=discriminator,
                 style_vectors=style_vectors)
    nets_ema = Munch(generator=generator_ema,
                     style_vectors=style_vectors_ema)

    return nets, nets_ema
