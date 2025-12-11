import torch
import torch.nn as nn
from torch.nn import Parameter
from collections import OrderedDict
# NOTE Inpainting Stage1: 
from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from typing import Callable, List, Optional, Union
# NOTE DIY adapter:
from src.models.transformer import TransformerEncoder, TransformerDecoder
from src.models.feature_fusion import FF
from src.models.ffc import FFCResnetBlock, ConcatTupleLayer

# -----------------------------------------------
#                    FFC Block
# -----------------------------------------------
class FFCBlock(torch.nn.Module):
    def __init__(self,
                 dim,  # Number of output/input channels.
                 kernel_size,  # Width and height of the convolution kernel.
                 padding,
                 ratio_gin=0.75,
                 ratio_gout=0.75,
                 activation='linear',  # Activation function: 'relu', 'lrelu', etc.
                 ):
        super().__init__()
        if activation == 'linear':
            self.activation = nn.Identity
        else:
            self.activation = nn.ReLU
        self.padding = padding
        self.kernel_size = kernel_size
        self.ffc_block = FFCResnetBlock(dim=dim,
                                        padding_type='reflect',
                                        norm_layer=nn.SyncBatchNorm,
                                        activation_layer=self.activation,
                                        dilation=1,
                                        ratio_gin=ratio_gin,
                                        ratio_gout=ratio_gout)

        self.concat_layer = ConcatTupleLayer()

    def forward(self, gen_ft, mask, fname=None):
        x = gen_ft#.float()

        x_l, x_g = x[:, :-self.ffc_block.conv1.ffc.global_in_num], x[:, -self.ffc_block.conv1.ffc.global_in_num:]
        id_l, id_g = x_l, x_g

        x_l, x_g = self.ffc_block((x_l, x_g), fname=fname)
        x_l, x_g = id_l + x_l, id_g + x_g
        x = self.concat_layer((x_l, x_g))

        return x + gen_ft#.float()


class FFCSkipLayer(torch.nn.Module):
    def __init__(self,
                 dim,  # Number of input/output channels.
                 kernel_size=3,  # Convolution kernel size.
                 ratio_gin=0.75,
                 ratio_gout=0.75,
                 ):
        super().__init__()
        self.padding = kernel_size // 2

        self.ffc_act = FFCBlock(dim=dim, kernel_size=kernel_size, activation=nn.ReLU,
                                padding=self.padding, ratio_gin=ratio_gin, ratio_gout=ratio_gout)

    def forward(self, gen_ft, mask=None, fname=None):
        x = self.ffc_act(gen_ft, mask, fname=fname)
        return x

# NOTE Inpainting Stage1: Module Encapsulation
class T2IAdapter(ModelMixin, ConfigMixin):
    r"""
    A simple ResNet-like model that accepts images containing control signals such as keyposes and depth. The model
    generates multiple feature maps that are used as additional conditioning in [`UNet2DConditionModel`]. The model's
    architecture follows the original implementation of
    [Adapter](https://github.com/TencentARC/T2I-Adapter/blob/686de4681515662c0ac2ffa07bf5dda83af1038a/ldm/modules/encoders/adapter.py#L97)
     and
     [AdapterLight](https://github.com/TencentARC/T2I-Adapter/blob/686de4681515662c0ac2ffa07bf5dda83af1038a/ldm/modules/encoders/adapter.py#L235).

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the model (such as downloading or saving, etc.)

    Parameters:
        in_channels (`int`, *optional*, defaults to 3):
            Number of channels of Aapter's input(*control image*). Set this parameter to 1 if you're using gray scale
            image as *control image*.
        channels (`List[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The number of channel of each downsample block's output hidden state. The `len(block_out_channels)` will
            also determine the number of downsample blocks in the Adapter.
        num_res_blocks (`int`, *optional*, defaults to 2):
            Number of ResNet blocks in each downsample block.
        downscale_factor (`int`, *optional*, defaults to 8):
            A factor that determines the total downscale factor of the Adapter.
        adapter_type (`str`, *optional*, defaults to `full_adapter`):
            The type of Adapter to use. Choose either `full_adapter` or `full_adapter_xl` or `light_adapter`.
    """

    @register_to_config
    def __init__(
        self,
        channels: List[int] = [320, 640, 1280, 1280], 
        nums_rb: int = 3, 
        cin: int = 64, 
        ksize: int = 3, 
        sk: bool = False, 
        use_conv: bool = True,
        # in_channels: int = 3,
        # channels: List[int] = [320, 640, 1280, 1280],
        # num_res_blocks: int = 2,
        # downscale_factor: int = 8,
        adapter_type: str = "Adapter",
    ):
        super().__init__()

        # if adapter_type == "full_adapter":
        #     self.adapter = FullAdapter(in_channels, channels, num_res_blocks, downscale_factor)
        # elif adapter_type == "full_adapter_xl":
        #     self.adapter = FullAdapterXL(in_channels, channels, num_res_blocks, downscale_factor)
        # elif adapter_type == "light_adapter":
        #     self.adapter = LightAdapter(in_channels, channels, num_res_blocks, downscale_factor)
        if adapter_type == "Adapter":
            self.adapter = Adapter(channels=channels, nums_rb=nums_rb, cin=cin, ksize=ksize, sk=sk, 
                                   use_conv=use_conv)
        elif adapter_type == "NoRes_Adapter":
             self.adapter = NoRes_Adapter(channels=channels, nums_rb=nums_rb, cin=cin, ksize=ksize, 
                                          sk=sk, use_conv=use_conv)
        else:
            raise ValueError(
                f"Unsupported adapter_type: '{adapter_type}'. Choose either 'full_adapter' or "
                "'full_adapter_xl' or 'light_adapter'."
            )

    def forward(self, x):     # : torch.Tensor) -> List[torch.Tensor]:
        r"""
        This function processes the input tensor `x` through the adapter model and returns a list of feature tensors,
        each representing information extracted at a different scale from the input. The length of the list is
        determined by the number of downsample blocks in the Adapter, as specified by the `channels` and
        `num_res_blocks` parameters during initialization.
        """
        return self.adapter(x)

    @property
    def total_downscale_factor(self):
        return self.adapter.total_downscale_factor

    @property
    def downscale_factor(self):
        """The downscale factor applied in the T2I-Adapter's initial pixel unshuffle operation. If an input image's dimensions are
        not evenly divisible by the downscale_factor then an exception will be raised.
        """
        return self.adapter.unshuffle.downscale_factor



def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResnetBlock(nn.Module):
    def __init__(self, in_c, out_c, down, ksize=3, sk=False, use_conv=True):
        super().__init__()

        ps = ksize // 2
        if in_c != out_c or sk == False:
            self.in_conv = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.in_conv = None

        # NOTE feature fusion branch:
        self.feature_fusion = FF(320, out_c, out_c)
        # self.gamma = nn.Parameter(torch.zeros(1))

        
        self.block1 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act_1 = nn.ReLU()

        # NOTE transformer or FFC:
        # self.attn = TransformerEncoder(patchsizes=1, num_hidden=out_c, dis=None)
        self.ffc = FFCSkipLayer(out_c)
        self.act_2 = nn.ReLU()

        self.block2 = nn.Conv2d(out_c, out_c, ksize, 1, ps)

        
        if sk == False:
            self.skep = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.skep = None

        self.down = down
        if self.down == True:
            self.down_opt = Downsample(in_c, use_conv=use_conv)
        self.down_origin = Downsample(320, use_conv=use_conv)

    def forward(self, x, origin=None):
        if self.down == True:
            x = self.down_opt(x)
        if self.in_conv is not None:  # edit
            x = self.in_conv(x)
        
        # NOTE feature fusion branch:
        if origin is not None and x.shape[2] != origin.shape[2]:
            origin = self.down_origin(origin)
            while x.shape[2] != origin.shape[2]:
                origin = self.down_origin(origin)
        ff, ff_skip = self.feature_fusion(origin, x)

        # NOTE feature fusion: add skip connect
        x_add = x + ff_skip
        # NOTE x -> x_add
        h = self.block1(x_add)
        h = self.act_1(h)
        
        # NOTE transformer or FFC:
        # h = self.attn(h)
        h = self.ffc(h)
        h = self.act_2(h)

        h = self.block2(h)       
        
        # NOTE feature fusion branch: add "ff *" and "+ ff"
        if self.skep is not None:
            return ff * h + self.skep(x) + ff
        else:
            return ff * h + x + ff


class NormalBlock(nn.Module):
    def __init__(self, in_c, out_c, down, ksize=3, sk=False, use_conv=True):
        super().__init__()
        ps = ksize // 2
        if in_c != out_c or sk == False:
            self.in_conv = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            # print('n_in')
            self.in_conv = None
        self.block1 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(out_c, out_c, ksize, 1, ps)
        if sk == False:
            self.skep = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.skep = None

        self.down = down
        if self.down == True:
            self.down_opt = Downsample(in_c, use_conv=use_conv)

    def forward(self, x):
        if self.down == True:
            x = self.down_opt(x)
        if self.in_conv is not None:  # edit
            x = self.in_conv(x)

        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)
        if self.skep is not None:
            return h
        else:
            return h

class DIY_NormalBlock(nn.Module):
    def __init__(self, in_c, out_c, down, ksize=3, sk=False, use_conv=True):
        super().__init__()
        ps = ksize // 2
        if in_c != out_c or sk == False:
            self.in_conv = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            # print('n_in')
            self.in_conv = None
        self.block1 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(out_c, out_c, ksize, 1, ps)
        if sk == False:
            self.skep = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.skep = None

        self.down = down
        if self.down == True:
            self.down_opt = Downsample(in_c, use_conv=use_conv)

    def forward(self, x):
        if self.down == True:
            x = self.down_opt(x)
        if self.in_conv is not None:  # edit
            x = self.in_conv(x)

        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)
        if self.skep is not None:
            return h
        else:
            return h

class Adapter(nn.Module):
    def __init__(self, channels=[320, 640, 1280, 1280], nums_rb=3, cin=64, ksize=3, sk=False, use_conv=True):
        super(Adapter, self).__init__()
        self.unshuffle = nn.PixelUnshuffle(8)  # (N, C * 8^2 = cin, H / 8 = 64, W / 8 = 64)
        self.channels = channels
        self.nums_rb = nums_rb
        self.body = []
        # NOTE transformer or FFC:
        # self.attn = []
        # NOTE: feature size = [64,32,16,8] for SD v1.5 UNet encoder.
        size = [64, 32, 16, 8]
        for i in range(len(channels)):
            for j in range(nums_rb):
                if (i != 0) and (j == 0):
                    self.body.append(
                        ResnetBlock(channels[i - 1], channels[i], down=True, ksize=ksize, sk=sk, use_conv=use_conv))
                    # NOTE transformer or FFC:
                    # self.attn.append(
                    #     TransformerEncoder(patchsizes=1, num_hidden=channels[i], dis=None))
                else:
                    self.body.append(
                        ResnetBlock(channels[i], channels[i], down=False, ksize=ksize, sk=sk, use_conv=use_conv))
                    # NOTE transformer or FFC:
                    # self.attn.append(
                    #     TransformerEncoder(patchsizes=1, num_hidden=channels[i], dis=None))
        self.body = nn.ModuleList(self.body)
        # NOTE transformer or FFC:
        # self.attn = nn.ModuleList(self.attn)
        self.conv_in = nn.Conv2d(cin, channels[0], 3, 1, 1)   # cin = 64 * 4

    def forward(self, x):

        x = self.unshuffle(x)

        # extract features
        features = []
        x = self.conv_in(x)
        # NOTE feature fusion branch:
        ff_origin = x

        for i in range(len(self.channels)):
            for j in range(self.nums_rb):
                idx = i * self.nums_rb + j
                x = self.body[idx](x, origin=ff_origin)
                # NOTE transformer or FFC:
                # x = self.attn[idx](x)
            features.append(x)   
            # NOTE: feature size = [64,32,16,8] for SD v1.5 UNet encoder.

        return features 



class NoRes_Adapter(nn.Module):
    def __init__(self, channels=[320, 640, 1280, 1280], nums_rb=3, cin=64, ksize=3, sk=False, use_conv=True):
        super(NoRes_Adapter, self).__init__()
        self.unshuffle = nn.PixelUnshuffle(8)
        self.channels = channels
        self.nums_rb = nums_rb
        self.body = []
        for i in range(len(channels)):
            for j in range(nums_rb):
                if (i != 0) and (j == 0):
                    self.body.append(
                        NormalBlock(channels[i - 1], channels[i], down=True, ksize=ksize, sk=sk, use_conv=use_conv))
                else:
                    self.body.append(
                        NormalBlock(channels[i], channels[i], down=False, ksize=ksize, sk=sk, use_conv=use_conv))
        self.body = nn.ModuleList(self.body)
        self.conv_in = nn.Conv2d(cin, channels[0], 3, 1, 1)


    def forward(self, x):

        x = self.unshuffle(x)

        features = []
        x = self.conv_in(x)

        for i in range(len(self.channels)):
            for j in range(self.nums_rb):
                idx = i * self.nums_rb + j
                x = self.body[idx](x)

            features.append(x)

        return features
    
class DIY_NoRes_Adapter(nn.Module):
    def __init__(self, channels=[320, 640, 1280, 1280], nums_rb=3, cin=64, ksize=3, sk=False, use_conv=True):
        super(NoRes_Adapter, self).__init__()
        self.unshuffle = nn.PixelUnshuffle(8)
        self.channels = channels
        self.nums_rb = nums_rb
        self.body = []
        for i in range(len(channels)):
            for j in range(nums_rb):
                if (i != 0) and (j == 0):
                    self.body.append(
                        NormalBlock(channels[i - 1], channels[i], down=True, ksize=ksize, sk=sk, use_conv=use_conv))
                else:
                    self.body.append(
                        NormalBlock(channels[i], channels[i], down=False, ksize=ksize, sk=sk, use_conv=use_conv))
        self.body = nn.ModuleList(self.body)
        self.conv_in = nn.Conv2d(cin, channels[0], 3, 1, 1)


    def forward(self, x):

        x = self.unshuffle(x)

        features = []
        x = self.conv_in(x)

        for i in range(len(self.channels)):
            for j in range(self.nums_rb):
                idx = i * self.nums_rb + j
                x = self.body[idx](x)

            features.append(x)

        return features

