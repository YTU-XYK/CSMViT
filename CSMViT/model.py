"""
original code from apple:
https://github.com/apple/ml-cvnets/blob/main/cvnets/models/classification/mobilevit.py
"""
from typing import Optional, Tuple, Union, Dict
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from transformer import TransformerEncoder
from model_config import get_config


def make_divisible(
        v: Union[float, int],  # 输入值，我们希望使其可被除尽
        divisor: Optional[int] = 8,  # 我们希望除尽该值的除数，默认为8
        min_value: Optional[Union[float, int]] = None,  # 划分后的最小值，默认为None
) -> Union[float, int]:  # 函数返回一个浮点数或整数
    """
    此函数来自原始的 TensorFlow 仓库。
    它确保所有的层的通道数都是8的倍数。
    它可以在这里找到：
    https://github.com/tensorflow/models
    """
    if min_value is None:  # 如果最小值为None
        min_value = divisor  # 则最小值为除数本身
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)  # 根据条件调整输入值
    # 确保向下取整不会降低超过原始值的10%。
    if new_v < 0.9 * v:  # 如果新值小于原始值的90%
        new_v += divisor  # 则增加除数
    return new_v  # 返回处理后的值


class ConvLayer(nn.Module):
    """
    对输入进行二维卷积处理

    Args:
        in_channels (int): 输入通道数，来自期望输入大小为 :math:`(N, C_{in}, H_{in}, W_{in})`
        out_channels (int): 输出通道数，来自期望输出大小为 :math:`(N, C_{out}, H_{out}, W_{out})`
        kernel_size (Union[int, Tuple[int, int]]): 卷积核大小。
        stride (Optional[Union[int, Tuple[int, int]]]): 卷积步长。默认：1
        groups (Optional[int]): 卷积组数。默认：1
        bias (Optional[bool]): 是否使用偏置。默认：``False``
        use_norm (Optional[bool]): 是否在卷积后使用规范化层。默认：``True``
        use_act (Optional[bool]): 是否在卷积后使用激活函数层。默认：``True``

    Shape:
        - 输入: :math:`(N, C_{in}, H_{in}, W_{in})`
        - 输出: :math:`(N, C_{out}, H_{out}, W_{out})`

    . note::
        对于深度卷积，`groups=C_{in}=C_{out}`。
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int]],
            stride: Optional[Union[int, Tuple[int, int]]] = 1,
            groups: Optional[int] = 1,
            bias: Optional[bool] = False,  # 是否使用偏置
            use_norm: Optional[bool] = True,  # 是否构建BN
            use_act: Optional[bool] = True,  # 是否使用激活函数
    ) -> None:
        super().__init__()

        if isinstance(kernel_size, int):  # 如果卷积核大小是整数，转换为元组形式
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):  # 如果步长是整数，转换为元组形式
            stride = (stride, stride)

        assert isinstance(kernel_size, Tuple)  # 断言卷积核大小为元组
        assert isinstance(stride, Tuple)  # 断言步长为元组

        # 计算填充量
        padding = (
            int((kernel_size[0] - 1) / 2),
            int((kernel_size[1] - 1) / 2),
        )

        block = nn.Sequential()  # 定义一个序列化模块

        conv_layer = nn.Conv2d(  # 实例化卷积层
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=padding,
            bias=bias
        )

        block.add_module(name="conv", module=conv_layer)  # 添加卷积层到序列化模块中

        if use_norm:  # 如果为True则添加一个BN层
            norm_layer = nn.BatchNorm2d(num_features=out_channels, momentum=0.1)
            block.add_module(name="norm", module=norm_layer)

        if use_act:  # 如果为True则添加一个SiLU激活函数
            act_layer = nn.SiLU()
            block.add_module(name="act", module=act_layer)

        self.block = block  # 将序列化模块传给self.block

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)  # 对输入数据进行卷积操作


class ECANet(nn.Module):
    def __init__(self, in_channels, gamma=2, b=1):
        super(ECANet, self).__init__()
        self.in_channels = in_channels
        self.fgp = nn.AdaptiveAvgPool2d((1, 1))
        kernel_size = int(abs((math.log(self.in_channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.con1 = nn.Conv1d(1,
                              1,
                              kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2,
                              bias=False)
        self.act1 = nn.Sigmoid()

    def forward(self, x):
        output = self.fgp(x)
        output = output.squeeze(-1).transpose(-1, -2)
        output = self.con1(output).transpose(-1, -2).unsqueeze(-1)
        output = self.act1(output)
        output = torch.multiply(x, output)
        return output


class SqueezeAndExcite(nn.Module):
    def __init__(self, in_channels, out_channels, divide=8):
        super(SqueezeAndExcite, self).__init__()
        mid_channels = in_channels // divide
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.SEblock = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=mid_channels),
            nn.SiLU(inplace=True),
            nn.Linear(in_features=mid_channels, out_features=out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        b, c, h, w = x.size()
        out = self.pool(x)
        out = out.view(b, -1)
        # print('out',out.shape)
        out = self.SEblock(out)
        out = out.view(b, c, 1, 1)
        return out * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg_out, max_out], dim=1)
        conv = self.conv1(cat)
        return x * self.sigmoid(conv)


class CSEAM(nn.Module):
    def __init__(self, input_c, output_c, input_1x1c, output_1x1c, kernel=3):
        super(CSEAM, self).__init__()
        self.conv = nn.Conv2d(input_c, output_c, kernel_size=kernel, stride=2, padding=1)
        self.conv1x1 = nn.Conv2d(input_1x1c, output_1x1c, kernel_size=1, stride=1)
        self.SA = SpatialAttention()
        self.SE = SqueezeAndExcite(output_c, output_c)
        self.eca = ECANet(output_c)
        self.max = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.adjust = nn.Conv2d(output_c, output_c // 2, kernel_size=1, stride=1)
        # 初始化可学习权重系数
        self.w = nn.Parameter(torch.ones(4))

    def forward(self, x, y):
        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        w3 = torch.exp(self.w[2]) / torch.sum(torch.exp(self.w))
        w4 = torch.exp(self.w[3]) / torch.sum(torch.exp(self.w))
        max = self.max(x)
        # print("x:",max.shape)
        # y1, y2 = y.chunk(2, dim=1)
        y = self.conv1x1(y)
        # print("y1:", y1.shape)
        # print("y2:", y2.shape)
        y2 = F.interpolate(y, size=x.size()[2:], mode='bilinear', align_corners=True)
        add = torch.cat((max, y), dim=1)
        add2 = torch.cat((x, y2), dim=1)
        # print("add:", add.shape)
        se = self.SE(add)
        se2 = self.SE(add2)
        sa1 = self.SA(se)
        # print("sa1:", sa1.shape)
        sa1 = sa1 * w3
        sa2 = self.SA(se2)
        sa2 = sa2 * w2
        # print("sa2:", sa2.shape)
        sa1_interpolate = F.interpolate(sa1, size=sa2.size()[2:], mode='bilinear', align_corners=True)
        sa1_interpolate = sa1_interpolate * w1
        out1 = torch.add(sa1_interpolate, sa2)
        # print("out1:", out1.shape)
        maxpool = self.max(sa2)
        maxpool = maxpool * w4
        out2 = torch.add(maxpool, sa1)
        # print("out2:", out2.shape)
        out1 = self.adjust(out1)
        # out2 = self.adjust(out2)
        return out1, out2


def Conv3x3BNReLU(in_channels, out_channels, stride, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1,
                  groups=groups, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True)
    )


class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, s=2, kernel_size=1, stride=1, use_relu=True):
        super(GhostModule, self).__init__()
        intrinsic_channels = out_channels // s
        ghost_channels = intrinsic_channels * (s - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=intrinsic_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(intrinsic_channels),
            nn.SiLU(inplace=True) if use_relu else nn.Sequential()
        )

        # self.cheap_op = DW_Conv3x3BNReLU(in_channels=intrinsic_channels, out_channels=ghost_channels, stride=stride,
        #                                  groups=intrinsic_channels)
        self.cheap_op = ConvLayer(in_channels=intrinsic_channels, out_channels=ghost_channels, stride=stride,
                                  kernel_size=3, groups=intrinsic_channels, use_norm=True, use_act=True)

    def forward(self, x):
        y = self.primary_conv(x)
        z = self.cheap_op(y)
        out = torch.cat([y, z], dim=1)
        return out


class GhostBottleneck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, stride, use_se, se_kernel_size=1):
        super(GhostBottleneck, self).__init__()
        self.stride = stride

        self.bottleneck = nn.Sequential(
            GhostModule(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, use_relu=True),
            ConvLayer(in_channels=mid_channels, out_channels=mid_channels, stride=stride, kernel_size=3, use_act=True,
                      use_norm=True,
                      groups=mid_channels) if self.stride > 1 else nn.Sequential(),
            SqueezeAndExcite(mid_channels, mid_channels, se_kernel_size) if use_se else nn.Sequential(),
            GhostModule(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, use_relu=False)
        )

        if self.stride > 1:
            self.shortcut = Conv3x3BNReLU(in_channels=in_channels, out_channels=out_channels, stride=stride)
        else:
            self.shortcut = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.bottleneck(x)
        residual = self.shortcut(x)
        out += residual
        return out


class MobileViTBlock(nn.Module):
    """
    This class defines the `MobileViT block <https://arxiv.org/abs/2110.02178?context=cs.LG>`_

    Args:
        opts: command line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)`
        transformer_dim (int): Input dimension to the transformer unit
        ffn_dim (int): Dimension of the FFN block
        n_transformer_blocks (int): Number of transformer blocks. Default: 2
        head_dim (int): Head dimension in the multi-head attention. Default: 32
        attn_dropout (float): Dropout in multi-head attention. Default: 0.0
        dropout (float): Dropout rate. Default: 0.0
        ffn_dropout (float): Dropout between FFN layers in transformer. Default: 0.0
        patch_h (int): Patch height for unfolding operation. Default: 8
        patch_w (int): Patch width for unfolding operation. Default: 8
        transformer_norm_layer (Optional[str]): Normalization layer in the transformer block. Default: layer_norm
        conv_ksize (int): Kernel size to learn local representations in MobileViT block. Default: 3
        no_fusion (Optional[bool]): Do not combine the input and output feature maps. Default: False
    """

    def __init__(
            self,
            in_channels: int,
            transformer_dim: int,  # 输入到transformer encoder中每个token对应的序列长度
            ffn_dim: int,  # transformer encoder MLP中第一个全连接层对应的个数
            n_transformer_blocks: int = 2,  # Transformer堆叠的次数
            head_dim: int = 32,  # 多头自注意力时每个头所对应的尺寸 dimensions 尺寸
            attn_dropout: float = 0.0,
            dropout: float = 0.0,
            ffn_dropout: float = 0.0,
            patch_h: int = 8,
            patch_w: int = 8,
            conv_ksize: Optional[int] = 3,
            *args,
            **kwargs
    ) -> None:
        super().__init__()

        # 定义卷积层用于局部特征学习
        conv_3x3_in = ConvLayer(  # unfold之前的3*3
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=conv_ksize,  # n = 3 3*3卷积
            stride=1
        )
        conv_1x1_in = ConvLayer(  # unfold之前的1*1
            in_channels=in_channels,
            out_channels=transformer_dim,
            kernel_size=1,  # 1*1卷积
            stride=1,
            use_norm=False,
            use_act=False
        )

        # 定义卷积层用于转换到transformer维度
        conv_1x1_out = ConvLayer(  # fold之后的1*1
            in_channels=transformer_dim,
            out_channels=in_channels,
            kernel_size=1,
            stride=1
        )
        conv_3x3_out = ConvLayer(  # 拼接之后的3*3
            in_channels=2 * in_channels,
            out_channels=in_channels,
            kernel_size=conv_ksize,
            stride=1
        )
        # 将3*3卷积和1*1卷积加入到local_rep中
        self.local_rep = nn.Sequential()
        self.local_rep.add_module(name="conv_3x3", module=conv_3x3_in)
        self.local_rep.add_module(name="conv_1x1", module=conv_1x1_in)

        assert transformer_dim % head_dim == 0
        num_heads = transformer_dim // head_dim

        global_rep = [
            TransformerEncoder(
                embed_dim=transformer_dim,
                ffn_latent_dim=ffn_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                dropout=dropout,
                ffn_dropout=ffn_dropout
            )
            for _ in range(n_transformer_blocks)
        ]
        global_rep.append(nn.LayerNorm(transformer_dim))
        self.global_rep = nn.Sequential(*global_rep)

        self.conv_proj = conv_1x1_out
        self.fusion = conv_3x3_out

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h

        self.cnn_in_dim = in_channels
        self.cnn_out_dim = transformer_dim
        self.n_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.n_blocks = n_transformer_blocks
        self.conv_ksize = conv_ksize

    def unfolding(self, x: Tensor) -> Tuple[Tensor, Dict]:
        # 解析输入特征图为patch形式
        patch_w, patch_h = self.patch_w, self.patch_h
        patch_area = patch_w * patch_h
        batch_size, in_channels, orig_h, orig_w = x.shape

        new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
        new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)

        interpolate = False
        if new_w != orig_w or new_h != orig_h:
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
            interpolate = True

        num_patch_w = new_w // patch_w
        num_patch_h = new_h // patch_h
        num_patches = num_patch_h * num_patch_w

        x = x.reshape(batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w)
        x = x.transpose(1, 2)
        x = x.reshape(batch_size, in_channels, num_patches, patch_area)
        x = x.transpose(1, 3)
        x = x.reshape(batch_size * patch_area, num_patches, -1)

        info_dict = {
            "orig_size": (orig_h, orig_w),
            "batch_size": batch_size,
            "interpolate": interpolate,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h,
        }

        return x, info_dict

    def folding(self, x: Tensor, info_dict: Dict) -> Tensor:
        n_dim = x.dim()
        assert n_dim == 3, "Tensor should be of shape BPxNxC. Got: {}".format(
            x.shape
        )

        x = x.contiguous().view(
            info_dict["batch_size"], self.patch_area, info_dict["total_patches"], -1
        )

        batch_size, pixels, num_patches, channels = x.size()
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]

        x = x.transpose(1, 3)
        x = x.reshape(batch_size * channels * num_patch_h, num_patch_w, self.patch_h, self.patch_w)
        x = x.transpose(1, 2)
        x = x.reshape(batch_size, channels, num_patch_h * self.patch_h, num_patch_w * self.patch_w)
        if info_dict["interpolate"]:
            x = F.interpolate(
                x,
                size=info_dict["orig_size"],
                mode="bilinear",
                align_corners=False,
            )
        return x

    def forward(self, x: Tensor) -> Tensor:  # 前向传播过程
        res = x  # 将一开始的输入保存到res

        fm = self.local_rep(x)  # local_rep中有1个3*3卷积有一个1*1卷积

        patches, info_dict = self.unfolding(fm)

        for transformer_layer in self.global_rep:
            patches = transformer_layer(patches)

        fm = self.folding(x=patches, info_dict=info_dict)

        fm = self.conv_proj(fm)  # unfold之后的1*1卷积

        fm = self.fusion(torch.cat((res, fm), dim=1))  # res为一开始的输入，与处理过的fm拼接
        return fm


class CMViT(nn.Module):
    """
    This class implements the `MobileViT architecture <https://arxiv.org/abs/2110.02178?context=cs.LG>`_
    """

    def __init__(self, model_cfg: Dict, num_classes: int = 1000):
        super().__init__()
        image_channels = 3  # 输入图片通道数
        out_channels = 16  # 经过3*3卷积之后输出的通道数
        self.conv_1 = ConvLayer(
            in_channels=image_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2
        )

        self.layer_1, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer1"])  # MV2
        self.layer_2, out_channels = self._make_layer(input_channel=out_channels,
                                                      cfg=model_cfg["layer2"])  # MV2下采样+MV2*2
        self.layer_3, out_channels = self._make_layer(input_channel=out_channels,
                                                      cfg=model_cfg["layer3"])  # MV2下采样 + mobilevit block
        self.layer_4, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer4"])

        self.conv_adjust = nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1)

        self.conv_adjust1 = ConvLayer(kernel_size=3, in_channels=24, out_channels=96, stride=2, use_act=True,
                                      use_norm=True)
        self.after_adjust1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv_adjust2 = ConvLayer(kernel_size=1, in_channels=48, out_channels=96, use_act=True, use_norm=True)
        # self.conv_adjust3 = DW_Conv3x3BNReLU(in_channels=48, out_channels=96, stride=2)
        self.after_adjust2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.conv_adjust3 = ConvLayer(in_channels=64, out_channels=96, kernel_size=1, use_act=False, use_norm=True)
        self.after_adjust3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv_branch_1 = ConvLayer(in_channels=24, out_channels=24, kernel_size=3, stride=2, use_norm=True,
                                       use_act=True)
        self.conv_branch_2 = ConvLayer(in_channels=24, out_channels=48, stride=4, kernel_size=5, use_norm=True,
                                       use_act=True)
        self.conv_branch_3_1 = ConvLayer(in_channels=24, out_channels=48, kernel_size=3, stride=2, use_norm=True,
                                         use_act=True)
        self.conv_branch_3_2 = ConvLayer(in_channels=48, out_channels=96, kernel_size=5, stride=4, use_norm=True,
                                         use_act=True)

        exp_channels = min(model_cfg["last_layer_exp_factor"] * out_channels, 960)  # 最后的1*1卷积输出的通道数
        self.conv_1x1_exp = ConvLayer(  # layer后的1*1卷积层
            in_channels=96,
            out_channels=exp_channels,
            kernel_size=1
        )
        self.a = nn.Parameter(torch.ones(2))
        self.MyAttention1 = CSEAM(96, 48, 48, 24)
        self.MyAttention2 = CSEAM(192, 96, 96, 48)
        self.after_attention = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.after_cat = ConvLayer(in_channels=144, out_channels=96, kernel_size=1, stride=1)
        self.classifier = nn.Sequential()  # 分类
        self.classifier.add_module(name="global_pool", module=nn.AdaptiveAvgPool2d(1))
        self.classifier.add_module(name="flatten", module=nn.Flatten())
        self.last1x1 = ConvLayer(in_channels=192, out_channels=96, kernel_size=1, stride=1)
        if 0.0 < model_cfg["cls_dropout"] < 1.0:
            self.classifier.add_module(name="dropout", module=nn.Dropout(p=model_cfg["cls_dropout"]))
        self.classifier.add_module(name="fc", module=nn.Linear(in_features=exp_channels, out_features=num_classes))
        # weight init
        self.apply(self.init_parameters)

    def _make_layer(self, input_channel, cfg: Dict) -> Tuple[nn.Sequential, int]:
        block_type = cfg.get("block_type", "mobilevit")
        if block_type.lower() == "mobilevit":  # 如果是mobilevit就构建_make_mit_layer，不是的话就是MV2
            return self._make_mit_layer(input_channel=input_channel, cfg=cfg)
        else:  # 如果不是就构建_make_mobilenet_layer，后续添加模块要修改if结构
            return self._make_mobilenet_layer(input_channel=input_channel, cfg=cfg)

    @staticmethod
    def _make_mobilenet_layer(input_channel: int, cfg: Dict) -> Tuple[nn.Sequential, int]:
        output_channels = cfg.get("out_channels")  # 获取输出通道
        num_blocks = cfg.get("num_blocks", 2)  # num_blocks为多少就构建多少MV2结构
        expand_ratio = cfg.get("expand_ratio", 4)  #
        block = []

        for i in range(num_blocks):  # num_blocks为多少就构建多少MV2结构
            stride = cfg.get("stride", 1) if i == 0 else 1  # 遍历第一次的时候stride就是传入的stride，剩下的stride都是默认为1

            layer = GhostBottleneck(
                in_channels=input_channel,
                mid_channels=input_channel * 2,
                out_channels=output_channels,
                stride=stride,
                kernel_size=3,
                use_se=False
                # expand_ratio=expand_ratio
            )
            block.append(layer)
            input_channel = output_channels

        return nn.Sequential(*block), input_channel

    @staticmethod
    def _make_mit_layer(input_channel: int, cfg: Dict) -> [nn.Sequential, int]:
        stride = cfg.get("stride", 1)
        block = []
        i = 0
        if stride == 2:  # 如果stride为2的话就构建一个下采样结构
            i = i + 1
            layer = GhostBottleneck(
                in_channels=input_channel,
                out_channels=cfg.get("out_channels"),
                stride=stride,
                mid_channels=input_channel * 2,
                kernel_size=3,
                use_se=True,
                # se_kernel_size=7*(5-i)
                se_kernel_size=3
                # expand_ratio=cfg.get("mv_expand_ratio", 4)
            )
            block.append(layer)
            input_channel = cfg.get("out_channels")

        transformer_dim = cfg["transformer_channels"]
        ffn_dim = cfg.get("ffn_dim")
        num_heads = cfg.get("num_heads", 4)
        head_dim = transformer_dim // num_heads

        if transformer_dim % head_dim != 0:
            raise ValueError("Transformer input dimension should be divisible by head dimension. "
                             "Got {} and {}.".format(transformer_dim, head_dim))

        block.append(MobileViTBlock(
            in_channels=input_channel,
            transformer_dim=transformer_dim,
            ffn_dim=ffn_dim,
            n_transformer_blocks=cfg.get("transformer_blocks", 1),
            patch_h=cfg.get("patch_h", 2),
            patch_w=cfg.get("patch_w", 2),
            dropout=cfg.get("dropout", 0.1),
            ffn_dropout=cfg.get("ffn_dropout", 0.0),
            attn_dropout=cfg.get("attn_dropout", 0.1),
            head_dim=head_dim,
            conv_ksize=3
        ))

        return nn.Sequential(*block), input_channel

    @staticmethod
    def init_parameters(m):
        if isinstance(m, nn.Conv2d):
            if m.weight is not None:
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Linear,)):
            if m.weight is not None:
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        else:
            pass

    def forward(self, x: Tensor) -> Tensor:
        # print(x.shape)
        w1 = torch.exp(self.a[0]) / torch.sum(torch.exp(self.a))
        w2 = torch.exp(self.a[1]) / torch.sum(torch.exp(self.a))
        conv1_output = self.conv_1(x)
        layer_1 = self.layer_1(conv1_output)
        layer_2 = self.layer_2(layer_1)
        layer_3 = self.layer_3(layer_2)
        layer_4 = self.layer_4(layer_3)

        adjust = self.conv_adjust(conv1_output)

        branch1 = self.conv_branch_1(adjust)
        add_1 = torch.add(layer_2, branch1)

        branch_2 = self.conv_branch_2(adjust)
        add_3 = torch.add(layer_3, branch_2)

        branch_3 = self.conv_branch_3_1(adjust)
        branch_3 = self.conv_branch_3_2(branch_3)
        add_5 = torch.add(branch_3, layer_4)

        attention1, attention3_1 = self.MyAttention1(add_1, add_3)
        attention3_2, attention5 = self.MyAttention2(add_3, add_5)

        adjust1 = self.conv_adjust1(attention1)
        after_adjust1 = self.after_adjust1(adjust1)
        # print('attention3_1',attention3_1.shape)
        # print('attention3_2', attention3_2.shape)

        add_attention = torch.add(attention3_1 * w1, attention3_2 * w2)
        adjust2 = self.conv_adjust2(add_attention)
        after_adjust2 = self.after_adjust2(adjust2)

        add = torch.add(after_adjust1, after_adjust2)
        add = torch.add(add, attention5)

        x = self.conv_1x1_exp(add)
        x = self.classifier(x)
        return x


def CMViT_Config(num_classes: int = 2):
    config = get_config("CMViT_Config")
    m = CMViT(config, num_classes=num_classes)
    return m
