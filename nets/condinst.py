import math
import torch
import torch.nn.functional as F
from torch import nn
from nets import resnet
from nets.common import FrozenBatchNorm2d, FPN
from utils.mask_utils import center_of_mass

default_cfg = {
    'num_cls': 80,
    "backbone": "resnet18",
    "pretrained": True,
    "norm_layer": FrozenBatchNorm2d,
    "reduction": False,
    "fpn_channels": 256,
    "conv_inner_channels": 256,
    "strides": [8, 16, 32, 64, 128],
    "mask_branch_inner_channels": 128,
    "mask_out_channels": 8
}


class Scale(nn.Module):
    def __init__(self, init_val=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(data=init_val), requires_grad=True)

    def forward(self, x):
        return x * self.scale


class CateControllerBranch(nn.Module):
    def __init__(self,
                 num_cls,
                 in_channels,
                 inner_channels,
                 num_conv=4,
                 norm_type=None,
                 mask_branch_num=3,
                 mask_out_channels=8,
                 mask_inner_channels=8,
                 coord_rel=True
                 ):
        super(CateControllerBranch, self).__init__()
        self.bones = list()
        for i in range(num_conv):
            if i == 0:
                self.bones.append(nn.Conv2d(
                    in_channels, inner_channels, 3, 1, 1, bias=norm_type is None
                ))
            else:
                self.bones.append(nn.Conv2d(
                    inner_channels, inner_channels, 3, 1, 1, bias=norm_type is None
                ))
            if norm_type == 'GN':
                self.bones.append(
                    nn.GroupNorm(32, inner_channels)
                )
            self.bones.append(nn.ReLU(inplace=True))
        self.bones = nn.Sequential(*self.bones)
        weight_nums, bias_nums = list(), list()
        for l_i in range(mask_branch_num):
            if l_i == 0:
                if coord_rel:
                    weight_nums.append((mask_out_channels + 2) * mask_inner_channels)
                else:
                    weight_nums.append(mask_inner_channels * mask_inner_channels)
                bias_nums.append(mask_inner_channels)
            elif l_i == mask_branch_num - 1:
                weight_nums.append(mask_inner_channels * 1)
                bias_nums.append(1)
            else:
                weight_nums.append(mask_inner_channels * mask_inner_channels)
                bias_nums.append(mask_inner_channels)
        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)

        self.cate_controller_head = nn.Conv2d(
            inner_channels,
            num_cls + 4 + 1 + self.num_gen_params,
            3, 1, 1
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.cate_controller_head.bias[:num_cls], -math.log((1 - 0.01) / 0.01))

    def forward(self, x, scale):
        x = self.bones(x)
        if scale is not None:
            x = scale(x)
        x = self.cate_controller_head(x)
        return x


class MaskBranch(nn.Module):
    def __init__(self,
                 in_channels,
                 inner_channels,
                 out_channels,
                 fpn_level_num=3,
                 norm='GN',
                 num_conv=4
                 ):
        super(MaskBranch, self).__init__()
        self.refine = nn.ModuleList()
        for i in range(fpn_level_num):
            block = list()
            block.append(nn.Conv2d(in_channels, inner_channels, 3, 1, 1, bias=norm is None))
            if norm == 'GN':
                block.append(nn.GroupNorm(1, inner_channels))
            block.append(nn.ReLU(inplace=True))
            self.refine.append(nn.Sequential(*block))
        tower = list()
        for i in range(num_conv):
            block = list()
            block.append(nn.Conv2d(inner_channels, inner_channels, 3, 1, 1, bias=norm is None))
            if norm == 'GN':
                block.append(nn.GroupNorm(1, inner_channels))
            block.append(nn.ReLU(inplace=True))
            tower.append(nn.Sequential(*block))
        tower.append(nn.Conv2d(inner_channels, out_channels, 1, 1, bias=norm is None))
        if norm == "GN":
            tower.append(nn.GroupNorm(1, out_channels))
        tower.append(nn.ReLU(inplace=True))
        self.tower = nn.Sequential(*tower)

    def forward(self, features):
        x, x_p = None, None
        for i, f in enumerate(features):
            if i == 0:
                x = self.refine[i](features[i])
            else:
                x_p = self.refine[i](features[i])
                target_h, target_w = x.size()[2:]
                x_p = F.interpolate(x_p,
                                    size=(target_h, target_w),
                                    mode="bilinear",
                                    align_corners=True)
                x = x + x_p
        mask_feat = self.tower(x)
        return mask_feat


class DynamicConv(object):
    def __init__(self,
                 weight_nums,
                 bias_nums,
                 coord_rel=True,
                 mask_inner_channels=8,
                 up_sample_factor=2):
        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.coord_rel = coord_rel
        self.mask_inner_channels = mask_inner_channels
        self.up_sample_factor = up_sample_factor

    def parse_weights_bias(self, params_list):
        all_params = torch.cat(params_list, dim=0)
        num_insts = len(all_params)
        layer_num = len(self.weight_nums)
        params_splits = list(all_params.split(self.weight_nums + self.bias_nums, dim=1))
        weight_splits, bias_splits = params_splits[:layer_num], params_splits[layer_num:]

        for l_i in range(layer_num):
            if l_i < layer_num - 1:
                weight_splits[l_i] = weight_splits[l_i].reshape(num_insts * self.mask_inner_channels, -1, 1, 1)
                bias_splits[l_i] = bias_splits[l_i].reshape(num_insts * self.mask_inner_channels)
            else:
                weight_splits[l_i] = weight_splits[l_i].reshape(num_insts * 1, -1, 1, 1)
                bias_splits[l_i] = bias_splits[l_i].reshape(num_insts)

        return weight_splits, bias_splits

    def __call__(self, mask_feature, params_list):
        """
        :param mask_feature: [batch,channel,h,w]
        :param params_list:[[num_instance,conv_params],[]]
        :return:
        """
        weight_splits, bias_splits = self.parse_weights_bias(params_list)
        _, _, h, w = mask_feature.size()
        batch_instance_num = [len(param) for param in params_list]
        img_ids = sum([[i] * j for i, j in zip(range(len(params_list)), batch_instance_num)], [])
        mask_feature_inp = mask_feature[img_ids, ...]
        if self.coord_rel:
            x_range = torch.linspace(-1, 1, w, device=mask_feature.device, dtype=mask_feature.dtype)
            y_range = torch.linspace(-1, 1, h, device=mask_feature.device, dtype=mask_feature.dtype)
            y, x = torch.meshgrid(y_range, x_range)
            y = y.expand([len(img_ids), 1, -1, -1])
            x = x.expand([len(img_ids), 1, -1, -1])
            mask_feature_inp = torch.cat([mask_feature_inp, x, y], dim=1)
        mask_feature_inp = mask_feature_inp.reshape(1, -1, h, w)
        x = mask_feature_inp
        for i, (weight, bias) in enumerate(zip(weight_splits, bias_splits)):
            x = F.conv2d(
                x, weight, bias=bias, stride=1, padding=0, groups=len(img_ids)
            )
            if i < len(weight_splits) - 1:
                x = F.relu(x)
        mask_logits = F.interpolate(x.reshape(-1, 1, h, w), scale_factor=2, align_corners=True, mode="bilinear")
        return mask_logits


class CondInst(nn.Module):
    def __init__(self, **kwargs):
        super(CondInst, self).__init__()
        self.cfg = {**default_cfg, **kwargs}
        self.backbone = getattr(resnet, self.cfg['backbone'])(
            pretrained=self.cfg['pretrained'],
            reduction=self.cfg['reduction'],
            norm_layer=self.cfg['norm_layer']
        )
        c3, c4, c5 = self.backbone.inner_channels[-3:]
        self.fpn = FPN(c3, c4, c5, out_channel=self.cfg['fpn_channels'])
        self.grids = None
        self.scales = nn.ModuleList([Scale(init_val=1.0) for _ in range(len(self.cfg['strides']))])
        self.cate_control_branch = CateControllerBranch(
            self.cfg['num_cls'],
            self.cfg['fpn_channels'],
            self.cfg['conv_inner_channels'],
            norm_type='GN',
            mask_out_channels=self.cfg['mask_out_channels']
        )

        self.mask_branch = MaskBranch(
            in_channels=self.cfg['fpn_channels'],
            inner_channels=self.cfg['mask_branch_inner_channels'],
            out_channels=self.cfg['mask_out_channels']
        )
        self.dynamic_convs = DynamicConv(
            self.cate_control_branch.weight_nums,
            self.cate_control_branch.bias_nums
        )

    def build_grids(self, feature_maps):
        strides = self.cfg['strides']
        assert len(strides) == len(feature_maps)
        grids = list()
        for s, x in zip(strides, feature_maps):
            _, _, ny, nx = x.shape
            yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
            grid = torch.stack([xv, yv], dim=2)
            grid = (grid + 0.5) * s
            grids.append(grid.to(x.device).view(-1, 2))
        return grids

    def forward(self, x, targets=None):
        xs = self.backbone(x)
        features = self.fpn(xs[-3:])
        if self.grids is None or sum([g.numel() for g in self.grids]) / 2 != sum([f[0][0].numel() for f in features]):
            self.grids = self.build_grids(features)
        predict_list = list()
        for i, f in enumerate(features):
            predict_list.append(self.cate_control_branch(f, self.scales[i]))
        mask_feats = self.mask_branch(features[:3])
        # proposal = [torch.rand((50, 169)), torch.rand((46, 169))]
        # self.dynamic_convs(mask_feats, proposal)

    def compute_loss(self, predict_list, mask_feats, targets):
        pass


if __name__ == '__main__':
    net = CondInst()
    inp = torch.rand(size=(2, 3, 640, 640))
    net(inp)
    # inp = torch.rand(size=(2, 3, 768, 640))
