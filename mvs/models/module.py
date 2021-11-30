import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        # TODO
        self.num_layers = 8
        input_size = 3 # initial number of input channels
        output_sizes = [8, 8, 16, 16, 16, 32, 32, 32]
        kernel_sizes = [3, 3, 5, 3, 3, 5, 3, 3]
        strides = [1, 1, 2, 1, 1, 2, 1]
        layers_conv = [nn.Conv2d(input_size if i==0 else output_sizes[i-1], output_sizes[i], kernel_size=kernel_sizes[i], stride=strides[i]) for i in range(self.num_layers)]
        layers_bn = [nn.BatchNorm2d(output_sizes[i]) for i in range(self.num_layers)]

        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)

        self.relu = nn.ReLU(True)

        self.final_layer = nn.Sequential(
            nn.Conv2d(output_sizes[-1], 32, 3)
        )

    def forward(self, x):
        # x: [B,3,H,W]
        # TODO
        for i in range(self.num_layers):
            x = self.layers_conv[i](x)
            x = self.layers_bn[i](x)
            x = self.relu(x)
        out = self.final_layer(x)
        return out


class SimlarityRegNet(nn.Module):
    def __init__(self, G):
        super(SimlarityRegNet, self).__init__()
        # TODO

    def forward(self, x):
        # x: [B,G,D,H,W]
        # out: [B,D,H,W]
        # TODO


def warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, D]
    # out: [B, C, D, H, W]
    B,C,H,W = src_fea.size()
    D = depth_values.size(1)
    # compute the warped positions with depth values
    with torch.no_grad():
        # relative transformation from reference to source view
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]
        y, x = torch.meshgrid([torch.arange(0, H, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, W, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(H * W), x.view(H * W)
        # TODO
        # The 2D coordinates for each pixels are x and y
        # We need to lift this with the depth values
        ref_3D = [x, y, depth_values]
        transformed_3D = ref_3D @ rot + trans
        src_p = src_proj @ transformed_3D
        # At the projection location, we can take the feature out
        feature = src_p[:, :3]

    # get warped_src_fea with bilinear interpolation (use 'grid_sample' function from pytorch)
    # TODO
    warped_src_fea = F.grid_sample(src_fea, feature)

    return warped_src_fea

def group_wise_correlation(ref_fea, warped_src_fea, G):
    # ref_fea: [B,C,H,W]
    # warped_src_fea: [B,C,D,H,W]
    # out: [B,G,D,H,W]
    # TODO


def depth_regression(p, depth_values):
    # p: probability volume [B, D, H, W]
    # depth_values: discrete depth values [B, D]
    # TODO

def mvs_loss(depth_est, depth_gt, mask):
    # depth_est: [B,1,H,W]
    # depth_gt: [B,1,H,W]
    # mask: [B,1,H,W]
    # TODO
