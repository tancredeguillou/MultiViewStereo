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
        strides = [1, 1, 2, 1, 1, 2, 1, 1]
        layers_conv = [nn.Conv2d(input_size if i==0 else output_sizes[i-1],
            output_sizes[i], kernel_size=kernel_sizes[i],
            stride=strides[i]) for i in range(self.num_layers)]
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
        self.num_conv_layers = 3
        self.num_transpose_layers = 2
        self.num_total_layers = self.num_conv_layers + self.num_transpose_layers
        input_size = G # initial number of input channels
        output_sizes = [8, 16, 32, 16, 8]
        kernel_sizes = [3, 3, 3, 3, 3]
        strides = [1, 2, 2, 2, 2]
        layers_conv = [nn.Conv2d(input_size if i==0 else output_sizes[i-1],
            output_sizes[i], kernel_size=kernel_sizes[i],
            stride=strides[i]) for i in range(self.num_conv_layers)]
        layers_transpose = [nn.ConvTranspose2d(output_sizes[i-1] if i == self.num_conv_layers else output_sizes[i-1]+output_sizes[i-3],
            output_sizes[i], kernel_size=kernel_sizes[i],
            stride=strides[i]) for i in range(self.num_conv_layers, self.num_total_layers)]

        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_transpose = nn.ModuleList(layers_transpose)

        self.relu = nn.ReLU(True)

        self.final_layer = nn.Sequential(
            nn.Conv2d(output_sizes[-1] + output_sizes[0], 1, 3)
        )

    def forward(self, x):
        # x: [B,G,D,H,W]
        # out: [B,D,H,W]
        # TODO
        B,G,D,H,W = x.size()
        x = x.transpose(1, 2).reshape(B*D, G, H, W)

        c0 = self.relu(self.layers_conv[0](x))
        c1 = self.relu(self.layers_conv[1](c0))
        c2 = self.relu(self.layers_conv[2](c1))

        c3 = self.layers_transpose[0](c2)
        c4 = self.layers_transpose[1](c3 + c1)

        out = self.final_layer(c4 + c0)
        return out.view(B, D, H, W)


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
        y, x = torch.meshgrid([torch.arange(0, H, dtype=torch.float32, device=src_fea.device), torch.arange(0, W, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(H * W), x.view(H * W)
        # TODO
        # The 2D coordinates for each pixels are x and y
        # We need to lift this with the depth values
        ref_3D = torch.stack((x, y, torch.ones_like(y)))  # [3, H*W]
        # Repeat the size to have B tensors of ref_3D
        ref_3D = torch.unsqueeze(ref_3D, 0).repeat(B, 1, 1) # [B, 3, H*W]
        # Apply the rotation to ref_3D
        rot_ref_3D = torch.matmul(rot, ref_3D.double()).unsqueeze(2).repeat(1, 1, D, 1) # [B, 3, D, H*W]
        # Expand depth_values to multiply it with rot_ref_3D
        exp_depth_values = depth_values.unsqueeze(2).repeat(1, 1, H * W).view(B, 1, D, H * W)
        # Multiply with the depth values
        rot_depth_ref_3D = rot_ref_3D * exp_depth_values    # [B, 3, D, H*W]
        # Add the translation
        trans_ref_3D = rot_depth_ref_3D + trans.view(B, 3, 1, 1)    # [B, 3, D, H*W]
        # Now take the depth feature out again
        trans_ref_2D = trans_ref_3D[:, :2, :, :] / trans_ref_3D[:, 2, :, :]   # [B, 2, D, H*W]
        # We need to normalize x and y values over height and width
        trans_x_norm = trans_ref_2D[:, 0, :, :] / ((W - 1) / 2) - 1     # [B, D, H*W]
        trans_y_norm = trans_ref_2D[:, 1, :, :] / ((H - 1) / 2) - 1     # [B, D, H*W]
        norm_2D = torch.stack((trans_x_norm, trans_y_norm), dim=3).float()   # [B, D, H*W, 2]

    # get warped_src_fea with bilinear interpolation (use 'grid_sample' function from pytorch)
    # TODO
    warped_src_fea = F.grid_sample(src_fea, norm_2D.view(B, D * H, W, 2))

    return warped_src_fea.view(B, C, D, H, W)

def group_wise_correlation(ref_fea, warped_src_fea, G):
    # ref_fea: [B,C,H,W]
    # warped_src_fea: [B,C,D,H,W]
    # out: [B,G,D,H,W]
    B,C,D,H,W = warped_src_fea.size()
    div_ref = ref_fea.view(B, G, C // G, 1, H, W) # [B, G, C//G, 1, H, W]
    div_warped = warped_src_fea.view(B, G, C // G, D, H, W) # [B, G, C//G, D, H, W]
    return (div_warped * div_ref).mean(2) # [B, G, D, H, W]


def depth_regression(p, depth_values):
    # p: probability volume [B, D, H, W]
    # depth_values: discrete depth values [B, D]
    # TODO
    # We sum over the D dimension i.e dim=1
    B,D = depth_values.size()
    sum = torch.sum(depth_values.view((B, 1, 1)) * p, dim=1) # [B, H, W]
    return sum.unsqueeze(1) # [B, 1, H, W]

def mvs_loss(depth_est, depth_gt, mask):
    # depth_est: [B,1,H,W]
    # depth_gt: [B,1,H,W]
    # mask: [B,1,H,W]
    # TODO
    mask_est = depth_est[mask]
    mask_gt = depth_gt[mask]
    return F.l1_loss(mask_est, mask_gt)
